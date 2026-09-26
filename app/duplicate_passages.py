"""重复口播片段检测核心逻辑（纯标准库，不依赖 FastAPI，便于单测）。

以每个词的 (text, speaker) 组成符号序列，查找两处词序列完全相同的连续
口播片段：两处词索引区间不重叠，且两处首词的 start_ms 至少相隔
min_gap_ms。单词文本不拆分——匹配以整词为最小单位。只返回词数最多的
一对；同长度时按两处起始词索引的二元组 (i, j)（i < j）字典序裁决。
最佳词数不足 min_words 时返回 None。

算法（后缀数组 + LCP + 答案二分，确定性索引算法），
O(n log² n) 时间、O(n) 额外空间（n 为词数，≤ 20000）：

* 符号压缩为整数 id 后用前缀倍增法建后缀数组：每轮按二元组
  (rank[i], rank[i+k]) 排序（k 逐轮翻倍），O(log n) 轮、每轮
  O(n log n)，合计 O(n log² n)；再用 Kasai 算法 O(n) 求出 SA 上
  相邻后缀的 LCP；
* 长度 L 可行 ⟺ 存在起点对 (i, j)（i < j）满足 lcp(i, j) ≥ L、
  j - i ≥ L（区间不重叠）且 start_ms[j] - start_ms[i] ≥ min_gap_ms。
  lcp(i, j) ≥ L 当且仅当两后缀落在 SA 上“相邻 LCP 均 ≥ L”的同一
  极大连续段（run）内；run 内最小、最大词下标的下标差与时间差
  （starts 非降）都不小于任何其他成员对，故该极值对可行即整段可行，
  单趟扫描 O(n) 完成一次判定；
* 可行性随 L 单调，二分求最大可行长度 L*，共 O(n log n)；
* 同长度裁决：对 L* 的每个 run，下标最小者 i 的可行下界也最小
  （j ≥ i + L 且 starts[j] ≥ starts[i] + min_gap_ms 都是“下标 ≥
  某下界”，starts 非降故两个下界可二分合并），它不可行则整段不可行；
  可行时取 run 内满足下界的最小 j，各 run 候选再取 (i, j) 字典序
  最小者，合计 O(n)；
* 全程只开若干 O(n) 整数数组（SA / rank / LCP / starts），不按毫秒
  跨度分配数组；最后校验 L* ≥ min_words，否则返回 None。
"""

from bisect import bisect_left

from .segmentation import validate_words


def _build_suffix_array(symbols):
    """前缀倍增法构建后缀数组，返回 sa（sa[p] 为排名第 p 的后缀起点下标）。"""
    n = len(symbols)
    # 初始排名：按符号值保序压缩为 0..m-1
    order = {v: k for k, v in enumerate(sorted(set(symbols)))}
    rank = [order[s] for s in symbols]
    sa = list(range(n))
    k = 1
    while k < n:
        keys = [(rank[i], rank[i + k] if i + k < n else -1) for i in range(n)]
        sa.sort(key=keys.__getitem__)
        new_rank = [0] * n
        classes = 0
        for p in range(1, n):
            if keys[sa[p]] != keys[sa[p - 1]]:
                classes += 1
            new_rank[sa[p]] = classes
        rank = new_rank
        if classes == n - 1:
            break  # 排名已全唯一，后缀序完全确定
        k *= 2
    return sa


def _build_lcp(symbols, sa):
    """Kasai 算法：lcp[p] 为后缀 sa[p] 与 sa[p+1] 的最长公共前缀长度。"""
    n = len(symbols)
    rank = [0] * n
    for p in range(n):
        rank[sa[p]] = p
    lcp = [0] * (n - 1)
    h = 0
    for i in range(n):
        p = rank[i]
        if p == n - 1:
            h = 0
            continue
        j = sa[p + 1]
        while i + h < n and j + h < n and symbols[i + h] == symbols[j + h]:
            h += 1
        lcp[p] = h
        if h:
            h -= 1
    return lcp


def find_duplicate_passages(words, min_words, min_gap_ms):
    """查找词数最多的重复口播片段对，返回 dict 或 None。

    参数 words 中每个对象需具备属性：start_ms / end_ms / text / speaker
    （沿用同一套词序与时间校验）。

    返回 dict：word_count / text / first / second；first/second 各含
    start_word / end_word（0-based、含端点）与 start_ms / end_ms，
    first 为起始词索引较小的一处；text 为共同文本（词 text 原样拼接，
    不插空格）。不存在词数 ≥ min_words 的合法片段对时返回 None。
    """
    validate_words(words)
    n = len(words)
    starts = [w.start_ms for w in words]
    # 快捷判空：词数不足以容纳两段不重叠的 min_words 片段，
    # 或全序列首末词的时间间隔都达不到 min_gap_ms。
    if n < 2 * min_words or starts[-1] - starts[0] < min_gap_ms:
        return None

    # 整词符号：(text, speaker) 压缩为整数 id
    token_id = {}
    symbols = []
    for w in words:
        key = (w.text, w.speaker)
        idx = token_id.get(key)
        if idx is None:
            idx = len(token_id)
            token_id[key] = idx
        symbols.append(idx)

    sa = _build_suffix_array(symbols)
    lcp = _build_lcp(symbols, sa)

    def feasible(length):
        """是否存在同一 length-run 内下标差与首词时间差均达标的起点对。"""
        run_min = run_max = sa[0]
        for p in range(1, n):
            if lcp[p - 1] >= length:
                idx = sa[p]
                if idx < run_min:
                    run_min = idx
                elif idx > run_max:
                    run_max = idx
                # 极值对支配整段：它可行当且仅当段内存在可行对
                if (
                    run_max - run_min >= length
                    and starts[run_max] - starts[run_min] >= min_gap_ms
                ):
                    return True
            else:
                run_min = run_max = sa[p]
        return False

    # 二分最大可行长度 L*（可行性随长度单调；不重叠两段要求 L ≤ n // 2，
    # 不足 min_words 的长度无需关心）
    lo, hi = min_words, n // 2
    best_len = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            best_len = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if best_len == 0:
        return None

    # 同长度裁决：逐 run 取字典序最小可行对，再取全局最小
    best_pair = None
    run = [sa[0]]

    def settle_run():
        nonlocal best_pair
        if len(run) < 2:
            return
        # 下标最小者的可行下界最小：它不可行则段内其他 i 都不可行
        i = min(run)
        bound = i + best_len
        t = bisect_left(starts, starts[i] + min_gap_ms)
        if t > bound:
            bound = t
        j = min((x for x in run if x >= bound), default=None)
        if j is not None:
            cand = (i, j)
            if best_pair is None or cand < best_pair:
                best_pair = cand

    for p in range(1, n):
        if lcp[p - 1] >= best_len:
            run.append(sa[p])
        else:
            settle_run()
            run = [sa[p]]
    settle_run()

    if best_pair is None:
        return None

    i, j = best_pair
    end_i = i + best_len - 1
    end_j = j + best_len - 1
    return {
        "word_count": best_len,
        "text": "".join(w.text for w in words[i : i + best_len]),
        "first": {
            "start_word": i,
            "end_word": end_i,
            "start_ms": starts[i],
            "end_ms": words[end_i].end_ms,
        },
        "second": {
            "start_word": j,
            "end_word": end_j,
            "start_ms": starts[j],
            "end_ms": words[end_j].end_ms,
        },
    }
