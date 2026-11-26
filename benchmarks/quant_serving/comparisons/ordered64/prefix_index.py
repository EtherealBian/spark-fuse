"""Exact all-pairs confirmed prefix lengths, without hashing or retaining state across tenants."""


def lcp(x, y):
    return next((i for i, (a, b) in enumerate(zip(x, y)) if a != b), min(len(x), len(y)))


def scan(queries, slots, threshold):
    out = []
    for scope, ids in queries:
        row = []
        for owner, seed in slots:
            n = lcp(ids, seed) if scope == owner else 0
            row.append(n if n >= threshold else 0)
        out.append(row)
    return out


def ordered(queries, slots, threshold):
    """Internal API: validated immutable (scope, token-tuple) inputs; only eligible slots.

    In lexicographic order the LCP of any two token sequences is the minimum LCP on
    the intervening adjacent edges. Walk outward from each query, stopping as soon as
    that minimum drops below threshold. Equal sequences / proper prefixes need no sentinel.
    """
    out = [[0] * len(slots) for _ in queries]
    groups = {}
    owners = set()
    for j, (scope, seq) in enumerate(slots):
        groups.setdefault(scope, []).append((seq, False, j))
        owners.add(scope)
    active = set()
    for i, (scope, seq) in enumerate(queries):
        if scope in owners:
            groups[scope].append((seq, True, i))
            active.add(scope)
    for scope in active:
        nodes = groups[scope]
        nodes.sort(key=lambda item: item[0])
        size = len(nodes)
        edges = [0] + [lcp(nodes[k - 1][0], nodes[k][0]) for k in range(1, size)]
        for k, (seq, is_query, i) in enumerate(nodes):
            if not is_query:
                continue
            row = out[i]
            gain = len(seq)
            for j in range(k - 1, -1, -1):
                gain = min(gain, edges[j + 1])
                if gain < threshold:
                    break
                _, is_other, idx = nodes[j]
                if not is_other:
                    row[idx] = gain
            gain = len(seq)
            for j in range(k + 1, size):
                gain = min(gain, edges[j])
                if gain < threshold:
                    break
                _, is_other, idx = nodes[j]
                if not is_other:
                    row[idx] = gain
    return out
