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


def scoped(queries, slots, threshold):
    """Ephemeral exact (scope, first k token IDs) buckets, not fingerprints.

    A positive eligible gain implies that both keys are equal. Only intersecting
    groups need an LCP or a weight matrix. Never slice a foreign-scope slot's IDs.
    Return None when every possible assignment is a zero-gain assignment.
    """
    scopes = {scope for scope, _ in queries}
    groups = {}
    for j, (scope, ids) in enumerate(slots):
        if scope in scopes and len(ids) >= threshold:
            groups.setdefault((scope, ids[:threshold]), []).append(j)
    if not groups:
        return None
    hits = {}
    for i, (scope, ids) in enumerate(queries):
        if len(ids) >= threshold:
            key = (scope, ids[:threshold])
            if key in groups:
                hits.setdefault(key, []).append(i)
    if not hits:
        return None
    weights = [[0] * len(slots) for _ in queries]
    for key, rows in hits.items():
        cols = groups[key]
        if len(rows) * len(cols) <= 8:
            for i in rows:
                for j in cols:
                    weights[i][j] = lcp(queries[i][1], slots[j][1])
        else:
            # Prefix ties and global rank assignment semantics remain unchanged.
            block = ordered([queries[i] for i in rows], [slots[j] for j in cols], threshold)
            for i, values in zip(rows, block):
                for j, value in zip(cols, values):
                    weights[i][j] = value
    return weights
