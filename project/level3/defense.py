'''Proposal defense methods (Sec 4): cosine + L2 detection, trimmed mean, Multi-Krum.'''

import torch

from shared.utils import fedavg, state_dict_to_vector


def update_deltas(global_state, client_states):
    deltas = []
    for state in client_states:
        delta = {key: state[key].float() - global_state[key].float().cpu() for key in global_state}
        deltas.append(delta)
    return deltas


def cosine_consensus_scores(deltas):
    vecs = [state_dict_to_vector(d) for d in deltas]
    norms = [v.norm().clamp(min=1e-12) for v in vecs]
    n = len(vecs)
    scores = []
    for i in range(n):
        sims = [
            torch.dot(vecs[i], vecs[j]) / (norms[i] * norms[j])
            for j in range(n) if i != j
        ]
        scores.append(torch.stack(sims).mean().item() if sims else 0.0)
    return scores


def l2_norm_scores(deltas):
    return [state_dict_to_vector(d).norm().item() for d in deltas]


def detect_malicious(global_state, client_states, max_flag=1, method="cosine_l2"):
    '''Flag suspicious clients. method: cosine_l2 | cosine | l2 | multi_krum.'''
    deltas = update_deltas(global_state, client_states)
    cos_scores = cosine_consensus_scores(deltas)
    norms = l2_norm_scores(deltas)
    med = sorted(norms)[len(norms) // 2]
    n = len(client_states)

    if method == "cosine":
        order = sorted(range(n), key=lambda i: cos_scores[i])
    elif method == "l2":
        order = sorted(range(n), key=lambda i: -abs(norms[i] - med))
    elif method == "multi_krum":
        # Score = sum of distances to nearest (n - f - 2) neighbors; high = bad.
        f = max_flag
        vecs = [state_dict_to_vector(d) for d in deltas]
        scores = []
        for i in range(n):
            dists = sorted((vecs[i] - vecs[j]).norm().item() for j in range(n) if i != j)
            keep = max(1, n - f - 2)
            scores.append(sum(dists[:keep]))
        order = sorted(range(n), key=lambda i: -scores[i])
        details = {"cosine": cos_scores, "l2": norms, "krum": scores, "flagged": order[:max_flag]}
        return order[:max_flag], details
    else:
        # cosine_l2 (default): weak cosine, L2 distance from median as secondary
        suspicion = []
        for i, (c, nrm) in enumerate(zip(cos_scores, norms)):
            rel = abs(nrm - med) / max(med, 1e-12)
            suspicion.append((c - 0.1 * rel, i))
        suspicion.sort()
        order = [i for _, i in suspicion]

    flagged = order[:max_flag]
    return flagged, {"cosine": cos_scores, "l2": norms, "flagged": flagged}


def trimmed_mean(states, trim_ratio=0.1):
    n = len(states)
    if n == 0:
        raise ValueError("no states to aggregate")
    if n < 3:
        return fedavg(states)
    k = max(0, int(n * trim_ratio))
    if 2 * k >= n:
        k = max(0, (n - 1) // 2)
    avg = {}
    for key in states[0]:
        stacked = torch.stack([s[key].float().reshape(-1) for s in states], dim=0)
        if k > 0:
            stacked, _ = torch.sort(stacked, dim=0)
            stacked = stacked[k:n - k]
        avg[key] = stacked.mean(dim=0).reshape_as(states[0][key]).to(dtype=states[0][key].dtype)
    return avg


def multi_krum_aggregate(global_state, client_states, f=1, m=None):
    '''Select m most consistent updates (Multi-Krum) then average them.'''
    n = len(client_states)
    if m is None:
        m = max(1, n - f)
    deltas = update_deltas(global_state, client_states)
    vecs = [state_dict_to_vector(d) for d in deltas]
    scores = []
    for i in range(n):
        dists = sorted((vecs[i] - vecs[j]).norm().item() for j in range(n) if i != j)
        keep = max(1, n - f - 2)
        scores.append(sum(dists[:keep]))
    chosen = sorted(range(n), key=lambda i: scores[i])[:m]
    return fedavg([client_states[i] for i in chosen]), chosen, scores


def robust_aggregate(
    global_state,
    client_states,
    weights=None,
    max_flag=1,
    detect="cosine_l2",
    aggregate="trimmed_mean",
    trim_ratio=0.1,
):
    '''
    detect: cosine_l2 | cosine | l2 | multi_krum
    aggregate: trimmed_mean | fedavg | multi_krum
    '''
    if aggregate == "multi_krum":
        agg, chosen, scores = multi_krum_aggregate(global_state, client_states, f=max_flag)
        flagged = [i for i in range(len(client_states)) if i not in set(chosen)]
        return agg, {"flagged": flagged[:max_flag], "kept": chosen, "krum": scores}

    flagged, details = detect_malicious(global_state, client_states, max_flag=max_flag, method=detect)
    keep = [i for i in range(len(client_states)) if i not in set(flagged)]
    if not keep:
        keep = list(range(len(client_states)))
    kept_states = [client_states[i] for i in keep]
    kept_weights = None if weights is None else [weights[i] for i in keep]

    if aggregate == "fedavg":
        agg = fedavg(kept_states, kept_weights)
    else:
        agg = trimmed_mean(kept_states, trim_ratio=trim_ratio)
    details["kept"] = keep
    return agg, details
