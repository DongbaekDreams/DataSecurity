'''Proposal Sec 4 defense: combined anomaly detection + robust aggregation.

Detection combines cosine consensus (sign-flip), L2 vs median (gaussian/scale),
and Multi-Krum distance (stealthier / label-flip). Whichever signal is strongest
flags the client. Aggregation excludes flagged clients then applies trimmed mean
(with Multi-Krum available as an alternate aggregator).
'''

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


def multi_krum_scores(deltas, f=1):
    '''Higher = more outlier-like (sum of distances to nearest n-f-2 neighbors).'''
    n = len(deltas)
    vecs = [state_dict_to_vector(d) for d in deltas]
    scores = []
    keep = max(1, n - f - 2)
    for i in range(n):
        dists = sorted((vecs[i] - vecs[j]).norm().item() for j in range(n) if i != j)
        scores.append(sum(dists[:keep]))
    return scores


def _minmax(xs):
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return [0.0] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]


def detect_malicious(global_state, client_states, max_flag=1, method="combined"):
    '''
    method:
      combined   — hard outlier rules + max(cosine, L2, Multi-Krum)  [default]
      cosine     — directional only
      l2         — magnitude only
      multi_krum — distance-to-neighbors only
      cosine_l2  — cosine + L2 (no Krum)
    '''
    deltas = update_deltas(global_state, client_states)
    cos_scores = cosine_consensus_scores(deltas)
    norms = l2_norm_scores(deltas)
    krum_scores = multi_krum_scores(deltas, f=max_flag)
    med = sorted(norms)[len(norms) // 2]
    n = len(client_states)

    # Anomaly directions: larger = more suspicious
    cos_anom = [1.0 - c for c in cos_scores]
    l2_anom = [abs(nrm - med) / max(med, 1e-12) for nrm in norms]
    krum_anom = list(krum_scores)

    cos_n = _minmax(cos_anom)
    l2_n = _minmax(l2_anom)
    krum_n = _minmax(krum_anom)

    if method == "cosine":
        anom = cos_n
    elif method == "l2":
        anom = l2_n
    elif method == "multi_krum":
        anom = krum_n
    elif method == "cosine_l2":
        anom = [max(c, l) for c, l in zip(cos_n, l2_n)]
    else:
        # combined: any strong signal can flag (covers all proposal attack types)
        anom = [max(c, l, k) for c, l, k in zip(cos_n, l2_n, krum_n)]

    # Hard rules so extreme magnitude / inverted updates cannot hide after a
    # poisoned global makes relative rankings noisy (critical for scale/gaussian).
    hard = []
    for i in range(n):
        if norms[i] > 3.0 * max(med, 1e-12):
            hard.append(i)
        elif cos_scores[i] < 0.0:
            hard.append(i)
    if hard:
        # Prefer the most extreme hard outlier(s)
        hard = sorted(hard, key=lambda i: -anom[i])
        flagged = hard[:max_flag]
    else:
        order = sorted(range(n), key=lambda i: -anom[i])
        flagged = order[:max_flag]

    details = {
        "cosine": cos_scores,
        "l2": norms,
        "krum": krum_scores,
        "anomaly": anom,
        "hard": hard,
        "flagged": flagged,
    }
    return flagged, details


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
    n = len(client_states)
    if m is None:
        m = max(1, n - f)
    deltas = update_deltas(global_state, client_states)
    scores = multi_krum_scores(deltas, f=f)
    chosen = sorted(range(n), key=lambda i: scores[i])[:m]
    return fedavg([client_states[i] for i in chosen]), chosen, scores


def robust_aggregate(
    global_state,
    client_states,
    weights=None,
    max_flag=1,
    detect="combined",
    aggregate="trimmed_mean",
    trim_ratio=0.1,
):
    if aggregate == "multi_krum":
        agg, chosen, scores = multi_krum_aggregate(global_state, client_states, f=max_flag)
        flagged = [i for i in range(len(client_states)) if i not in set(chosen)]
        return agg, {"flagged": flagged[:max_flag], "kept": chosen, "krum": scores}

    flagged, details = detect_malicious(
        global_state, client_states, max_flag=max_flag, method=detect,
    )
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
