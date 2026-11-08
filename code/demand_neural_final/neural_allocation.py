"""Exact daily inventory costs and common-fleet integer DP."""
import numpy as np
from numba import njit

@njit(cache=True)
def scenario_costs(paths, weights, caps):
    days, sites, scenarios, hours = paths.shape
    costs = np.full((days, sites, caps.max() + 1), np.inf)
    for d in range(days):
        for i in range(sites):
            cap = caps[i]
            for x in range(cap + 1):
                total = 0.0
                for s in range(scenarios):
                    b = float(x)
                    penalty = 0.0
                    for h in range(hours):
                        z = b + paths[d, i, s, h]
                        u, o = max(-z, 0.0), max(z - cap, 0.0)
                        b = min(float(cap), max(0.0, z))
                        penalty += 6 * u * u + 5 * o + max(b - cap // 2, 0.0)
                    total += weights[d, i, s] * penalty
                costs[d, i, x] = total
    return costs


@njit(cache=True)
def dp_all_totals(cost, caps):
    total = int(caps.sum())
    prev = np.full(total + 1, np.inf)
    prev[0] = 0.0
    max_before = 0
    for i in range(len(caps)):
        nxt = np.full(total + 1, np.inf)
        for n in range(max_before + 1):
            for x in range(caps[i] + 1):
                value = prev[n] + cost[i, x]
                if value < nxt[n + x]:
                    nxt[n + x] = value
        prev = nxt
        max_before += caps[i]
    return prev


@njit(cache=True)
def dp_reconstruct(cost, caps, total):
    prev = np.full(total + 1, np.inf)
    prev[0] = 0.0
    chosen = np.full((len(caps), total + 1), -1, dtype=np.int64)
    max_before = 0
    for i in range(len(caps)):
        nxt = np.full(total + 1, np.inf)
        for n in range(min(max_before, total) + 1):
            for x in range(min(caps[i], total - n) + 1):
                value = prev[n] + cost[i, x]
                if value < nxt[n + x]:
                    nxt[n + x] = value
                    chosen[i, n + x] = x
        prev = nxt
        max_before += caps[i]
    result = np.zeros(len(caps), dtype=np.int64)
    remaining = total
    for i in range(len(caps) - 1, -1, -1):
        result[i] = chosen[i, remaining]
        remaining -= result[i]
    return result


def optimize(costs, caps):
    totals = np.stack([dp_all_totals(day, caps) for day in costs])
    scores = 20 * np.arange(len(totals[0])) + totals.mean(0)
    scores[0] = np.inf
    fleet = int(np.argmin(scores))
    allocation = np.stack([dp_reconstruct(day, caps, fleet) for day in costs])
    if (allocation < 0).any() or (allocation > caps).any() or not np.all(allocation.sum(1) == fleet):
        raise ValueError("Invalid allocation from DP")
    obtained = costs[np.arange(len(costs))[:, None], np.arange(len(caps)), allocation].sum(1)
    if not np.allclose(obtained, totals[:, fleet]):
        raise ValueError("Allocation does not reproduce DP optimum")
    return allocation, float(scores[fleet])


def realized_score(truth, allocation, caps):
    b = allocation.astype(float).copy()
    q = np.zeros(len(truth))
    o = np.zeros(len(truth))
    e = np.zeros(len(truth))
    for hour in range(24):
        z = b + truth[:, :, hour]
        q += (np.maximum(-z, 0) ** 2).sum(1)
        o += np.maximum(z - caps, 0).sum(1)
        b = np.clip(z, 0, caps)
        e += np.maximum(b - caps // 2, 0).sum(1)
    n = allocation.sum(1)
    if not np.all(n == n[0]) or n[0] < 1:
        raise ValueError("Fleet must be positive and fixed over days")
    return {"score": float((20 * n + 6 * q + 5 * o + e).mean()),
            "N": int(n[0]), "Q_daily": float(q.mean()), "O_daily": float(o.mean()),
            "E_daily": float(e.mean())}
