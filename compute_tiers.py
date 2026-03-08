# Partition leaderboard scores (descending) into K=10 tiers using optimal 1-D partitioning (minimize intra-tier variance).
# Output: tiers.json with a list of tier dicts: {tier, start, end, models}

import json
import numpy as np

K = 10
LEADERBOARD_FILE = "leaderboard.json"
TIERS_OUT = "tiers.json"

def prefix_sums(arr):
    n = len(arr)
    ps = np.zeros(n+1, dtype=float)
    ps2 = np.zeros(n+1, dtype=float)
    for i in range(n):
        ps[i+1] = ps[i] + arr[i]
        ps2[i+1] = ps2[i] + arr[i]*arr[i]
    return ps, ps2

def cost_interval(ps, ps2, i, j):
    # cost = sum (x - mean)^2 over items i..j inclusive
    n = j - i + 1
    S = ps[j+1] - ps[i]
    S2 = ps2[j+1] - ps2[i]
    return S2 - (S*S)/n

def optimal_partition(scores, K=10):
    n = len(scores)
    if K >= n:
        return [(i,i) for i in range(n)]
    ps, ps2 = prefix_sums(scores)
    INF = 1e30
    dp = [[INF] * (n+1) for _ in range(K+1)]
    back = [[-1] * (n+1) for _ in range(K+1)]
    dp[0][0] = 0.0
    for k in range(1, K+1):
        for j in range(1, n+1):
            best = INF
            best_i = -1
            # ensure at least k-1 elements in previous groups
            for i in range(k-1, j):
                c = dp[k-1][i] + cost_interval(ps, ps2, i, j-1)
                if c < best:
                    best = c
                    best_i = i
            dp[k][j] = best
            back[k][j] = best_i
    parts = []
    k = K
    j = n
    while k > 0:
        i = back[k][j]
        parts.append((i, j-1))
        j = i
        k -= 1
    parts.reverse()
    return parts

def main():
    with open(LEADERBOARD_FILE, "r", encoding="utf-8") as f:
        models = json.load(f)

    if not isinstance(models, list):
        # support structure {"models": [ ... ]}
        if isinstance(models, dict) and "models" in models:
            models = models["models"]
        else:
            raise ValueError("leaderboard.json must be a list or contain 'models' key")

    # sort descending by score (safety)
    models_sorted = sorted(models, key=lambda x: -float(x.get("score", 0.0)))
    scores = [float(x.get("score", 0.0)) for x in models_sorted]

    parts = optimal_partition(scores, K=K)

    tiers = []
    for t, (i, j) in enumerate(parts):
        tier_models = [models_sorted[idx]["model"] for idx in range(i, j+1)]
        tiers.append({"tier": t, "start": i, "end": j, "models": tier_models})

    with open(TIERS_OUT, "w", encoding="utf-8") as fo:
        json.dump(tiers, fo, indent=2, ensure_ascii=False)

    print(f"Wrote {TIERS_OUT} with {len(tiers)} tiers (K={K})")

if __name__ == "__main__":
    main()