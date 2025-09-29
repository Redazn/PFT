Import numpy as np
from scipy.stats import entropy
import heapq
import itertools
import time

class RubrikHybridOptimizer:
    def __init__(self, matrices, priors=None, coeffs=None, constraint_limit=1.0, beta=0.5):
        """
        matrices: dict of 6 np.array (3x3)
        priors: dict or None -> if None, use automatic entropy-based priors
        coeffs: dict or None -> if None, use automatic mean-based coeffs
        constraint_limit: float - limit for linear constraint
        beta: float - tradeoff used in auto_weights (if priors provided or computed)
        """
        self.matrices = matrices
        self.beta = beta

        # Mask bobot internal 3×3 (consistent, no typos)
        self.weight_mask = np.array([
            [0.9, 0.8, 0.7],
            [0.6, 0.5, 0.4],
            [0.3, 0.2, 0.1]
        ], dtype=float)

        # If priors/coeffs not provided, compute automatically from matrices
        self.priors = priors if priors is not None else self._auto_generate_priors()
        self.coeffs = coeffs if coeffs is not None else self._auto_generate_coeffs()

        # Constraint limit (can be passed manually or left default)
        self.constraint_limit = constraint_limit

    # -------------------
    # Utilities
    # -------------------
    def normalize_matrix(self, M):
        M = np.array(M, dtype=float)
        mn = M.min()
        mx = M.max()
        if mx - mn < 1e-12:
            return np.zeros_like(M)
        return (M - mn) / (mx - mn)

    def compute_entropy(self, M):
        flat = np.array(M, dtype=float).flatten()
        s = flat.sum()
        if s <= 0:
            # degenerate -> entropy 0
            return 0.0
        probs = flat / s
        # small epsilon protection
        probs = np.where(probs <= 0, 1e-12, probs)
        return float(entropy(probs))

    def _auto_generate_priors(self):
        """Priors derived from entropies of normalized matrices (higher entropy -> higher prior)."""
        ent = {}
        for k, M in self.matrices.items():
            M_norm = self.normalize_matrix(M)
            ent[k] = self.compute_entropy(M_norm)
        # softmax-normalize entropies to sum to 1
        vals = np.array(list(ent.values()), dtype=float)
        # if all zeros, fallback to uniform
        if np.allclose(vals, 0):
            keys = list(ent.keys())
            uniform = 1.0 / len(keys)
            return {kk: uniform for kk in keys}
        ex = np.exp(vals - vals.max())
        soft = ex / ex.sum()
        return {k: float(soft[i]) for i, k in enumerate(ent.keys())}

    def _auto_generate_coeffs(self):
        """
        Coeffs derived from mean utility of masked normalized matrices.
        Normalize to sum to 1 (or scaled if you prefer).
        """
        means = {}
        for k, M in self.matrices.items():
            Mn = self.normalize_matrix(M)
            # apply same weight mask to represent positional importance in coeff creation
            masked = Mn * self.weight_mask
            means[k] = float(masked.mean())
        vals = np.array(list(means.values()), dtype=float)
        # if all zeros fallback to uniform
        if np.allclose(vals, 0):
            keys = list(means.keys())
            uniform = 1.0 / len(keys)
            return {kk: uniform for kk in keys}
        # normalize to sum to 1
        norm = vals / vals.sum()
        return {k: float(norm[i]) for i, k in enumerate(means.keys())}

    # -------------------
    # Core functions (unchanged in logic)
    # -------------------
    def auto_weights(self):
        """Combine entropy and prior with beta, then softmax-normalize to global weights."""
        entropies = {k: self.compute_entropy(self.normalize_matrix(M)) for k, M in self.matrices.items()}
        raw = {}
        for k in self.matrices.keys():
            # combine signal (entropy) with prior
            prior = self.priors.get(k, 1.0 / len(self.matrices))
            raw[k] = (self.beta * entropies[k]) + ((1.0 - self.beta) * prior)
        arr = np.array(list(raw.values()), dtype=float)
        # softmax
        ex = np.exp(arr - arr.max())
        soft = ex / ex.sum()
        return {k: float(soft[i]) for i, k in enumerate(raw.keys())}

    def apply_gamma(self, M, gamma):
        """Rotate matrix by gamma in {0,90,180,270}"""
        k = int((gamma // 90) % 4)
        return np.rot90(np.array(M, dtype=float), k)

    def utility(self, M, gamma=0, mode="mean"):
        """Return masked utility (normalized -> masked)"""
        Mn = self.normalize_matrix(M)
        Mrot = self.apply_gamma(Mn, gamma)
        Mmasked = Mrot * self.weight_mask
        if mode == "mean":
            return float(Mmasked.mean())
        elif mode == "max":
            return float(Mmasked.max())
        elif mode == "min":
            return float(Mmasked.min())
        else:
            return float(Mmasked.mean())

    def score_and_constraint(self, weights, gammas):
        """Compute aggregated score and linear constraint value given weights and per-matrix gammas."""
        score = 0.0
        constraint_val = 0.0
        for k, M in self.matrices.items():
            g = gammas.get(k, 0)
            util = self.utility(M, gamma=g)
            w = float(weights.get(k, 0.0))
            c = float(self.coeffs.get(k, 0.0))
            score += w * util
            constraint_val += c * util
        feasible = constraint_val <= self.constraint_limit
        return float(score), float(constraint_val), bool(feasible)

    # ---------- Brute Force ----------
    def run_bruteforce(self):
        weights = self.auto_weights()
        gammas_options = [0, 90, 180, 270]
        keys = list(self.matrices.keys())

        best = {"score": -np.inf, "gammas": None, "constraint": None, "feasible": False}
        # brute force over all combinations (4^n)
        combos = itertools.product(gammas_options, repeat=len(keys))
        for combo in combos:
            gammas = {keys[i]: combo[i] for i in range(len(keys))}
            score, cons, feasible = self.score_and_constraint(weights, gammas)
            if feasible and score > best["score"]:
                best["score"] = score
                best["gammas"] = gammas.copy()
                best["constraint"] = cons
                best["feasible"] = feasible
        return best

    # ---------- A* Search ----------
    def run_astar(self):
        weights = self.auto_weights()
        keys = list(self.matrices.keys())
        gammas_options = [0, 90, 180, 270]

        def heuristic_upper_bound(idx):
            # optimistic upper bound for remaining matrices: assume util = 1.0
            remaining_keys = keys[idx:]
            est = 0.0
            for kk in remaining_keys:
                est += weights[kk] * 1.0
            return est

        # Priority queue: (f_score, idx, current_gammas, score_so_far, cons_so_far)
        pq = []
        start = ( -heuristic_upper_bound(0), 0, {}, 0.0, 0.0 )
        heapq.heappush(pq, start)

        best = {"score": -np.inf, "gammas": None, "constraint": None, "feasible": False}
        seen = {}  # memoization for best score at (idx, frozenset(gammas.items()))

        while pq:
            _, idx, current_gammas, score_so_far, cons_so_far = heapq.heappop(pq)

            # if we've assigned all matrices
            if idx == len(keys):
                if cons_so_far <= self.constraint_limit and score_so_far > best["score"]:
                    best = {"score": score_so_far, "gammas": current_gammas.copy(),
                            "constraint": cons_so_far, "feasible": True}
                continue

            key = keys[idx]
            # expand children
            for g in gammas_options:
                util = self.utility(self.matrices[key], gamma=g)
                new_score = score_so_far + weights[key] * util
                new_cons = cons_so_far + self.coeffs[key] * util
                if new_cons > self.constraint_limit:
                    # prune infeasible branches early
                    continue
                new_gammas = current_gammas.copy()
                new_gammas[key] = g

                memo_key = (idx + 1, tuple(sorted(new_gammas.items())))
                prev_best = seen.get(memo_key, -np.inf)
                if new_score <= prev_best:
                    # we've seen a better or equal partial solution
                    continue
                seen[memo_key] = new_score

                f_score = -(new_score + heuristic_upper_bound(idx + 1))  # negative because we use min-heap
                heapq.heappush(pq, (f_score, idx + 1, new_gammas, new_score, new_cons))

        return best


# -------------------------
# DEMO (no manual priors/coeffs)
# -------------------------
if __name__ == "__main__":
    # Create 6 matrices (3x3) representing the 6 faces of the Rubrik
    # Values in [0.1, 0.9] like your intention (mask uses 0.9..0.1 separately)
    np.random.seed(0)
    matrices = {
        "cost": np.round(np.random.uniform(0.1, 0.9, (3, 3)), 3),
        "capacity": np.round(np.random.uniform(0.1, 0.9, (3, 3)), 3),
        "service": np.round(np.random.uniform(0.1, 0.9, (3, 3)), 3),
        "time": np.round(np.random.uniform(0.1, 0.9, (3, 3)), 3),
        "risk": np.round(np.random.uniform(0.1, 0.9, (3, 3)), 3),
        "co2": np.round(np.random.uniform(0.1, 0.9, (3, 3)), 3)
    }

    # Instantiate optimizer WITHOUT manual priors/coeffs
    # The class will auto-generate priors (from entropy) and coeffs (from masked means)
    opt = RubrikHybridOptimizer(matrices, priors=None, coeffs=None, constraint_limit=0.8, beta=0.5)

    # show what got auto-generated
    print("=== Matrices (6 faces) ===")
    for k, m in matrices.items():
        print(f"\n-- {k} --\n{m}")

    print("\n=== Weight mask (3x3) ===")
    print(opt.weight_mask)

    print("\n=== Auto-generated priors (from entropies, softmax) ===")
    for k, v in opt.priors.items():
        print(f"{k}: {v:.4f}")

    print("\n=== Auto-generated coeffs (from masked means, normalized) ===")
    for k, v in opt.coeffs.items():
        print(f"{k}: {v:.4f}")

    print("\n=== Running bruteforce (exhaustive over gamma combos) ===")
    t0 = time.time()
    best_brute = opt.run_bruteforce()
    t1 = time.time()
    print(f"Brute force time: {t1 - t0:.3f}s")
    print(best_brute)

    print("\n=== Running A* (heuristic search over gamma assignments) ===")
    t0 = time.time()
    best_astar = opt.run_astar()
    t1 = time.time()
    print(f"A* time: {t1 - t0:.3f}s")
    print(best_astar)

    # If both found feasible, print comparison
    if best_brute["feasible"] and best_astar["feasible"]:
        print("\nBoth methods found feasible solutions.")
        print("Brute best score:", best_brute["score"])
        print("A* best score   :", best_astar["score"])
    elif best_brute["feasible"]:
        print("\nOnly brute force found feasible solution.")
    elif best_astar["feasible"]:
        print("\nOnly A* found feasible solution.")
    else:
        print("\nNo feasible solution found under current auto coeffs/limit.")

