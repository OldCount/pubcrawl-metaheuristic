"""
Pub Crawl Orienteering Solver
Compares three construction heuristics (Greedy, Density+CIL, Lookahead+CIL)
across municipality pub datasets, using multiprocessing for repeated trials.
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import random
import time
import glob
import os
import argparse
import traceback
import multiprocessing
from multiprocessing import Pool, cpu_count
from scipy import stats

BUDGET = 32.0

# Multiprocessing shared state
_distance_matrix = None
_num_pubs = None


def _init_worker(distance_matrix, num_pubs):
    global _distance_matrix, _num_pubs
    _distance_matrix = distance_matrix
    _num_pubs = num_pubs


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def route_distance(route):
    return sum(_distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))


def build_distance_matrix(df):
    n = len(df)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(
                df.iloc[i]["latitude"], df.iloc[i]["longitude"],
                df.iloc[j]["latitude"], df.iloc[j]["longitude"],
            )
            matrix[i][j] = matrix[j][i] = d
    return matrix


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------

def greedy_nearest(start_idx):
    route, visited, total = [start_idx], {start_idx}, 0.0
    while True:
        cur = route[-1]
        best_j, best_d = None, float("inf")
        for j in range(_num_pubs):
            if j not in visited:
                d = _distance_matrix[cur][j]
                if total + d <= BUDGET and d < best_d:
                    best_d, best_j = d, j
        if best_j is None:
            break
        route.append(best_j)
        visited.add(best_j)
        total += best_d
    return route


def density_construction(start_idx, radius=2.0):
    route, visited, budget_left = [start_idx], {start_idx}, BUDGET
    while True:
        cur = route[-1]
        best_next, best_score = None, -float("inf")
        for c in range(_num_pubs):
            if c in visited:
                continue
            d = _distance_matrix[cur][c]
            if d > budget_left:
                continue
            nearby = _distance_matrix[c] < radius
            unvisited = np.array([i not in visited for i in range(_num_pubs)])
            score = np.sum(nearby & unvisited) / (d + 0.1)
            if score > best_score:
                best_score, best_next = score, c
        if best_next is None:
            break
        route.append(best_next)
        visited.add(best_next)
        budget_left -= _distance_matrix[cur][best_next]
    return route


def two_opt(route, max_iter=20):
    best = route[:]
    for _ in range(max_iter):
        improved = False
        for i in range(len(best) - 2):
            for j in range(i + 2, len(best)):
                candidate = best[: i + 1] + best[i + 1 : j + 1][::-1] + best[j + 1 :]
                if route_distance(candidate) <= BUDGET and route_distance(candidate) < route_distance(best):
                    best = candidate
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def cheapest_insertion(route, max_inserts=20, sample_size=30):
    visited = set(route)
    current, dist = route[:], route_distance(route)
    for _ in range(max_inserts):
        unvisited = [i for i in range(_num_pubs) if i not in visited]
        if not unvisited:
            break
        candidates = random.sample(unvisited, min(sample_size, len(unvisited)))
        best_cand, best_pos, best_dist, best_inc = None, None, None, float("inf")
        for cand in candidates:
            for pos in range(1, len(current)):
                old = _distance_matrix[current[pos - 1]][current[pos]]
                new = _distance_matrix[current[pos - 1]][cand] + _distance_matrix[cand][current[pos]]
                inc = new - old
                if dist + inc <= BUDGET and inc < best_inc:
                    best_inc, best_dist, best_cand, best_pos = inc, dist + inc, cand, pos
        if best_cand is None:
            break
        current = current[:best_pos] + [best_cand] + current[best_pos:]
        dist = best_dist
        visited.add(best_cand)
    return current


def lookahead_construction(start_idx, radius=2.0, lookahead_weight=0.3):
    route, visited, budget_left = [start_idx], {start_idx}, BUDGET
    while True:
        cur = route[-1]
        best_next, best_score = None, -float("inf")
        for c in range(_num_pubs):
            if c in visited:
                continue
            d = _distance_matrix[cur][c]
            if d > budget_left:
                continue
            nearby = _distance_matrix[c] < radius
            unvisited = np.array([i not in visited for i in range(_num_pubs)])
            immediate = np.sum(nearby & unvisited)
            neighbors = [i for i in range(_num_pubs)
                         if i not in visited and _distance_matrix[c][i] < radius]
            la_density = 0
            if neighbors:
                for nb in neighbors[:10]:
                    la_density += np.sum((_distance_matrix[nb] < radius) & unvisited)
                la_density /= len(neighbors[:10])
            score = (immediate + lookahead_weight * la_density) / (d + 0.1)
            if score > best_score:
                best_score, best_next = score, c
        if best_next is None:
            break
        route.append(best_next)
        visited.add(best_next)
        budget_left -= _distance_matrix[cur][best_next]
    return route


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def run_trial(args):
    start_idx, trial_num = args
    try:
        result = {"trial": trial_num, "start_idx": start_idx}

        t = time.time()
        r = greedy_nearest(start_idx)
        result.update(greedy_pubs=len(r), greedy_dist=route_distance(r), greedy_time=time.time() - t)

        t = time.time()
        r = two_opt(cheapest_insertion(density_construction(start_idx)))
        result.update(density_cil_pubs=len(r), density_cil_dist=route_distance(r), density_cil_time=time.time() - t)

        t = time.time()
        r = two_opt(cheapest_insertion(lookahead_construction(start_idx), max_inserts=25))
        result.update(lookahead_pubs=len(r), lookahead_dist=route_distance(r), lookahead_time=time.time() - t)

        return result
    except Exception as e:
        print(f"Trial {trial_num} failed: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Analysis & reporting
# ---------------------------------------------------------------------------

METHODS = ["greedy", "density_cil", "lookahead"]


def print_analysis(df, name):
    print(f"\n{'=' * 70}\n  {name.upper()} — {len(df)} trials\n{'=' * 70}")

    print("\nMean performance:")
    for m in METHODS:
        mu = df[f"{m}_pubs"].mean()
        sd = df[f"{m}_pubs"].std()
        lo, hi = df[f"{m}_pubs"].min(), df[f"{m}_pubs"].max()
        t_avg = df[f"{m}_time"].mean()
        print(f"  {m:15s}  {mu:6.2f} ± {sd:5.2f}  (min {lo}, max {hi})  {t_avg:.4f}s")

    g_mean = df["greedy_pubs"].mean()
    print("\nImprovement over greedy:")
    for m in METHODS[1:]:
        delta = df[f"{m}_pubs"].mean() - g_mean
        wr = (df[f"{m}_pubs"] > df["greedy_pubs"]).mean() * 100
        print(f"  {m:15s}  {delta:+6.2f} ({delta / g_mean * 100:+.1f}%)  win rate {wr:.1f}%")

    print("\nBest method frequency:")
    cols = [f"{m}_pubs" for m in METHODS]
    best = df[cols].idxmax(axis=1).str.replace("_pubs", "")
    for m in METHODS:
        print(f"  {m:15s}  {(best == m).mean() * 100:5.1f}%")

    print("\nStatistical tests (paired t-test vs greedy):")
    for m in METHODS[1:]:
        t_stat, p = stats.ttest_rel(df[f"{m}_pubs"], df["greedy_pubs"])
        print(f"  {m:15s}  t={t_stat:7.3f}  p={p:.4e}  {'*' if p < 0.05 else ''}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Pub crawl orienteering benchmark")
    p.add_argument("--pubs-dir", default="pubs", help="Directory containing pub CSV files")
    p.add_argument("--results-dir", default="results", help="Directory for output CSVs")
    p.add_argument("--trials", type=int, default=600, help="Number of trials per municipality")
    p.add_argument("--cores", type=float, default=0.6,
                   help="Fraction of CPU cores to use (0.0–1.0)")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--batch-delay", type=float, default=1.5,
                   help="Seconds to pause between batches (0 for none)")
    p.add_argument("--municipalities", nargs="*", default=None,
                   help="Specific CSV filenames to process (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    n_cores = max(1, int(cpu_count() * args.cores))
    print(f"Budget: {BUDGET} km | Trials: {args.trials} | Cores: {n_cores} | "
          f"Batch: {args.batch_size} (delay {args.batch_delay}s)")

    if args.municipalities:
        csv_files = [os.path.join(args.pubs_dir, f) for f in args.municipalities]
    else:
        csv_files = sorted(glob.glob(os.path.join(args.pubs_dir, "*.csv")))

    print(f"Municipalities: {len(csv_files)}")
    all_summaries = {}

    for csv_path in csv_files:
        name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"\n--- {name} ---")

        df = pd.read_csv(csv_path)
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
        num_pubs = len(df)

        if num_pubs < 10:
            print(f"Skipping ({num_pubs} pubs)")
            continue

        print(f"{num_pubs} pubs — building distance matrix…")
        dm = build_distance_matrix(df)

        results, t0 = [], time.time()
        n_batches = (args.trials + args.batch_size - 1) // args.batch_size
        out_path = os.path.join(args.results_dir, f"{name}_results.csv")

        try:
            with Pool(n_cores, initializer=_init_worker, initargs=(dm, num_pubs)) as pool:
                for b in range(n_batches):
                    lo = b * args.batch_size
                    hi = min(lo + args.batch_size, args.trials)
                    batch_args = [(random.randint(0, num_pubs - 1), i) for i in range(lo, hi)]
                    batch = [r for r in pool.map(run_trial, batch_args) if r]
                    results.extend(batch)

                    done = len(results)
                    elapsed = time.time() - t0
                    remaining = (elapsed / done) * (args.trials - done) / 60 if done else 0
                    print(f"  [{done}/{args.trials}] {elapsed:.0f}s elapsed, ~{remaining:.1f}m left",
                          flush=True)

                    if done % (args.batch_size * 20) == 0 and results:
                        pd.DataFrame(results).to_csv(out_path, index=False)

                    if args.batch_delay > 0 and b < n_batches - 1:
                        time.sleep(args.batch_delay)

        except KeyboardInterrupt:
            print("\nInterrupted — saving partial results.")

        if results:
            rdf = pd.DataFrame(results)
            rdf.to_csv(out_path, index=False)
            print(f"Saved {len(results)} trials → {out_path}")
            print_analysis(rdf, name)

            g = rdf["greedy_pubs"].mean()
            all_summaries[name] = {
                "num_pubs": num_pubs,
                "greedy": g,
                "density_cil": rdf["density_cil_pubs"].mean(),
                "lookahead": rdf["lookahead_pubs"].mean(),
                "lookahead_win%": (rdf["lookahead_pubs"] > rdf["greedy_pubs"]).mean() * 100,
            }

    if len(all_summaries) > 1:
        print(f"\n{'=' * 70}\n  CROSS-MUNICIPALITY SUMMARY\n{'=' * 70}")
        sdf = pd.DataFrame(all_summaries).T.sort_values("num_pubs")
        for mun, row in sdf.iterrows():
            delta = row["lookahead"] - row["greedy"]
            print(f"  {mun:25s} ({row['num_pubs']:3.0f} pubs)  "
                  f"{delta:+.2f} ({delta / row['greedy'] * 100:+.1f}%)  "
                  f"win {row['lookahead_win%']:.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    main()
