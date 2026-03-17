"""
Route Visualization — generates a Folium map comparing the three algorithms
on a single municipality.
"""

import argparse
import pandas as pd
import numpy as np
import folium
from folium import plugins
from math import radians, sin, cos, sqrt, atan2
import random

BUDGET = 32.0
COLORS = {
    "lookahead": "#2E86AB",
    "density": "#A23B72",
    "greedy": "#8B8B8B",
    "shared": "#FFB627",
}


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def route_distance(route, dm):
    return sum(dm[route[i]][route[i + 1]] for i in range(len(route) - 1))


def build_distance_matrix(df):
    n = len(df)
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(
                df.iloc[i]["latitude"], df.iloc[i]["longitude"],
                df.iloc[j]["latitude"], df.iloc[j]["longitude"],
            )
            dm[i][j] = dm[j][i] = d
    return dm


# ---------------------------------------------------------------------------
# Algorithms (non-global-state versions for single-run visualization)
# ---------------------------------------------------------------------------

def greedy_nearest(start, dm, n):
    route, visited, total = [start], {start}, 0.0
    while True:
        cur = route[-1]
        best_j, best_d = None, float("inf")
        for j in range(n):
            if j not in visited:
                d = dm[cur][j]
                if total + d <= BUDGET and d < best_d:
                    best_d, best_j = d, j
        if best_j is None:
            break
        route.append(best_j)
        visited.add(best_j)
        total += best_d
    return route


def density_construction(start, dm, n, radius=2.0):
    route, visited, budget_left = [start], {start}, BUDGET
    while True:
        cur = route[-1]
        best_next, best_score = None, -float("inf")
        for c in range(n):
            if c in visited:
                continue
            d = dm[cur][c]
            if d > budget_left:
                continue
            nearby = dm[c] < radius
            unvisited = np.array([i not in visited for i in range(n)])
            score = np.sum(nearby & unvisited) / (d + 0.1)
            if score > best_score:
                best_score, best_next = score, c
        if best_next is None:
            break
        route.append(best_next)
        visited.add(best_next)
        budget_left -= dm[cur][best_next]
    return route


def cheapest_insertion(route, dm, n, max_inserts=50):
    visited = set(route)
    current, dist = route[:], route_distance(route, dm)
    for _ in range(max_inserts):
        unvisited = [i for i in range(n) if i not in visited]
        if not unvisited:
            break
        best_cand, best_pos, best_dist, best_inc = None, None, None, float("inf")
        for cand in unvisited:
            for pos in range(1, len(current)):
                old = dm[current[pos - 1]][current[pos]]
                new = dm[current[pos - 1]][cand] + dm[cand][current[pos]]
                inc = new - old
                if dist + inc <= BUDGET and inc < best_inc:
                    best_inc, best_dist, best_cand, best_pos = inc, dist + inc, cand, pos
        if best_cand is None:
            break
        current = current[:best_pos] + [best_cand] + current[best_pos:]
        dist = best_dist
        visited.add(best_cand)
    return current


def lookahead_construction(start, dm, n, radius=2.0, depth=3):
    route, visited, budget_left = [start], {start}, BUDGET
    while True:
        cur = route[-1]
        best_next, best_score = None, -float("inf")
        for c in range(n):
            if c in visited:
                continue
            d = dm[cur][c]
            if d > budget_left:
                continue
            nearby = dm[c] < radius
            unvisited = np.array([i not in visited for i in range(n)])
            immediate = np.sum(nearby & unvisited)
            neighbors = [i for i in range(n)
                         if i not in visited and dm[c][i] < radius and i != c]
            if neighbors:
                nb_densities = [np.sum((dm[nb] < radius) & unvisited) for nb in neighbors[:depth]]
                avg_nb = np.mean(nb_densities)
                penalty = 0.6 if (immediate > 5 and avg_nb < immediate * 0.5) else 1.0
            else:
                avg_nb, penalty = 0, 0.7
            score = (immediate * penalty + avg_nb * 0.3) / (d + 0.1)
            if score > best_score:
                best_score, best_next = score, c
        if best_next is None:
            break
        route.append(best_next)
        visited.add(best_next)
        budget_left -= dm[cur][best_next]
    return route


def density_cil(start, dm, n):
    return cheapest_insertion(density_construction(start, dm, n), dm, n)


def lookahead_cil(start, dm, n):
    return cheapest_insertion(lookahead_construction(start, dm, n), dm, n)


# ---------------------------------------------------------------------------
# Map creation
# ---------------------------------------------------------------------------

def shared_segments(routes):
    usage = {}
    for name, route in routes.items():
        for i in range(len(route) - 1):
            seg = tuple(sorted([route[i], route[i + 1]]))
            usage.setdefault(seg, set()).add(name)
    return usage


def create_map(df, dm, routes, title):
    center = [df["latitude"].mean(), df["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    seg_usage = shared_segments(routes)
    st = {name: {"pubs": len(r), "dist": route_distance(r, dm)} for name, r in routes.items()}

    # Unvisited pubs
    all_visited = set().union(*routes.values())
    layer = folium.FeatureGroup(name="Unvisited Pubs", show=True)
    for idx, row in df.iterrows():
        if idx not in all_visited:
            folium.CircleMarker(
                [row["latitude"], row["longitude"]], radius=5,
                color="#666", fill=True, fillColor="#CCC", fillOpacity=0.6, weight=2,
                popup=f"<b>UNVISITED</b><br>{row['name']}",
            ).add_to(layer)
    layer.add_to(m)

    # Shared path segments
    layer = folium.FeatureGroup(name="Shared Paths", show=True)
    for seg, methods in seg_usage.items():
        if len(methods) >= 2:
            p1, p2 = seg
            folium.PolyLine(
                [[df.iloc[p1]["latitude"], df.iloc[p1]["longitude"]],
                 [df.iloc[p2]["latitude"], df.iloc[p2]["longitude"]]],
                color=COLORS["shared"], weight=6, opacity=0.7,
                tooltip=f"Shared: {', '.join(sorted(methods))}",
            ).add_to(layer)
    layer.add_to(m)

    # Per-algorithm layers
    for method, route in routes.items():
        layer = folium.FeatureGroup(name=f"{method.title()} ({len(route)} pubs)", show=True)
        for i in range(len(route) - 1):
            seg = tuple(sorted([route[i], route[i + 1]]))
            if len(seg_usage[seg]) == 1:
                folium.PolyLine(
                    [[df.iloc[route[i]]["latitude"], df.iloc[route[i]]["longitude"]],
                     [df.iloc[route[i + 1]]["latitude"], df.iloc[route[i + 1]]["longitude"]]],
                    color=COLORS[method], weight=3, opacity=0.6,
                ).add_to(layer)
        for i, pub in enumerate(route):
            lat, lon = df.iloc[pub]["latitude"], df.iloc[pub]["longitude"]
            name = df.iloc[pub]["name"]
            if i == 0 and method == "greedy":
                folium.Marker(
                    [lat, lon], popup=f"<b>START</b><br>{name}",
                    icon=folium.Icon(color="green", icon="star", prefix="fa"),
                ).add_to(m)
            elif i > 0:
                folium.CircleMarker(
                    [lat, lon], radius=6, color="white", fill=True,
                    fillColor=COLORS[method], fillOpacity=0.9, weight=2,
                    popup=f"<b>{method.upper()} #{i}</b><br>{name}",
                ).add_to(layer)
        layer.add_to(m)

    # Legend
    la_imp = st["lookahead"]["pubs"] - st["greedy"]["pubs"]
    dn_imp = st["density"]["pubs"] - st["greedy"]["pubs"]
    legend = f"""
    <div style="position:fixed;top:10px;right:10px;width:320px;background:rgba(255,255,255,.97);
         border:2px solid #333;z-index:9999;font-size:13px;padding:12px;border-radius:8px;
         box-shadow:0 4px 12px rgba(0,0,0,.3);">
      <b style="font-size:16px;">{title}</b><hr style="margin:6px 0;">
      <div style="background:#E8F4F8;padding:8px;border-radius:6px;margin:6px 0;border-left:4px solid {COLORS['lookahead']};">
        <b>Lookahead</b> — {st['lookahead']['pubs']} pubs
        <span style="color:green;">(+{la_imp})</span><br>
        <small>{st['lookahead']['dist']:.2f} km</small>
      </div>
      <div style="background:#F5E8F5;padding:8px;border-radius:6px;margin:6px 0;border-left:4px solid {COLORS['density']};">
        <b>Density+CIL</b> — {st['density']['pubs']} pubs
        <span style="color:{'green' if dn_imp > 0 else 'red'};">({dn_imp:+d})</span><br>
        <small>{st['density']['dist']:.2f} km</small>
      </div>
      <div style="background:#F0F0F0;padding:8px;border-radius:6px;margin:6px 0;border-left:4px solid {COLORS['greedy']};">
        <b>Greedy</b> — {st['greedy']['pubs']} pubs<br>
        <small>{st['greedy']['dist']:.2f} km</small>
      </div>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    plugins.Fullscreen().add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Visualize pub crawl routes on a Folium map")
    p.add_argument("csv", help="Path to municipality CSV (e.g. pubs/Derby.csv)")
    p.add_argument("--output", default=None, help="Output HTML path (default: <city>_comparison.html)")
    p.add_argument("--samples", type=int, default=15, help="Starting points to evaluate")
    p.add_argument("--prefer-bad-greedy", action="store_true", default=True,
                   help="Pick start where greedy does worst relative to others")
    return p.parse_args()


def main():
    args = parse_args()
    city = args.csv.replace(".csv", "").split("/")[-1]

    df = pd.read_csv(args.csv)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    n = len(df)
    print(f"{city}: {n} pubs")

    dm = build_distance_matrix(df)

    best_start, best_score, best_res = None, float("-inf"), None
    for s in random.sample(range(n), min(args.samples, n)):
        g = greedy_nearest(s, dm, n)
        d = density_cil(s, dm, n)
        la = lookahead_cil(s, dm, n)
        score = (len(la) - len(g)) + (len(d) - len(g)) if args.prefer_bad_greedy else len(la) + len(d)
        if score > best_score:
            best_start, best_score, best_res = s, score, (len(g), len(d), len(la))
        print(f"  start {s:3d}: G={len(g)} D={len(d)} L={len(la)}")

    print(f"Selected start #{best_start} — {df.iloc[best_start]['name']}  "
          f"(G={best_res[0]}, D={best_res[1]}, L={best_res[2]})")

    routes = {
        "greedy": greedy_nearest(best_start, dm, n),
        "density": density_cil(best_start, dm, n),
        "lookahead": lookahead_cil(best_start, dm, n),
    }

    out = args.output or f"{city.lower()}_comparison.html"
    m = create_map(df, dm, routes, f"{city} — Algorithm Comparison")
    m.save(out)
    print(f"Map saved → {out}")


if __name__ == "__main__":
    main()
