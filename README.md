# Pub Crawl Orienteering Solver

A metaheuristics benchmark for the **orienteering problem** applied to real UK pub locations. Given a travel budget (default 32 km), the goal is to visit as many pubs as possible starting from a random node — every pub has equal value, so it reduces to maximising the number of stops.

Three construction heuristics are compared across repeated trials:

| Method | Description |
|---|---|
| **Greedy** | Nearest-neighbour baseline |
| **Density + CIL** | Density-aware construction → Cheapest Insertion → 2-opt |
| **Lookahead + CIL** | Lookahead density construction → Cheapest Insertion → 2-opt |

## Setup

```bash
pip install -r requirements.txt
```

## Data

Place pub CSVs in the `pubs/` directory. Each CSV needs at minimum the columns `name`, `latitude`, and `longitude`. An example row:

```
fas_id,name,address,postcode,easting,northing,latitude,longitude,local_authority
339814,146 High Street,"Aston, Birmingham",B6 4US,407245,288989.0,52.498758,-1.894707,Birmingham
```

CSVs are gitignored to keep the repo lightweight — source your own from [OpenStreetMap](https://www.openstreetmap.org/) or the [Open Pubs dataset](https://www.getthedata.com/open-pubs).

## Usage

### Benchmark trials

```bash
# Run 600 trials across all CSVs in pubs/, using 60% of CPU cores
python pubcrawl.py

# Customise
python pubcrawl.py --pubs-dir pubs --trials 1000 --cores 0.8 --results-dir results
```

> **Note:** Each municipality takes a while — 600 trials on a dataset with 200+ pubs can run for several hours.

### Route visualisation

```bash
# Generate an interactive Folium map comparing all three algorithms
python route_viz_improved.py pubs/Derby.csv --output derby_comparison.html
```

Opens an HTML map with layer toggles for each algorithm's route, shared path segments, and unvisited pubs.

### CLI reference

**pubcrawl.py**

| Flag | Default | Description |
|---|---|---|
| `--pubs-dir` | `pubs` | Directory containing pub CSVs |
| `--results-dir` | `results` | Output directory for trial CSVs |
| `--trials` | `600` | Trials per municipality |
| `--cores` | `0.6` | Fraction of CPU cores to use |
| `--batch-size` | `10` | Trials per multiprocessing batch |
| `--batch-delay` | `1.5` | Seconds between batches |
| `--municipalities` | all | Specific CSV filenames to process |

**route_viz_improved.py**

| Flag | Default | Description |
|---|---|---|
| `csv` (positional) | — | Path to a municipality CSV |
| `--output` | `<city>_comparison.html` | Output HTML path |
| `--samples` | `15` | Starting points to evaluate |

## Output

`pubcrawl.py` writes per-municipality result CSVs to the results directory and prints a summary table with mean performance, improvement over greedy, win rates, and paired t-tests.

## License

MIT
