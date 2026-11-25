# Eden versus DLA

This program was developed for Seminar I of Computational Simulations II. Its objective is to compare Eden's compact local growth with diffusion-limited aggregation (DLA) on a square lattice. It uses different parameters, such as the number of particles, lattice size, and number of replicas to compare the two growth models. The program generates CSV files with the mass and radius of gyration, PNG figures, and a GIF animation of the growth process.

## Execution

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python run_experiment.py
```

The default command uses a 501 x 501 lattice, 1,500 particles, and 20 replicas. For a
quick verification:

```bash
python run_experiment.py --size 151 --particles 180 --replicas 2
pytest -q
```

The main program displays the replica being processed, its seed, the individual
fractal-dimension fit, and each export step in the terminal. Run it without
arguments to produce the official dataset:

```bash
python run_experiment.py
```

The following figures are examples produced by the default command. They are stored in `../results/seed_20261122/`.

### Growth animation

![Eden versus DLA animation](../results/seed_20261122/eden_vs_dla.gif)

### Animation frames

![Eden and DLA animation frames](../results/seed_20261122/quadros_animacao.png)



## How the code works

The entry point is `run_experiment.py`. It parses the command-line arguments,
creates a `SimulationConfig`, and calls `run_experiment()` for both growth models.

For every replica, the code generates independent seeds and performs one of two
processes:

- **Eden model:** starts with one occupied site at the center of the lattice.
  At each growth step, it randomly selects one empty site from the boundary of
the occupied cluster and adds that site to the cluster.
- **DLA model:** starts with one occupied site at the center. A random walker is
  launched at a radius slightly larger than the current cluster. The walker moves
  randomly across the lattice until it reaches an occupied neighbor or leaves the
  kill radius; if it attaches, the new site becomes part of the cluster.

During growth, snapshots are saved at regular mass values. For each replica, the
program calculates the radius of gyration and estimates the fractal dimension
from the log-log relationship:

```text
log(M) = D log(Rg) + intercept
```

The exponent `D` is the fractal dimension. Finally, the experiment orchestration
combines all replicas and exports the CSV, PNG, and GIF files described below.

## Command-line arguments

Run `python run_experiment.py --help` to display the available options. The
supported arguments are:

| Argument | Type | Default | Description |
|---|---:|---:|---|
| `--replicas` | integer | `20` | Number of independent replicas generated for Eden and DLA. |
| `--particles` | integer | `1500` | Number of occupied particles in each generated cluster. |
| `--size` | integer | `501` | Side length of the square lattice, in lattice sites. |
| `--seed` | integer | `20261122` | Base random seed. Replica seeds are derived from this value. |
| `--output` | path | `results` | Directory where the seed-specific output subfolder is created. |

For example:

```bash
python run_experiment.py --size 151 --particles 180 --replicas 2 --seed 20261125 --output results
```

This command runs two replicas of each model on a 151 x 151 lattice, with 180
particles per cluster, and writes the output to `results/seed_20261125/`.

## Outputs

Each run creates the `results/seed_<base-seed>/` subfolder, which contains the
mass and radius-of-gyration CSV files, the PNG figures at 300 dpi, and exclusively
the `eden_vs_dla.gif` animation, at 1200 x 661, 12 fps, and approximately 20 seconds.

The fractal dimension is estimated by the fit `M proportional to R_g^D`. The model
steps are algorithmic growth steps and not a calibration of physical time.
