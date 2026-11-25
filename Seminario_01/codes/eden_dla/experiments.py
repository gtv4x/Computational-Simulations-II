"""Orquestracao das replicas, exportacao de dados e figuras."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import SimulationConfig
from .models import GrowthResult, run_dla, run_eden
from .observables import FractalFit, fit_fractal_dimension, radius_of_gyration
from .plotting import save_animation_frames, save_final_states, save_gif_animation
from . import style  # garante tipografia uniforme tambem no grafico de ajuste


def _records(result: GrowthResult, model: str, replica: int) -> list[dict[str, float | int | str]]:
    return [
        {"model": model, "replica": replica, "mass": mass,
         "radius_gyration": radius_of_gyration(points)}
        for mass, points in sorted(result.snapshots.items())
    ]


def _save_csv(records: list[dict[str, float | int | str]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("model", "replica", "mass", "radius_gyration"))
        writer.writeheader()
        writer.writerows(records)


def _save_fit_plot(records: list[dict[str, float | int | str]], fits: dict[str, list[FractalFit]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"Eden": "tab:blue", "DLA": "tab:orange"}
    for model in ("Eden", "DLA"):
        subset = [record for record in records if record["model"] == model]
        masses = np.asarray([record["mass"] for record in subset], dtype=float)
        radii = np.asarray([record["radius_gyration"] for record in subset], dtype=float)
        fit_values = fits[model]
        mean_dimension = np.mean([fit.dimension for fit in fit_values])
        std_dimension = np.std([fit.dimension for fit in fit_values], ddof=1)
        ax.scatter(radii, masses, s=10, alpha=0.25, c=colors[model])
        x = np.geomspace(radii[radii > 0].min(), radii.max(), 120)
        intercept = np.mean([fit.intercept for fit in fit_values])
        ax.plot(x, np.exp(intercept) * x**mean_dimension, c=colors[model], lw=2,
                label=f"{model}: D = {mean_dimension:.3f} +/- {std_dimension:.3f}")
    ax.set(xscale="log", yscale="log", xlabel=r"Raio de giracao $R_g$", ylabel="Massa M")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def run_experiment(config: SimulationConfig, replicas: int, output: Path) -> tuple[dict[str, list[FractalFit]], Path]:
    """Executa um conjunto de réplicas e agrupa suas saídas pela semente-base."""
    output = output / f"seed_{config.seed}"
    output.mkdir(parents=True, exist_ok=True)
    models = {"Eden": run_eden, "DLA": run_dla}
    records: list[dict[str, float | int | str]] = []
    fits: dict[str, list[FractalFit]] = {"Eden": [], "DLA": []}
    representatives: dict[str, GrowthResult] = {}

    print("[1/5] Iniciando replicas independentes...", flush=True)
    for model, runner in models.items():
        print(f"  Modelo {model}: {replicas} replica(s).", flush=True)
        for replica in range(replicas):
            seed = config.seed + 10_000 * (model == "DLA") + replica
            print(f"    [{model}] replica {replica + 1:02d}/{replicas:02d} | semente {seed}", flush=True)
            result = runner(SimulationConfig(**{**config.__dict__, "seed": seed}))
            if replica == 0:
                representatives[model] = result
            model_records = _records(result, model, replica)
            records.extend(model_records)
            masses = np.asarray([item["mass"] for item in model_records], dtype=int)
            radii = np.asarray([item["radius_gyration"] for item in model_records], dtype=float)
            fit = fit_fractal_dimension(masses, radii)
            fits[model].append(fit)
            print(f"      concluida: D = {fit.dimension:.3f}, R2 = {fit.r_squared:.4f}", flush=True)

    print("[2/5] Salvando series de massa e raio de giracao...", flush=True)
    _save_csv(records, output / "massa_raio_giracao.csv")
    print("[3/5] Gerando figuras em alta resolucao (300 dpi)...", flush=True)
    _save_fit_plot(records, fits, output / "massa_vs_raio_giracao.png")
    save_final_states(representatives["Eden"], representatives["DLA"], output / "estados_finais.png")
    save_animation_frames(representatives["Eden"], representatives["DLA"], output / "quadros_animacao.png")
    print("[4/5] Exportando GIF animado, aproximadamente 20 s...", flush=True)
    save_gif_animation(representatives["Eden"], representatives["DLA"], output / "eden_vs_dla.gif")

    print("[5/5] Salvando resumo estatistico...", flush=True)
    with (output / "resumo_dimensoes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model", "D_media", "D_desvio", "R2_medio", "massa_min", "massa_max"))
        for model, values in fits.items():
            writer.writerow((model, np.mean([fit.dimension for fit in values]),
                             np.std([fit.dimension for fit in values], ddof=1),
                             np.mean([fit.r_squared for fit in values]),
                             values[0].mass_min, values[0].mass_max))
    return fits, output
