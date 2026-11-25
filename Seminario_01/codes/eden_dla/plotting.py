"""Figuras estaticas e animacao GIF do seminario."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .models import GrowthResult
from . import style  # aplica o padrao serif/Computer Modern a todas as saidas


STATIC_DPI = 300
GIF_FIGSIZE = (12, 12 * 1984 / 3600)
GIF_DPI = 100
GIF_FPS = 12

def _limits(*results: GrowthResult, padding_fraction: float = 0.12) -> tuple[float, float, float, float]:
    """Limites quadrados com folga para que nenhum marcador seja recortado."""
    points = np.vstack([result.coordinates for result in results])
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    side = max(x_max - x_min, y_max - y_min)
    padding = max(8.0, side * padding_fraction)
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    radius = side / 2.0 + padding
    return center_x - radius, center_x + radius, center_y - radius, center_y + radius


def save_final_states(eden: GrowthResult, dla: GrowthResult, output: Path) -> None:
    x_low, x_high, y_low, y_high = _limits(eden, dla)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharex=True, sharey=True)
    for ax, result, title, color in zip(
        axes, (eden, dla), ("Eden: crescimento compacto", "DLA: crescimento ramificado"),
        ("tab:blue", "tab:orange"), strict=True,
    ):
        ax.scatter(result.coordinates[:, 0], result.coordinates[:, 1], s=5, c=color)
        mass = len(result.coordinates)
        ax.set(
            title=f"{title}\nM = {mass} partículas",
            xlim=(x_low, x_high), ylim=(y_low, y_high), aspect="equal",
        )
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.tight_layout()
    fig.savefig(output, dpi=STATIC_DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def save_animation_frames(eden: GrowthResult, dla: GrowthResult, output: Path) -> None:
    masses = sorted(set(eden.snapshots) & set(dla.snapshots))
    chosen = [masses[0], masses[len(masses) // 2], masses[-1]]
    x_low, x_high, y_low, y_high = _limits(eden, dla)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for column, mass in enumerate(chosen):
        for row, (result, color, label) in enumerate(
            ((eden, "tab:blue", "Eden"), (dla, "tab:orange", "DLA"))
        ):
            ax = axes[row, column]
            points = result.snapshots[mass]
            ax.scatter(points[:, 0], points[:, 1], s=5, c=color)
            ax.set(xlim=(x_low, x_high), ylim=(y_low, y_high), aspect="equal")
            ax.set_title(f"{label}, M = {mass}")
    fig.tight_layout()
    fig.savefig(output, dpi=STATIC_DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def save_gif_animation(eden: GrowthResult, dla: GrowthResult, output: Path, fps: int = GIF_FPS) -> None:
    """Exporta a evolucao dos dois modelos em GIF animado."""
    masses = sorted(set(eden.snapshots) & set(dla.snapshots))
    x_low, x_high, y_low, y_high = _limits(eden, dla)
    fig, axes = plt.subplots(1, 2, figsize=GIF_FIGSIZE, sharex=True, sharey=True)
    artists = []
    title_artists = []
    model_names = ("Eden", "DLA")
    for ax, title, color in zip(
        axes, model_names, ("tab:blue", "tab:orange"), strict=True
    ):
        artist = ax.scatter([], [], s=5, c=color)
        artists.append(artist)
        title_artists.append(ax.set_title(f"{title}\nM = 1"))
        ax.set(xlim=(x_low, x_high), ylim=(y_low, y_high), aspect="equal")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    label = fig.text(0.5, 0.02, "", ha="center")

    def update(index: int):
        mass = masses[index]
        artists[0].set_offsets(eden.snapshots[mass])
        artists[1].set_offsets(dla.snapshots[mass])
        for title_artist, model in zip(title_artists, model_names, strict=True):
            title_artist.set_text(f"{model}\nM = {mass}")
        label.set_text(f"Número de partículas agregadas: M = {mass}")
        return (*artists, *title_artists, label)

    animation = FuncAnimation(fig, update, frames=len(masses), interval=1000 / fps, blit=True)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=GIF_DPI)
    plt.close(fig)
