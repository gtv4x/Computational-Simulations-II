"""Observaveis geometricos e ajuste de dimensao fractal."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FractalFit:
    dimension: float
    intercept: float
    r_squared: float
    mass_min: int
    mass_max: int


def radius_of_gyration(coordinates: np.ndarray) -> float:
    """Raio de giracao em unidades do espacamento da rede."""
    centered = coordinates.astype(float) - coordinates.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))


def fit_fractal_dimension(
    masses: np.ndarray, radii: np.ndarray, minimum_mass: int = 20
) -> FractalFit:
    """Ajusta log(M) = D log(Rg) no regime longe da escala microscopica."""
    mask = (masses >= minimum_mass) & (radii > 0.0)
    x = np.log(radii[mask])
    y = np.log(masses[mask])
    if x.size < 3:
        raise ValueError("Poucos pontos para estimar a dimensao fractal.")
    dimension, intercept = np.polyfit(x, y, 1)
    residual = y - (dimension * x + intercept)
    total = np.sum((y - y.mean()) ** 2)
    r_squared = 1.0 - np.sum(residual**2) / total if total else 1.0
    return FractalFit(
        float(dimension), float(intercept), float(r_squared),
        int(masses[mask].min()), int(masses[mask].max()),
    )
