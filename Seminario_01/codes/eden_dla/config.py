"""Parametros centralizados e reprodutiveis da simulacao."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    lattice_size: int = 501
    particles: int = 1500
    seed: int = 20261122
    launch_offset: float = 5.0
    kill_margin: float = 30.0
    max_walker_steps: int = 50_000

    def snapshot_masses(self, count: int = 240) -> tuple[int, ...]:
        """Estados uniformes para uma animacao de 20 segundos a 12 fps."""
        import numpy as np

        masses = np.unique(np.rint(np.linspace(1, self.particles, count)).astype(int))
        return tuple(int(value) for value in masses)
