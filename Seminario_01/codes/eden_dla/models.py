"""Implementacoes em rede dos modelos de Eden e DLA."""

from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig


NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class GrowthResult:
    """Coordenadas finais e estados usados para medidas e animacao."""

    coordinates: np.ndarray
    snapshots: dict[int, np.ndarray]


def _in_bounds(x: int, y: int, size: int) -> bool:
    return 0 <= x < size and 0 <= y < size


def _add_frontier_neighbors(
    x: int, y: int, occupied: np.ndarray, frontier: set[tuple[int, int]]
) -> None:
    size = occupied.shape[0]
    for dx, dy in NEIGHBORS:
        nx, ny = x + dx, y + dy
        if _in_bounds(nx, ny, size) and not occupied[ny, nx]:
            frontier.add((nx, ny))


def run_eden(config: SimulationConfig) -> GrowthResult:
    """Crescimento de Eden: um sitio livre da fronteira e ocupado por passo."""
    rng = np.random.default_rng(config.seed)
    size = config.lattice_size
    center = size // 2
    occupied = np.zeros((size, size), dtype=bool)
    occupied[center, center] = True
    coordinates: list[tuple[int, int]] = [(center, center)]
    frontier: set[tuple[int, int]] = set()
    _add_frontier_neighbors(center, center, occupied, frontier)
    requested = set(config.snapshot_masses())
    snapshots: dict[int, np.ndarray] = {1: np.asarray(coordinates, dtype=int)}

    while len(coordinates) < config.particles:
        candidates = tuple(frontier)
        x, y = candidates[rng.integers(len(candidates))]
        frontier.remove((x, y))
        if occupied[y, x]:
            continue
        occupied[y, x] = True
        coordinates.append((x, y))
        _add_frontier_neighbors(x, y, occupied, frontier)
        mass = len(coordinates)
        if mass in requested:
            snapshots[mass] = np.asarray(coordinates, dtype=int).copy()

    return GrowthResult(np.asarray(coordinates, dtype=int), snapshots)


def _has_occupied_neighbor(x: int, y: int, occupied: np.ndarray) -> bool:
    size = occupied.shape[0]
    for dx, dy in NEIGHBORS:
        nx, ny = x + dx, y + dy
        if _in_bounds(nx, ny, size) and occupied[ny, nx]:
            return True
    return False


def _launch_walker(rng: np.random.Generator, center: int, radius: float) -> tuple[int, int]:
    theta = rng.uniform(0.0, 2.0 * np.pi)
    return (
        int(round(center + radius * np.cos(theta))),
        int(round(center + radius * np.sin(theta))),
    )


def run_dla(config: SimulationConfig) -> GrowthResult:
    """DLA 2D com lancamento e descarte adaptados ao raio do agregado."""
    rng = np.random.default_rng(config.seed)
    size = config.lattice_size
    center = size // 2
    occupied = np.zeros((size, size), dtype=bool)
    occupied[center, center] = True
    coordinates: list[tuple[int, int]] = [(center, center)]
    radius = 0.0
    requested = set(config.snapshot_masses())
    snapshots: dict[int, np.ndarray] = {1: np.asarray(coordinates, dtype=int)}

    while len(coordinates) < config.particles:
        launch_radius = max(config.launch_offset, radius + config.launch_offset)
        kill_radius = launch_radius + config.kill_margin
        x, y = _launch_walker(rng, center, launch_radius)
        if not _in_bounds(x, y, size):
            continue

        for _ in range(config.max_walker_steps):
            direction = rng.integers(4)
            dx, dy = NEIGHBORS[direction]
            nx, ny = x + dx, y + dy
            if not _in_bounds(nx, ny, size):
                break
            x, y = nx, ny
            distance = float(np.hypot(x - center, y - center))
            if distance > kill_radius:
                break
            if _has_occupied_neighbor(x, y, occupied):
                if occupied[y, x]:
                    break
                occupied[y, x] = True
                coordinates.append((x, y))
                radius = max(radius, distance)
                mass = len(coordinates)
                if mass in requested:
                    snapshots[mass] = np.asarray(coordinates, dtype=int).copy()
                break

    return GrowthResult(np.asarray(coordinates, dtype=int), snapshots)
