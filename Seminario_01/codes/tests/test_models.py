import numpy as np

from eden_dla.config import SimulationConfig
from eden_dla.models import run_dla, run_eden


def _is_connected(coordinates: np.ndarray) -> bool:
    points = {tuple(point) for point in coordinates}
    seen = {next(iter(points))}
    stack = list(seen)
    while stack:
        x, y = stack.pop()
        for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if neighbor in points and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == len(points)


def test_eden_is_reproducible_connected_and_unique():
    config = SimulationConfig(lattice_size=101, particles=90, seed=7)
    first, second = run_eden(config), run_eden(config)
    assert np.array_equal(first.coordinates, second.coordinates)
    assert len(first.coordinates) == len(set(map(tuple, first.coordinates))) == 90
    assert _is_connected(first.coordinates)


def test_dla_is_connected_and_unique():
    result = run_dla(SimulationConfig(lattice_size=101, particles=55, seed=9))
    assert len(result.coordinates) == len(set(map(tuple, result.coordinates))) == 55
    assert _is_connected(result.coordinates)
