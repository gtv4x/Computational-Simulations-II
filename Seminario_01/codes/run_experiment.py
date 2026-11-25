#!/usr/bin/env python3
"""Executa o estudo Eden versus DLA e gera artefatos em results/."""

import argparse
from pathlib import Path

from eden_dla.config import SimulationConfig
from eden_dla.experiments import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=int, default=20)
    parser.add_argument("--particles", type=int, default=1500)
    parser.add_argument("--size", type=int, default=501)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    config = SimulationConfig(lattice_size=args.size, particles=args.particles, seed=args.seed)
    print("=" * 72)
    print("Eden versus DLA")
    print(f"Rede: {args.size} x {args.size} | particulas: {args.particles} | replicas: {args.replicas}")
    print(f"Semente-base: {args.seed} | saida: {args.output / f'seed_{args.seed}'}")
    print("=" * 72, flush=True)
    fits, result_dir = run_experiment(config, args.replicas, args.output)
    print("=" * 72)
    print("Resultado final")
    for model, values in fits.items():
        mean = sum(item.dimension for item in values) / len(values)
        print(f"{model}: D medio = {mean:.3f} ({len(values)} replicas)")
    print(f"Arquivos gerados em: {result_dir.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
