from pathlib import Path

results_dir = Path(__file__).parent / "results" # caminho para os resultados
results_dir.mkdir(parents=True, exist_ok=True)

# data_dir = Path(__file__).parent / "data" # caminho para os dados
# data_dir.mkdir(parents=True, exist_ok=True)