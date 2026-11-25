"""Padrao visual compartilhado por todas as figuras e animacoes."""

import matplotlib.pyplot as plt


def configure_matplotlib() -> None:
    """Aplica a tipografia compativel com as formulas do LaTeX."""
    plt.rcParams['font.family'] = 'serif' # Configurar fonte do gráfico
    plt.rcParams['mathtext.fontset'] = 'cm'


configure_matplotlib()
