"""
Author: Gabriel T. Villela
Last edited: 17/11/2026 - 20:08
Leia o README.md para instruções de uso.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from dir_manager import results_dir

# ============================================================
# PARÂMETROS FÍSICOS
# ============================================================

J = 1.0
kB = 1.0
h = 0.0

# Tamanhos das redes
L_values = [8, 16, 24, 32]

# Intervalo de temperaturas
temperatures = np.linspace(2.10, 2.45, 71)

# Número de MCS
N_THERM = 5000
N_MEASURE = 20000

# Semente para reprodutibilidade
SEED = 42

# Temperatura crítica exata
TC_EXACT = 2.0 / np.log(1.0 + np.sqrt(2.0))


# ============================================================
# GERADOR ALEATÓRIO PARA NUMBA
# ============================================================

@njit
def seed_rng(seed):
    np.random.seed(seed)


# ============================================================
# ENERGIA TOTAL
# ============================================================

def energia_total(spins):

    E = -J * np.sum(
        spins * (
            np.roll(spins, -1, axis=0)
            + np.roll(spins, -1, axis=1)
        )
    )

    E -= h * np.sum(spins)

    return E


# ============================================================
# ALGORITMO DE METROPOLIS
# ============================================================

@njit
def metropolis_sweep(spins, T, E, M):

    L = spins.shape[0]
    N = L * L

    beta = 1.0 / (kB * T)

    # Uma varredura contém N tentativas
    for attempt in range(N):

        # Escolha aleatória do sítio
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)

        s = spins[i, j]

        # Condições periódicas de contorno
        ip = (i + 1) % L
        im = (i - 1) % L
        jp = (j + 1) % L
        jm = (j - 1) % L

        # Soma dos primeiros vizinhos
        neighbors = (
            spins[ip, j]
            + spins[im, j]
            + spins[i, jp]
            + spins[i, jm]
        )

        # Variação de energia ao inverter o spin
        delta_E = 2.0 * s * (J * neighbors + h)

        # Critério de aceitação de Metropolis
        if delta_E <= 0.0:

            accept = True

        else:

            accept = np.random.random() < np.exp(-beta * delta_E)

        # Atualização da configuração
        if accept:

            spins[i, j] = -s

            E += delta_E
            M -= 2 * s

    return E, M


# ============================================================
# SIMULAÇÃO PARA UMA TEMPERATURA
# ============================================================

def simular_temperatura(L, T, seed):

    N = L * L

    seed_rng(seed)

    # Configuração inicial aleatória
    spins = np.random.choice(
        np.array([-1, 1]),
        size=(L, L)
    )

    E = energia_total(spins)
    M = np.sum(spins)

    # --------------------------
    # Termalização
    # --------------------------

    for _ in range(N_THERM):

        E, M = metropolis_sweep(
            spins, T, E, M
        )

    # --------------------------
    # Coleta de amostras
    # --------------------------

    energies = np.empty(N_MEASURE)

    for step in range(N_MEASURE):

        E, M = metropolis_sweep(
            spins, T, E, M
        )

        energies[step] = E

    return energies


# ============================================================
# CALOR ESPECÍFICO
# ============================================================

def calcular_calor_especifico(energies, T, N):

    mean_E = np.mean(energies)
    mean_E2 = np.mean(energies**2)

    variance_E = mean_E2 - mean_E**2

    cv = variance_E / (N * kB * T**2)

    return cv


# ============================================================
# SIMULAÇÃO PARA DIFERENTES TAMANHOS DE REDE
# ============================================================

results = {}

for L in L_values:

    print(f"\nSimulando rede L = {L}")

    cv_values = []

    for index, T in enumerate(temperatures):

        energies = simular_temperatura(
            L,
            T,
            SEED + 1000 * L + index
        )

        cv = calcular_calor_especifico(
            energies,
            T,
            L**2
        )

        cv_values.append(cv)

        print(
            f"T = {T:.3f} | "
            f"cV = {cv:.5f}"
        )

    results[L] = np.array(cv_values)


# ============================================================
# IDENTIFICAÇÃO DOS MÁXIMOS
# ============================================================

print("\nTemperaturas pseudocríticas:")

peak_temperatures = []

for L in L_values:

    index_max = np.argmax(results[L])

    T_peak = temperatures[index_max]
    cv_max = results[L][index_max]

    peak_temperatures.append(T_peak)

    print(
        f"L = {L:2d} | "
        f"T_peak = {T_peak:.5f} | "
        f"cV_max = {cv_max:.5f}"
    )


# ============================================================
# GRÁFICO DO CALOR ESPECÍFICO
# ============================================================

plt.rcParams['font.family'] = 'serif' # Configurar fonte do gráfico
plt.rcParams['mathtext.fontset'] = 'cm'

plt.figure(figsize=(9, 6))

for L in L_values:

    plt.plot(
        temperatures,
        results[L],
        marker='o',
        markersize=3,
        label=f'L = {L}'
    )

plt.axvline(
    TC_EXACT,
    linestyle='--',
    label=fr'$T_c = {TC_EXACT:.6f}$'
)

plt.xlabel('Temperatura (T)')
plt.ylabel(r'Calor específico por spin ($c_V$)')

plt.title('Modelo de Ising 2D: calor específico')

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(results_dir /'calor_especifico.png', dpi=300)



# ============================================================
# GRÁFICO DA TEMPERATURA PSEUDOCRÍTICA
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    L_values,
    peak_temperatures,
    marker='o',
    label='Simulação'
)

plt.axhline(
    TC_EXACT,
    linestyle='--',
    label='Temperatura crítica exata'
)

plt.xlabel('Tamanho linear da rede (L)')
plt.ylabel(r'Temperatura do máximo de $c_V$')

plt.title('Temperatura pseudocrítica versus tamanho da rede')

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(results_dir / 'temperatura_critica.png', dpi=300)



# ============================================================
# VISUALIZAÇÃO DAS CONFIGURAÇÕES DE SPIN
# ============================================================

def simular_configuracoes(L, T, n_mcs=2000, seed=123):

    rng = np.random.default_rng(seed)

    seed_rng(seed)

    spins = rng.choice(
        np.array([-1, 1]),
        size=(L, L)
    )

    E = energia_total(spins)
    M = np.sum(spins)

    for _ in range(n_mcs):

        E, M = metropolis_sweep(
            spins, T, E, M
        )

    return spins.copy()


L_visual = 64

temperatures_visual = [
    1.5,
    TC_EXACT,
    3.0
]

fig, axes = plt.subplots(
    1, 3,
    figsize=(12, 4)
)

for ax, T in zip(axes, temperatures_visual):

    spins = simular_configuracoes(
        L_visual,
        T,
        seed=SEED + int(100 * T)
    )

    ax.imshow(
        spins,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        interpolation='nearest'
    )

    ax.set_title(f'T = {T:.3f}')
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(
    'Configurações do modelo de Ising 2D',
    fontsize=14
)

plt.tight_layout()
plt.savefig(results_dir / 'configuracoes_ising.png', dpi=300)

print("\nSimulação concluída com sucesso. Consulte os resultados salvos na pasta '/results'.")