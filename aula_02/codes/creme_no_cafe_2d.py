"""
Author: Gabriel T. Villela
Last edited: 19/10/2026 - 16:30

Exercício 4 — Simulações Computacionais II
Modelo 2D do "creme-no-café" de Giordano/Nakanishi.

Implementação:
    * recipiente quadrado L x L;
    * N partículas não interagentes;
    * cada partícula tenta mover-se para um dos 4 vizinhos;
    * paredes refletoras por rejeição do passo;
    * gota inicial compacta;
    * entropia coarse-grained em uma malha Mx x My (20 x 20 por padrão);
    * animação da evolução da distribuição;
    * curva S(t)/Smax mostrada simultaneamente.

Dependências:
    numpy
    matplotlib

Executar:
    python creme_no_cafe_2d.py
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# PARÂMETROS DA SIMULAÇÃO
# ============================================================

L = 100                    # tamanho linear da malha microscópica
N = 3000                  # número de partículas
N_STEPS = 6000            # número de passos de Monte Carlo (1 sweep = N tentativas)
MX = 20                    # número de caixas coarse-grained em x
MY = 20                    # número de caixas coarse-grained em y

DROP_FRACTION = 0.15      # lado da gota inicial como fração de L
DROP_CENTER = (0.25, 0.75)  # (x/L, y/L), aqui: região superior esquerda

FRAME_EVERY = 10          # a animação atualiza a cada tantos sweeps

SEED = 12345              # None para uma semente aleatória
OUTPUT_DIR = Path("resultados_creme_cafe")

SAVE_ANIMATION = False    # True para tentar salvar MP4/GIF
FPS = 30

plt.rcParams['font.family'] = 'serif' # Configurar fonte do gráfico
plt.rcParams['mathtext.fontset'] = 'cm'


# ============================================================
# FUNÇÕES DO MODELO
# ============================================================

def initialize_particles(rng):
    """Inicializa N partículas numa região quadrada compacta."""
    side = max(1, int(round(DROP_FRACTION * L)))

    cx = int(round(DROP_CENTER[0] * (L - 1)))
    cy = int(round(DROP_CENTER[1] * (L - 1)))

    x_min = max(0, cx - side // 2)
    x_max = min(L - 1, x_min + side - 1)
    y_min = max(0, cy - side // 2)
    y_max = min(L - 1, y_min + side - 1)

    # Escolha uniforme das posições dentro da gota compacta.
    x = rng.integers(x_min, x_max + 1, size=N)
    y = rng.integers(y_min, y_max + 1, size=N)

    return x, y


def one_sweep(x, y, rng):
    """
    Faz um passo de Monte Carlo:
    cada partícula tenta uma vez mover-se para um dos 4 vizinhos.

    As paredes são refletoras por rejeição:
    se o movimento sair da malha, a partícula permanece onde estava.
    """
    directions = rng.integers(0, 4, size=N)

    dx = np.zeros(N, dtype=np.int8)
    dy = np.zeros(N, dtype=np.int8)

    dx[directions == 0] = 1    # direita
    dx[directions == 1] = -1   # esquerda
    dy[directions == 2] = 1    # cima
    dy[directions == 3] = -1   # baixo

    new_x = x + dx
    new_y = y + dy

    valid = (
        (new_x >= 0) & (new_x < L) &
        (new_y >= 0) & (new_y < L)
    )

    # Rejeição do passo: só aceitamos movimentos para dentro do recipiente.
    x[valid] = new_x[valid]
    y[valid] = new_y[valid]


def coarse_occupancy(x, y):
    """
    Conta as partículas nas Mx x My caixas coarse-grained.
    """
    # Cada posição microscópica é convertida para o índice da caixa.
    ix = np.minimum((x * MX) // L, MX - 1)
    iy = np.minimum((y * MY) // L, MY - 1)

    occupation = np.zeros((MY, MX), dtype=np.int64)
    np.add.at(occupation, (iy, ix), 1)

    return occupation


def normalized_entropy(occupation):
    """
    s = -sum_i p_i ln(p_i) / ln(M),
    com M = MX * MY.

    Retorna um valor entre 0 e 1.
    """
    p = occupation.ravel().astype(float) / N
    p = p[p > 0]

    if p.size == 0:
        return 0.0

    S = -np.sum(p * np.log(p))
    S_max = np.log(MX * MY)

    return S / S_max


def advance(x, y, rng, n_sweeps):
    """Executa n_sweeps passos de Monte Carlo."""
    for _ in range(n_sweeps):
        one_sweep(x, y, rng)


# ============================================================
# SIMULAÇÃO
# ============================================================

def run_simulation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    x, y = initialize_particles(rng)

    times = [0]
    entropies = [normalized_entropy(coarse_occupancy(x, y))]

    # Guardamos os estados que serão usados pela animação.
    animation_states = [(0, x.copy(), y.copy(), entropies[-1])]


    for step in range(1, N_STEPS + 1):
        one_sweep(x, y, rng)

        if step % FRAME_EVERY == 0 or step == N_STEPS:
            occ = coarse_occupancy(x, y)
            s = normalized_entropy(occ)

            times.append(step)
            entropies.append(s)
            animation_states.append((step, x.copy(), y.copy(), s))


    save_entropy_csv(times, entropies)

    return animation_states, np.asarray(times), np.asarray(entropies)



def save_entropy_csv(times, entropies):
    """Salva t e entropia normalizada para análise posterior."""
    path = OUTPUT_DIR / "entropia.csv"

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "S_normalizada"])
        writer.writerows(zip(times, entropies))


# ============================================================
# ANIMAÇÃO
# ============================================================

def make_animation(animation_states, times, entropies):
    """
    Cria uma animação com dois painéis:
      esquerda: partículas na malha microscópica;
      direita: S(t)/Smax até o instante atual.
    """
    fig, (ax_map, ax_entropy) = plt.subplots(
        1, 2,
        figsize=(12, 5),
        gridspec_kw={"width_ratios": [1, 1]}
    )

    # ---- Painel da distribuição ----
    step0, x0, y0, s0 = animation_states[0]

    particles = ax_map.scatter(
        x0, y0,
        s=5,
        alpha=0.35
    )
    ax_map.set_xlim(-0.5, L - 0.5)
    ax_map.set_ylim(-0.5, L - 0.5)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("x")
    ax_map.set_ylabel("y")
    ax_map.set_title("Distribuição do creme")

    # Mostra as fronteiras da partição coarse-grained 7 x 7.
    for i in range(1, MX):
        ax_map.axvline(i * L / MX - 0.5, alpha=0.20, linewidth=0.7)
    for j in range(1, MY):
        ax_map.axhline(j * L / MY - 0.5, alpha=0.20, linewidth=0.7)

    # ---- Painel da entropia ----
    entropy_line, = ax_entropy.plot([], [], linewidth=2)
    current_point, = ax_entropy.plot([], [], "o")
    ax_entropy.set_xlim(0, N_STEPS)
    ax_entropy.set_ylim(0, 1.05)
    ax_entropy.set_xlabel("t  (sweeps)")
    ax_entropy.set_ylabel(r"$S/S_{\max}$")
    ax_entropy.set_title("Entropia coarse-grained")
    ax_entropy.grid(alpha=0.25)

    title = fig.suptitle("Modelo 2D do creme-no-café")

    def update(frame):
        step, x, y, s = animation_states[frame]

        particles.set_offsets(np.column_stack((x, y)))

        # Entropia acumulada até o instante atual.
        entropy_line.set_data(times[:frame + 1], entropies[:frame + 1])
        current_point.set_data([step], [s])

        title.set_text(
            f"Creme no café — t = {step} sweeps    |    S/Smax = {s:.3f}"
        )

        return particles, entropy_line, current_point, title

    animation = FuncAnimation(
        fig,
        update,
        frames=len(animation_states),
        interval=1000 / FPS,
        blit=False,
        repeat=False
    )

    fig.tight_layout()

    if SAVE_ANIMATION:
        # MP4 normalmente requer ffmpeg instalado.
        try:
            animation.save(
                OUTPUT_DIR / "creme_no_cafe.mp4",
                writer="ffmpeg",
                fps=FPS,
                dpi=120
            )
            print("Animação salva em:", OUTPUT_DIR / "creme_no_cafe.mp4")
        except Exception as exc:
            print("Não foi possível salvar MP4:", exc)
            print("A animação continuará disponível na janela do Matplotlib.")

    plt.show()


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":
    print("Executando modelo 2D do creme-no-café...")
    print(f"L = {L}, N = {N}, sweeps = {N_STEPS}, partição = {MX} x {MY}")

    states, times, entropies = run_simulation()

    print()
    print("Simulação concluída.")
    print(f"Entropia inicial: {entropies[0]:.4f}")
    print(f"Entropia final:   {entropies[-1]:.4f}")
    print(f"Resultados em:    {OUTPUT_DIR.resolve()}")

    make_animation(states, times, entropies)
