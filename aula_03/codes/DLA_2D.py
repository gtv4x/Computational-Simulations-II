import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
from pathlib import Path


# ============================================================
# 1. PARÂMETROS DA SIMULAÇÃO
# ============================================================

L = 101                  # tamanho da malha: L x L
N_PARTICULAS = 650      # número total de partículas, incluindo a semente

DELTA = 5                # distância entre o agregado e o ponto de lançamento
MARGEM_MORTE = 30        # distância adicional para o raio de morte

MAX_PASSOS = 20000       # máximo de passos permitidos para cada caminhante

PASSOS_POR_FRAME = 100    # passos da caminhada executados a cada quadro
INTERVALO = 20            # intervalo entre quadros da animação (ms)

TRILHA_MAX = 100         # número máximo de posições mostradas na trilha

ARQUIVO_CSV = "dla_coordenadas_finais.csv"

SEED = 13                # semente para tornar o resultado reprodutível
rng = np.random.default_rng(SEED)

SAVE_ANIMATION = True
OUTPUT_DIR = Path("results")

plt.rcParams['font.family'] = 'serif' # Configurar fonte do gráfico
plt.rcParams['mathtext.fontset'] = 'cm'


# ============================================================
# 2. MALHA E SEMENTE INICIAL
# ============================================================

grade = np.zeros((L, L), dtype=bool)

centro_x = L // 2
centro_y = L // 2

# A primeira partícula é a semente do agregado
grade[centro_y, centro_x] = True

# Lista contendo as coordenadas (x, y) das partículas agregadas
coordenadas = [(centro_x, centro_y)]

# Número atual de partículas no agregado
particulas_agregadas = 1

# Raio atual do agregado
R = 0.0


# ============================================================
# 3. FUNÇÕES FÍSICAS/GEOMÉTRICAS DO MODELO
# ============================================================

def distancia_centro(x, y):
    """
    Distância radial da partícula ao centro do agregado.
    """
    dx = x - centro_x
    dy = y - centro_y

    return np.sqrt(dx**2 + dy**2)


def raio_lancamento():
    """
    Retorna o raio de lançamento:

        R_l = R + DELTA

    seguindo a estrutura apresentada na aula.
    """
    return R + DELTA


def raio_morte():
    """
    Raio de morte:

        R_kill = R + MARGEM_MORTE

    Se o caminhante ultrapassar esse raio, ele é descartado.
    """
    return R + MARGEM_MORTE


def nova_particula():
    """
    Sorteia uma posição inicial para o novo caminhante.

    A partícula é lançada aleatoriamente sobre uma circunferência
    de raio R_l em torno do agregado.
    """

    r = raio_lancamento()

    theta = rng.uniform(0, 2 * np.pi)

    x = int(round(centro_x + r * np.cos(theta)))
    y = int(round(centro_y + r * np.sin(theta)))

    return x, y


def tem_vizinho_ocupado(x, y):
    """
    Verifica se o sítio (x,y) possui pelo menos um vizinho
    ocupado.

    Usamos os quatro vizinhos da rede quadrada:

                (x, y+1)

        (x-1,y) (x,y) (x+1,y)

                (x, y-1)
    """

    vizinhos = [
        (x + 1, y),
        (x - 1, y),
        (x, y + 1),
        (x, y - 1)
    ]

    for xx, yy in vizinhos:

        # Primeiro verificamos se o vizinho está dentro da malha
        if 0 <= xx < L and 0 <= yy < L:

            if grade[yy, xx]:
                return True

    return False


def mover_particula(x, y):
    """
    Executa um passo de caminhada aleatória em uma rede quadrada.

    Há quatro possibilidades equiprováveis:
        +x
        -x
        +y
        -y
    """

    direcao = rng.integers(0, 4)

    if direcao == 0:
        x += 1

    elif direcao == 1:
        x -= 1

    elif direcao == 2:
        y += 1

    elif direcao == 3:
        y -= 1

    return x, y


# ============================================================
# 4. ESTADO DO CAMINHANTE ATUAL
# ============================================================

# Primeira partícula que será lançada
walker_x, walker_y = nova_particula()

# Histórico recente da caminhada.
# Serve para desenhar a "trilha" deixada pelo caminhante.
trilha_x = [walker_x]
trilha_y = [walker_y]

# Número de passos realizados pelo caminhante atual
passos_walker = 0

# Número total de caminhantes lançados
caminhantes_lancados = 1

# Flag para indicar que a simulação terminou
simulacao_terminada = False


# ============================================================
# 5. FUNÇÃO PARA LANÇAR UM NOVO CAMINHANTE
# ============================================================

def lancar_novo_walker():
    """
    Cria um novo caminhante e reinicia sua trilha.
    """

    global walker_x, walker_y
    global trilha_x, trilha_y
    global passos_walker
    global caminhantes_lancados

    walker_x, walker_y = nova_particula()

    trilha_x = [walker_x]
    trilha_y = [walker_y]

    passos_walker = 0

    caminhantes_lancados += 1


# ============================================================
# 6. PROCESSAMENTO DE UM PASSO DO DLA
# ============================================================

def executar_um_passo():
    """
    Executa um passo do caminhante atual.

    Retorna:
        True  -> a simulação terminou
        False -> a simulação continua
    """

    global walker_x, walker_y
    global passos_walker
    global particulas_agregadas
    global R
    global simulacao_terminada

    # --------------------------------------------------------
    # Verifica se já atingimos o número desejado de partículas
    # --------------------------------------------------------

    if particulas_agregadas >= N_PARTICULAS:

        simulacao_terminada = True

        return True


    # --------------------------------------------------------
    # Incrementa o contador de passos
    # --------------------------------------------------------

    passos_walker += 1


    # --------------------------------------------------------
    # Verifica se o caminhante saiu da malha
    # --------------------------------------------------------

    if not (0 <= walker_x < L and 0 <= walker_y < L):

        # Caminhante perdido: descartamos e lançamos outro
        lancar_novo_walker()

        return False


    # --------------------------------------------------------
    # Verifica o raio de morte
    # --------------------------------------------------------

    r = distancia_centro(walker_x, walker_y)

    if r > raio_morte():

        # O caminhante se afastou demais do agregado.
        # Ele é descartado.

        lancar_novo_walker()

        return False


    # --------------------------------------------------------
    # Verifica se o caminhante pode aderir ao agregado
    # --------------------------------------------------------

    if tem_vizinho_ocupado(walker_x, walker_y):

        # O caminhante encontrou o agregado.
        #
        # O sítio atual passa a fazer parte do cluster.

        if not grade[walker_y, walker_x]:

            grade[walker_y, walker_x] = True

            coordenadas.append((walker_x, walker_y))

            particulas_agregadas += 1

            # Atualiza o raio do agregado
            R = max(R, r)


        # Se atingimos o número final de partículas,
        # encerramos a simulação.
        if particulas_agregadas >= N_PARTICULAS:

            simulacao_terminada = True

            salvar_csv()

            return True


        # Caso contrário, lançamos um novo caminhante.
        lancar_novo_walker()

        return False


    # --------------------------------------------------------
    # Se não aderiu, realiza um passo de caminhada aleatória
    # --------------------------------------------------------

    walker_x, walker_y = mover_particula(walker_x, walker_y)


    # --------------------------------------------------------
    # Atualiza a trilha do caminhante
    # --------------------------------------------------------

    trilha_x.append(walker_x)
    trilha_y.append(walker_y)

    # Mantemos apenas os últimos pontos da trilha
    if len(trilha_x) > TRILHA_MAX:

        trilha_x.pop(0)
        trilha_y.pop(0)


    # --------------------------------------------------------
    # Se o caminhante ficou tempo demais andando, descartamos
    # --------------------------------------------------------

    if passos_walker >= MAX_PASSOS:

        lancar_novo_walker()


    return False


# ============================================================
# 7. SALVAMENTO DAS COORDENADAS EM CSV
# ============================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def salvar_csv():
    """
    Salva as coordenadas finais do agregado em um arquivo CSV.

    O arquivo terá três colunas:

        particula, x, y
    """

    with open(OUTPUT_DIR / ARQUIVO_CSV, "w", newline="") as arquivo:

        escritor = csv.writer(arquivo)

        escritor.writerow(["particula", "x", "y"])

        for i, (x, y) in enumerate(coordenadas):

            escritor.writerow([i, x, y])

    print()
    print("=" * 60)
    print("SIMULAÇÃO FINALIZADA")
    print("=" * 60)
    print(f"Partículas agregadas : {particulas_agregadas}")
    print(f"Raio final           : {R:.2f}")
    print(f"Arquivo CSV          : {ARQUIVO_CSV}")
    print("=" * 60)


# ============================================================
# 8. CONFIGURAÇÃO DA ANIMAÇÃO
# ============================================================

fig, ax = plt.subplots(figsize=(8, 8))


# ------------------------------------------------------------
# Agregado
# ------------------------------------------------------------

imagem = ax.imshow(
    grade,
    origin="lower",
    interpolation="nearest",
    cmap="Greys",
    vmin=0,
    vmax=1
)


# ------------------------------------------------------------
# Caminhante atual
# ------------------------------------------------------------

walker_plot, = ax.plot(
    [walker_x],
    [walker_y],
    "o",
    markersize=6
)


# ------------------------------------------------------------
# Trilha da caminhada
# ------------------------------------------------------------

trilha_plot, = ax.plot(
    trilha_x,
    trilha_y,
    "-",
    linewidth=1,
    alpha=0.5
)


# ------------------------------------------------------------
# Configuração dos eixos
# ------------------------------------------------------------

ax.set_xlim(0, L - 1)
ax.set_ylim(0, L - 1)

ax.set_aspect("equal")

ax.set_xlabel("x")
ax.set_ylabel("y")

ax.set_title("DLA 2D — caminhada aleatória")


# ============================================================
# 9. FUNÇÃO DE ATUALIZAÇÃO DA ANIMAÇÃO
# ============================================================

def atualizar(frame):

    global simulacao_terminada

    # Executamos vários passos por quadro para acelerar
    # a animação sem eliminar a visualização da caminhada.

    for _ in range(PASSOS_POR_FRAME):

        if not simulacao_terminada:

            terminou = executar_um_passo()

            if terminou:
                break


    # --------------------------------------------------------
    # Atualiza a imagem do agregado
    # --------------------------------------------------------

    imagem.set_data(grade)


    # --------------------------------------------------------
    # Atualiza a posição do caminhante atual
    # --------------------------------------------------------

    if not simulacao_terminada:

        walker_plot.set_data(
            [walker_x],
            [walker_y]
        )

        trilha_plot.set_data(
            trilha_x,
            trilha_y
        )

    else:

        # Quando termina, escondemos o caminhante.
        walker_plot.set_data([], [])
        trilha_plot.set_data([], [])


    # --------------------------------------------------------
    # Atualiza o título
    # --------------------------------------------------------

    ax.set_title(
        f"DLA 2D — Partículas: "
        f"{particulas_agregadas}/{N_PARTICULAS}"
        f"\nRaio: {R:.1f}"
    )


    return imagem, walker_plot, trilha_plot


# ============================================================
# 10. CRIAÇÃO DA ANIMAÇÃO
# ============================================================

# ------------------------------------------------------------
# Determina o número de frames efetivamente necessários.
#
# Como a simulação usa uma SEED fixa, fazemos uma primeira
# passagem para descobrir quantos frames são necessários até
# atingir N_PARTICULAS. Depois reiniciamos a simulação com a
# mesma SEED e usamos exatamente esse número de frames no MP4.
# ------------------------------------------------------------

def contar_frames_necessarios():
    global grade, coordenadas, particulas_agregadas, R
    global walker_x, walker_y, trilha_x, trilha_y
    global passos_walker, caminhantes_lancados
    global rng

    # Reinicializa o estado exatamente como no início
    rng = np.random.default_rng(SEED)

    grade = np.zeros((L, L), dtype=bool)

    centro_x = L // 2
    centro_y = L // 2

    grade[centro_y, centro_x] = True

    coordenadas = [(centro_x, centro_y)]
    particulas_agregadas = 1
    R = 0.0

    walker_x, walker_y = nova_particula()

    trilha_x = [walker_x]
    trilha_y = [walker_y]

    passos_walker = 0
    caminhantes_lancados = 1

    frames = 0

    while particulas_agregadas < N_PARTICULAS:

        terminou = False

        for _ in range(PASSOS_POR_FRAME):

            terminou = executar_um_passo()

            if terminou:
                break

        frames += 1

        if terminou:
            break

    return frames


# Primeira passagem: apenas conta os frames.
NUM_FRAMES = contar_frames_necessarios()

print()
print("=" * 60)
print("NÚMERO EXATO DE FRAMES")
print("=" * 60)
print(f"Frames necessários : {NUM_FRAMES}")
print(f"FPS do vídeo       : 30")
print(f"Duração do vídeo   : {NUM_FRAMES / 30:.2f} s")
print("=" * 60)


# ------------------------------------------------------------
# Reinicializa a simulação para produzir o vídeo.
# ------------------------------------------------------------

rng = np.random.default_rng(SEED)

grade = np.zeros((L, L), dtype=bool)

centro_x = L // 2
centro_y = L // 2

grade[centro_y, centro_x] = True

coordenadas = [(centro_x, centro_y)]
particulas_agregadas = 1
R = 0.0

walker_x, walker_y = nova_particula()

trilha_x = [walker_x]
trilha_y = [walker_y]

passos_walker = 0
caminhantes_lancados = 1

simulacao_terminada = False


# Agora o Matplotlib sabe exatamente quantos frames existem.
animacao = FuncAnimation(
    fig,
    atualizar,
    frames=NUM_FRAMES,
    interval=INTERVALO,
    blit=True,
    repeat=False
)

if SAVE_ANIMATION:
    try:
        print("Salvando animação...")
        animacao.save(
            OUTPUT_DIR / "DLA_2D.mp4",
            writer="ffmpeg",
            fps=30,
            dpi=120
        )
        print("Animação salva em:", OUTPUT_DIR / "DLA_2D.mp4")
    except Exception as exc:
        print("Não foi possível salvar MP4:", exc)

