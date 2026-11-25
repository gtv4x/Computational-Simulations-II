"""
Author: Gabriel T. Villela
Last edited: 10/11/2026 - 21:48
"""


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.family'] = 'serif' # Configurar fonte do gráfico
plt.rcParams['mathtext.fontset'] = 'cm'

# ==================================================
# 1. Parâmetros físicos
# ==================================================

J = 1.0
kB = 1.0
z = 4
h = 0.0

temperaturas = [3.2, 4.8]
condicoes_iniciais = [0.0, 1e-3, -1e-3, 0.8, -0.8]

tol = 1e-10
max_iter = 10000


# ==================================================
# 2. Equação de campo médio e suas derivadas
# ==================================================

def F(m, T):
    beta = 1.0 / (kB * T)
    return m - np.tanh(beta * (J*z*m + h))


def dF(m, T):
    beta = 1.0 / (kB * T)
    x = beta * (J*z*m + h)
    return 1.0 - beta*J*z / np.cosh(x)**2


# ==================================================
# 3. Método do ponto fixo
# ==================================================

def ponto_fixo(m0, T):

    m = m0
    historico = [m]

    for k in range(1, max_iter + 1):

        beta = 1.0 / (kB*T)
        m_novo = np.tanh(beta*(J*z*m + h))

        historico.append(m_novo)

        residuo = abs(F(m_novo, T))

        if abs(m_novo - m) < tol and residuo < tol:
            return m_novo, k, residuo, historico

        m = m_novo

    raise RuntimeError("Ponto fixo não convergiu.")


# ==================================================
# 4. Método de Newton-Raphson
# ==================================================

def newton(m0, T):

    m = m0
    historico = [m]

    for k in range(1, max_iter + 1):

        fm = F(m, T)
        dfm = dF(m, T)

        if abs(fm) < tol:
            return m, k-1, abs(fm), historico

        if abs(dfm) < 1e-14:
            raise RuntimeError("Derivada próxima de zero.")

        m_novo = m - fm/dfm

        historico.append(m_novo)

        residuo = abs(F(m_novo, T))

        if abs(m_novo - m) < tol and residuo < tol:
            return m_novo, k, residuo, historico

        m = m_novo

    raise RuntimeError("Newton não convergiu.")


# ==================================================
# 5. Energia livre de campo médio
# ==================================================

def energia_livre(m, T):

    p = np.clip((1 + m)/2, 1e-15, 1 - 1e-15)
    q = 1 - p

    entropia = p*np.log(p) + q*np.log(q)

    return -0.5*J*z*m**2 + kB*T*entropia


def curvatura(m, T):

    return kB*T/(1 - m**2) - J*z


def estabilidade(m, T):

    c = curvatura(m, T)

    if c > 1e-8:
        return "Estável (mínimo)"
    elif c < -1e-8:
        return "Instável (máximo)"
    else:
        return "Marginal"


# ==================================================
# 6. Execução dos métodos
# ==================================================

resultados = {}
historicos = {}
linhas_resultado = []

for T in temperaturas:

    linhas_resultado.append(f"\n{'='*75}")
    linhas_resultado.append(f"Temperatura T = {T}")
    linhas_resultado.append(f"{'='*75}")

    for nome, metodo in [
        ("Ponto fixo", ponto_fixo),
        ("Newton", newton)
    ]:

        linhas_resultado.append(f"\nMétodo: {nome}")

        for m0 in condicoes_iniciais:

            m, n, res, hist = metodo(m0, T)

            resultados[(T, nome, m0)] = (m, n, res)
            historicos[(T, nome, m0)] = hist

            linha = (
                f"m0={m0: .4f} | "
                f"raiz={m: .10f} | "
                f"iterações={n:4d} | "
                f"resíduo={res:.2e} | "
                f"{estabilidade(m, T)}"
            )
            linhas_resultado.append(linha)

resultado_dir = Path(__file__).resolve().parent.parent / "results"
resultado_dir.mkdir(parents=True, exist_ok=True)
resultado_path = resultado_dir / "results.txt"
resultado_path.write_text("\n".join(linhas_resultado) + "\n", encoding="utf-8")

# Mensagem de Confirmação
print(f"Resultados salvos em: {resultado_dir}/results.txt")

# ==================================================
# 7. Gráficos da função de resíduos e energia livre
# ==================================================

m_vals = np.linspace(-0.999, 0.999, 1000)

fig, ax = plt.subplots(2, 2, figsize=(12, 8))

for j, T in enumerate(temperaturas):

    # Função de resíduos
    ax[0, j].plot(
        m_vals,
        F(m_vals, T),
        label=r"$F(m)$"
    )

    ax[0, j].axhline(0, color="black", linestyle="--")
    ax[0, j].axvline(0, color="gray", alpha=0.5)

    ax[0, j].set_title(f"Função de resíduos - T = {T}")
    ax[0, j].set_xlabel("Magnetização m")
    ax[0, j].set_ylabel("F(m)")
    ax[0, j].grid(alpha=0.3)
    ax[0, j].legend()

    # Energia livre
    ax[1, j].plot(
        m_vals,
        energia_livre(m_vals, T),
        label=r"$f(m)$"
    )

    ax[1, j].set_title(f"Energia livre - T = {T}")
    ax[1, j].set_xlabel("Magnetização m")
    ax[1, j].set_ylabel("f(m)")
    ax[1, j].grid(alpha=0.3)
    ax[1, j].legend()

plt.tight_layout()
plt.savefig(resultado_dir / "residuos_energia.png", dpi=300)


# ==================================================
# 8. Gráfico de convergência dos métodos
# ==================================================

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

for j, T in enumerate(temperaturas):

    for nome in ["Ponto fixo", "Newton"]:

        for m0 in [1e-3, 0.8]:

            hist = historicos[(T, nome, m0)]

            ax[j].plot(
                range(len(hist)),
                hist,
                label=f"{nome}, $m_0$={m0}"
            )

    ax[j].set_title(f"Convergência - T = {T}")
    ax[j].set_xlabel("Iteração")
    ax[j].set_ylabel("Magnetização m")
    ax[j].grid(alpha=0.3)
    ax[j].legend()

plt.tight_layout()
plt.savefig(resultado_dir / "convergencia_metodos.png", dpi=300)

# Mensagem de Confirmação
print(f"Resultados salvos em: {resultado_dir}")