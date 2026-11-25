import numpy as np
import matplotlib.pyplot as plt

n = 100000

# 1. Método Hit-or-Miss (Quarto de Círculo)
x_hm = np.random.rand(n)
y_hm = np.random.rand(n)
dentro = (x_hm**2 + y_hm**2) <= 1.0
pi_hit_acumulado = 4.0 * np.cumsum(dentro) / np.arange(1, n + 1)

# 2. Método Estimador Integral (int 1/(1+x^2))
x_int = np.random.rand(n)
f_x = 4.0 / (1.0 + x_int**2)
pi_int_acumulado = np.cumsum(f_x) / np.arange(1, n + 1)

# Visualização da convergência
plt.rcParams['font.family'] = 'serif' # Configurar fonte do gráfico
plt.rcParams['mathtext.fontset'] = 'cm'
plt.figure(figsize=(9, 5))
plt.plot(pi_hit_acumulado, label='Hit-or-Miss (Círculo)', color='orange', alpha=0.7)
plt.plot(pi_int_acumulado, label='Estimador Integral', color='blue', alpha=0.7)
plt.axhline(np.pi, color='red', linestyle='--', label=f'Valor Real ($\pi$ $\\approx$ {np.pi:.4f})')
plt.title('Questão 4: Convergência de Monte Carlo para $\pi$')
plt.xlabel('Número de Amostras (N)')
plt.ylabel('Valor Estimado de $\\pi$')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

print(f"Estimativa final (Hit-or-Miss):     {pi_hit_acumulado[-1]:.6f}")
print(f"Estimativa final (Est. Integral):    {pi_int_acumulado[-1]:.6f}")