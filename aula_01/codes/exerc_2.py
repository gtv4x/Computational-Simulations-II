import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
n = 100000
lam = 2.0

# 1. Geração por transformação inversa: t = -ln(U) / lambda
u = np.random.rand(n)
t_samples = -np.log(u) / lam

# 2. Validação gráfica e estatística
plt.rcParams['font.family'] = 'serif' # Configurar fonte do gráfico
plt.rcParams['mathtext.fontset'] = 'cm'
plt.figure(figsize=(8, 5))
count, bins, ignored = plt.hist(t_samples, bins=60, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Histograma Amostral')

# Curva da PDF teórica
t_val = np.linspace(0, max(t_samples), 200)
pdf_teorica = lam * np.exp(-lam * t_val)
plt.plot(t_val, pdf_teorica, 'r-', lw=2, label=f'PDF Teórica ($\lambda = {lam}$)')

plt.title('Questão 2: Distribuição Exponencial (Transformação Inversa)')
plt.xlabel('t')
plt.ylabel('Densidade de Probabilidade')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

print(f"Média amostral: {np.mean(t_samples):.4f} (Teórico: {1/lam:.4f})")
print(f"Variância amostral: {np.var(t_samples):.4f} (Teórico: {1/(lam**2):.4f})")