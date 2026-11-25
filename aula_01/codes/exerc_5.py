import numpy as np
import matplotlib.pyplot as plt

passos = 1000
pi = np.pi

# Geração de ângulos por aceitação-rejeição para f(theta) = (1 + cos(theta)) / (2*pi)
theta_list = []
while len(theta_list) < passos:
    y = np.random.rand() * 2 * pi  # Proposta uniforme em [0, 2*pi]
    u = np.random.rand()
    if u <= (1.0 + np.cos(y)) / 2.0:
        theta_list.append(y)

theta_arr = np.array(theta_list)

# Construção da trajetória 2D
dx = np.cos(theta_arr)
dy = np.sin(theta_arr)
x_traj = np.concatenate([[0], np.cumsum(dx)])
y_traj = np.concatenate([[0], np.cumsum(dy)])

# Visualização
plt.rcParams['font.family'] = 'serif' # Configurar fonte do gráfico
plt.rcParams['mathtext.fontset'] = 'cm'
plt.figure(figsize=(7, 7))
plt.plot(x_traj, y_traj, '-o', markersize=3, color='purple', label='Caminhada com Viés')
plt.axvline(0, color='gray', linestyle='--')
plt.axhline(0, color='gray', linestyle='--')
plt.title('Questão 5: Caminhada 2D com $f(\\theta) \\propto 1 + \\cos\\theta$')
plt.xlabel('Deslocamento X')
plt.ylabel('Deslocamento Y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.axis('equal')
plt.show()

print(f"Deslocamento final em X (Drift): {x_traj[-1]:.4f}")
print(f"Deslocamento final em Y: {y_traj[-1]:.4f}")