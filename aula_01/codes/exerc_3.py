import numpy as np
import time

n = 100000

# --- Método 1: Transformação Inversa (x = sqrt(U)) ---
start_time = time.time()
u_inv = np.random.rand(n)
x_inv = np.sqrt(u_inv)
time_inv = time.time() - start_time

# --- Método 2: Aceitação-Rejeição (g(x) = 1, c = 2) ---
start_time = time.time()
aceitos = []
tentativas = 0
while len(aceitos) < n:
    tentativas += 1
    y = np.random.rand()
    u = np.random.rand()
    if u <= y:  # u <= f(y) / (c * g(y)) => u <= 2y / 2 => u <= y
        aceitos.append(y)
x_rej = np.array(aceitos)
time_rej = time.time() - start_time

print("--- Questão 3: Comparação de Desempenho ---")
print(f"Média (Transformação Inversa): {np.mean(x_inv):.4f}")
print(f"Tempo (Transformação Inversa): {time_inv:.5f} s\n")
print(f"Média (Aceitação-Rejeição):   {np.mean(x_rej):.4f}")
print(f"Tempo (Aceitação-Rejeição):   {time_rej:.5f} s")
print(f"Taxa de aceitação empírica:   {n / tentativas:.4f} (Teórico: 0.50)")