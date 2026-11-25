# Modelo de Ising 2D — Exercício 02

## Objetivo

Este projeto implementa uma simulação computacional do modelo de Ising bidimensional em rede quadrada. O objetivo é calcular o calor específico por spin em diferentes temperaturas e tamanhos de rede, observar a transição de fase e estimar as temperaturas pseudocríticas próximas à temperatura crítica exata.

## Dependências

- Python 3
- NumPy
- Matplotlib
- Numba

Instale as bibliotecas necessárias com:

```bash
pip install numpy matplotlib numba
```

## Execução

A partir da sua pasta atual, execute:

```bash
python exerc_02.py
```

Os resultados gráficos são salvos automaticamente na pasta `/results`.

## Lógica do código

1. Define os parâmetros físicos, tamanhos de rede, temperaturas e números de passos Monte Carlo.
2. Inicializa uma configuração aleatória de spins $s_i = \pm 1$.
3. Calcula a energia total considerando interações com primeiros vizinhos, condições periódicas e campo magnético externo.
4. Aplica o algoritmo de Metropolis para inverter spins e atualizar a rede.
5. Realiza a termalização da configuração para cada temperatura.
6. Coleta as energias ao longo das medidas e calcula o calor específico por spin pela variância:

$$
c_V = \frac{\langle E^2 \rangle - \langle E \rangle^2}{N k_B T^2}
$$

7. Repete a simulação para redes de diferentes tamanhos e identifica os máximos de $c_V$.
8. Gera gráficos do calor específico, da temperatura pseudocrítica e das configurações de spins.

## Resultados

O código produz:

- `calor_especifico.png`: calor específico por spin em função da temperatura.
- `temperatura_critica.png`: temperatura do máximo de $c_V$ em função do tamanho linear da rede.
- `configuracoes_ising.png`: configurações de spins em temperaturas abaixo, próximas e acima da temperatura crítica.

A temperatura crítica exata usada como referência é:

$$
T_c = \frac{2}{\ln(1+\sqrt{2})}.
$$
