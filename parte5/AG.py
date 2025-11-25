import numpy as np

INTERVALO = [-10, 10]
POPULACAO_INICIAL = 4


def func(x: int):
    return pow(x, 2) - 3 * x + 4


def calc_prob(cromossomos):
    soma = np.sum(func(cromossomos))
    probabilidades = []

    for i in cromossomos:
        probabilidade = round((func(i) / soma), 2)
        probabilidades.append(probabilidade)

    # Faz a normalização do array
    soma = np.sum(probabilidades)
    probabilidades = probabilidades / soma

    return np.array(probabilidades)


def gerar_populacao_inicial():
    ja_foi = set()
    cromossomos = []

    while len(ja_foi) != POPULACAO_INICIAL:
        numero_aleatorio = np.random.randint(low=INTERVALO[0], high=INTERVALO[1])
        if numero_aleatorio not in ja_foi:
            ja_foi.add(numero_aleatorio)
            cromossomos.append(numero_aleatorio)

    return np.array(cromossomos)


def roleta(cromossomos):
    probabilidades = calc_prob(cromossomos)

    return np.random.choice(a=cromossomos, size=4, p=probabilidades)

def crossover(cromossomos):
    

cromossomos = gerar_populacao_inicial()

print(roleta(cromossomos=cromossomos))
