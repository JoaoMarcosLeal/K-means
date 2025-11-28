import numpy as np
from random import choices

INTERVALO = [-10, 10]
POPULACAO = 4
TAM_CROMOSSOMO = 5


def func(x: int):
    return pow(x, 2) - 3 * x + 4

def gerar_indiviuo():
    individuo_int = np.random.randint(low=INTERVALO[0], high=INTERVALO[1])
    individuo = "{0:b}".format(abs(individuo_int))

    # Adiciona os zeros para ter o mesmo tamanho de um cromossomo
    individuo = "0" * (TAM_CROMOSSOMO - 1 - len(individuo)) + individuo

    # Atreibui um bit para representar o sinal (1 negativo ou 0 positivo)
    sinal = "1" if individuo_int < 0 else "0"
    individuo = sinal + individuo

    return individuo


def converter_para_int(num_binario: str):
    sinal = -1 if num_binario[0] == 1 else 1
    num_int = sinal * int(num_binario[0:], 2)
    return num_int

def calc_prob(cromossomos):
    individuos = []
    individuos.append(converter_para_int(cromossomo) for cromossomo in cromossomos) 
    soma = np.sum(func(individuos))
    probabilidades = []

    for i in individuos:
        probabilidade = round((func(i) / soma), 2)
        probabilidades.append(probabilidade)

    # Faz a normalização do array
    soma = np.sum(probabilidades)
    probabilidades = probabilidades / soma

    return np.array(probabilidades)


def gerar_populacao_inicial():
    cromossomos = []

    for _ in range(POPULACAO):
        cromossomos.append(gerar_indiviuo())

    return np.array(cromossomos)


def torneio(cromossomos):
    probabilidades = calc_prob(cromossomos)
    
    for _ in range(POPULACAO):
        escolhas = np.random.choice(a=cromossomos, size=2, p=probabilidades)

    return 

if __name__ == "__main__":
    cromossomos = gerar_populacao_inicial()

    print(cromossomos)
