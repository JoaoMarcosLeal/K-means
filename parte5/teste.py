import numpy as np 

def func(x: int):
    return pow(x, 2)

def calc_prob(cromossomos):
    soma = np.sum(func(cromossomos))
    probabilidades = {}

    for i in cromossomos:
        probabilidades[i] = func(i) / soma

    return probabilidades

arr = np.array([25, 15, 14, 10])


print(calc_prob(arr))