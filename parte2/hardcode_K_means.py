import pandas as pd
from sklearn.datasets import load_iris
import numpy as np


def getRandomCentroid(dset, k):
    return dset.sample(k)

def getEuclidianDiscance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def 

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

centroids = getRandomCentroid(df, 3)


print(centroids)
