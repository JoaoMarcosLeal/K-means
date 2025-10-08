from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
)
import time

from KNN_hardcore import My_KNN

import matplotlib.pyplot as plt


def KNN_hardcore(k: int):
    """
    Previsão com o algoritmo k-NN.

    Esta função utiliza a minha implementação do algoritmo k-Nearest Neighbors (k-NN)
    para prever os resultados com base em um conjunto de vizinhos.

    Args:
        k (int): O número de vizinhos a serem considerados para a previsão.

    Returns:
        list[int]: Uma lista contendo as previsões (os "chutes") do algoritmo para cada ponto de dados.
    """
    clf = My_KNN(n_neighbors=k)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def KNN_sklearn(k: int):
    """
    Previsão com o algoritmo k-NN.

    Esta função utiliza a implementação do algoritmo k-Nearest Neighbors (k-NN), retirada da biblioteca Sklearn,
    para prever os resultados com base em um conjunto de vizinhos.

    Args:
        k (int): O número de vizinhos a serem considerados para a previsão.

    Returns:
        list[int]: Uma lista contendo as previsões (os "chutes") do algoritmo para cada ponto de dados.
    """
    
    knn = KNeighborsClassifier(n_neighbors=k)
    
    curr = time.time()
    knn.fit(X_train, y_train)
    total_time_training = time.time() - curr
    
    curr = time.time()
    predictions = knn.predict(X_test)
    total_time_clf = time.time() - curr    
    
    print(f"tempo de treinamento \n {total_time_training:.4f}")
    print(f"tempo de classificação \n {total_time_clf:.4f}")
    
    return predictions


def precision(y_pred):
    return precision_score(y_test, y_pred, average="micro")


def accuracy(y_pred):
    return accuracy_score(y_test, y_pred, normalize=True)


def recall(y_pred):
    return recall_score(y_test, y_pred, average="micro")


def printMetrics(y_pred, title):
    print(f"{title} precision score \n {precision(y_pred)}")
    print(f"{title} accuracy score \n {accuracy(y_pred)}")
    print(f"{title} recall score \n {recall(y_pred)}")


# Carrega o conjunto de dados iris
iris = load_iris()

# Armazena os dados e os valores das previsões em 2 variáveis
X, y = iris.data, iris.target

# Armazena o nome das classes
class_names = iris.target_names

# Separa os dados em um conjunto para testes e outro para treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

neighbors = [1, 3, 5, 7]

for k in neighbors:
    pred_my_KNN = KNN_hardcore(k)
    pred_sklearn_KNN = KNN_sklearn(k)

    titles = [
        (
            f"KNN sklearn, {k} neighbors",
            pred_sklearn_KNN,
        ),
        (f"My KNN {k}, neighbors", pred_my_KNN),
    ]

    # Percorre a lista de títulos exibindo uma matriz de confusão para cada um dos elementos presentes
    for title, pred in titles:
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, pred, cmap=plt.cm.Blues, display_labels=class_names
        )

        disp.ax_.set_title(title)

        # Exibe os dados da matriz de confusão no terminal
        print(title)
        print(disp.confusion_matrix)

        printMetrics(pred, title)

        # Exibe a matriz de confusão
        plt.show()
