import numpy as np

from collections import Counter

class My_KNN:
    def __init__(self, n_neighbors=3):
        self.k = n_neighbors

    def fit(self, X, y):
        """
        Prepara o algoritmo k-NN com os dados de treinamento.

        Esta função inicializa o modelo k-NN, salvando os dados de treinamento para
        que possam ser utilizados em futuras previsões.

        Args:
            X (np.ndarray): Uma matriz ou array contendo as características (features)
                            do conjunto de dados de treinamento.
            y (np.ndarray): Um array contendo os rótulos ou valores alvo
                            correspondentes a cada amostra em X.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Realiza previsões com o classificador k-NN.

        Este método utiliza o modelo treinado para prever os rótulos de um novo conjunto
        de dados de entrada.

        Args:
            X (np.ndarray): Uma matriz ou array contendo as características (features)
                            do conjunto de dados para o qual as previsões serão feitas.

        Returns:
            np.ndarray: Um array contendo os rótulos previstos pelo classificador para
                        cada amostra em X.
        """
        return np.array([self.predict_aux(x) for x in X])

    def predict_aux(self, x):
        """
        Função auxiliar para prever um único ponto de dados.

        Args:
            x (np.ndarray): Um array contendo as características (features)
                            de um único ponto de dados para o qual se deseja prever o rótulo.

        Returns:
            int: O rótulo previsto para o ponto de dados 'x'.
        """
        # Calcula a distância euclidiana
        distances = [self.euclidian_distance(x, x_train) for x_train in self.X_train]

        # Obtém os K primeiros indices caso a array de distancias fosse ordenada
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Voto da maioria
        return int(Counter(k_nearest_labels).most_common()[0][0])

    def euclidian_distance(self, point1, point2):
        """
        Calcula a distância Euclidiana entre dois pontos.

        Args:
            point1 (np.ndarray): O primeiro ponto, representado como um array.
            point2 (np.ndarray): O segundo ponto, também como um array.

        Returns:
            float: A distância Euclidiana entre os dois pontos.
        """
        return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))
