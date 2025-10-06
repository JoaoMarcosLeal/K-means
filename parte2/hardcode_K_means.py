import pandas as pd
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def get_random_centroid(df, k):
    return df.sample(k)


def get_euclidian_discance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def atribuir_centroid(df: pd.DataFrame, centroids: pd.DataFrame):
    k = centroids.shape[0]
    n = df.shape[0]
    atribuicao = []
    distancias = []

    for obs in range(n):
        all_errors = np.array([])
        for centroid in range(k):
            err = get_euclidian_discance(centroids.iloc[centroid, :], df.iloc[obs, :])
            all_errors = np.append(all_errors, err)

        nearest_centroid = np.where(all_errors == np.amin(all_errors))[0].tolist()[0]
        nearest_centroid_error = np.amin(all_errors)

        atribuicao.append(nearest_centroid)
        distancias.append(nearest_centroid_error)

    return atribuicao, distancias


def atualizar_centroids(df: pd.DataFrame, atribuicao: list, k: int):
    """
    Recalcula a posição dos centróides como a média dos pontos de dados
    atribuídos a cada um.
    """
    new_centroids = pd.DataFrame(columns=df.columns)
    df["atribuicao"] = atribuicao  # Adiciona a atribuição ao DataFrame para agrupar

    for cluster_id in range(k):
        # Seleciona todos os pontos de dados pertencentes a este cluster
        cluster_points = df[df["atribuicao"] == cluster_id]

        if not cluster_points.empty:
            # Calcula a média das colunas para obter o novo centróide
            new_centroid_pos = cluster_points.drop(columns=["atribuicao"]).mean()
            new_centroids.loc[cluster_id] = new_centroid_pos

    df = df.drop(columns=["atribuicao"])  # Remove a coluna de atribuição temporária
    return new_centroids.drop(columns=["dist"], errors="ignore")


def k_means_algorithm(df: pd.DataFrame, k: int, max_iter: int = 100):
    """
    Executa o algoritmo K-Means completo.
    """
    # 1. Inicialização: Escolhe k centróides aleatoriamente
    centroids = get_random_centroid(df, k)

    for i in range(max_iter):
        # 2. Atribuição: Atribui cada ponto ao centróide mais próximo
        atribuicao, distancias = atribuir_centroid(
            df.drop(columns=["dist"], errors="ignore"), centroids
        )

        # Guarda os centróides antigos para checar a convergência
        old_centroids = centroids.copy()

        # 3. Atualização: Recalcula os centróides
        centroids = atualizar_centroids(df, atribuicao, k)

        # Checa se os centróides mudaram significativamente. Se não, o algoritmo convergiu.
        if old_centroids.equals(centroids):
            break

    # Retorna os centróides finais e a atribuição de cada ponto
    df["centroid"] = atribuicao
    df["dist"] = distancias
    return df, centroids


def plotar(df, final_df):
    plt.figure(figsize=(10, 6))

    # Plota os pontos de dados coloridos por cluster
    plt.scatter(
        final_df["sepal length (cm)"],
        final_df["petal length (cm)"],
        c=final_df["centroid"],
        cmap="viridis",
        s=50,
        alpha=0.8,
    )

    # Plota os centróides finais como triângulos grandes e vermelhos
    plt.scatter(
        final_centroids["sepal length (cm)"],
        final_centroids["petal length (cm)"],
        color="red",
        marker="^",
        s=200,
        label="Centróides",
    )

    # Calcula e imprime o score de silhueta
    print(
        f"Score de Silhueta: {silhouette_score(df[features], final_df['centroid']):.2f}"
    )

    plt.title("K-Means Clustering (Iris Dataset)")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.legend()
    plt.grid(True)
    plt.show()


iris = load_iris()
clusters = [3, 5]
df = pd.DataFrame(iris.data, columns=iris.feature_names)
features = df.columns


for k in clusters:
    # Roda o algoritmo K-Means
    final_df, final_centroids = k_means_algorithm(df, k)
    print(f"##### Para k sendo {k} ######")
    plotar(df, final_df)
