import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

def plot(df, y_predicted, km): 
    # Cria um único gráfico de dispersão, colorindo os pontos com base no cluster
    plt.figure(figsize=(10, 6))
    plt.scatter(
        df["sepal length (cm)"],
        df["sepal width (cm)"],
        c=df["cluster"],
        cmap="viridis",
        s=50,
    )

    # Plota os centróides para visualização
    centroids = km.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=250,
        marker="*",
        c="red",
        edgecolor="k",
        label="Centróides",
    )

    plt.title("Clusters K-Means no Dataset Iris")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    plt.legend()
    plt.grid(True)

    # Calcula e imprime o score de silhueta
    print(f"Score de Silhueta: {silhouette_score(df[features], y_predicted):.2f}")

    plt.show()
    
# Carrega o dataset Iris
iris = load_iris()

# Cria o DataFrame e as colunas corretas
df = pd.DataFrame(iris.data, columns=iris.feature_names)


clusters = [3, 5]

for k in clusters:
    # Cria o modelo KMeans com 3 clusters
    km = KMeans(
        n_clusters=k, random_state=42
    )  # Adicionando random_state para reprodutibilidade

    # O K-Means precisa de todas as features para o agrupamento, então vamos corrigir a lista de colunas.
    # O seu código original tinha "petal width (cm)" duplicada.
    features = df.columns
    y_predicted = km.fit_predict(df[features])

    # Adiciona a coluna 'cluster' ao DataFrame
    df["cluster"] = y_predicted
    
    plot(df, y_predicted, km)


