import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

plt.scatter(df["sepal length (cm)"], df["sepal width (cm)"])

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(
    df[
        [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal width (cm)",
            "petal width (cm)",
        ]
    ]
)

df["cluster"] = y_predicted

df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

plt.scatter(
    df1["sepal length (cm)"],
    df1["sepal width (cm)"],
    color="green",
)
plt.scatter(
    df2["sepal length (cm)"],
    df2["sepal width (cm)"],
    color="red",
)
plt.scatter(
    df3["sepal length (cm)"],
    df3["sepal width (cm)"],
    color="black",
)

plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")

print(silhouette_score(df, y_predicted))

plt.show()
