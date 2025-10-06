import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
df_original = pd.DataFrame(iris.data, columns=iris.feature_names)

best_k = 3 

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(f'Clusters K-Means (k={best_k}) com PCA', fontsize=16)

pca_versions = [1, 2]

for i, n_components in enumerate(pca_versions):    
    pca = PCA(n_components=n_components, random_state=42)
    principal_components = pca.fit_transform(df_original)
    
    kmeans_pca = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters_pca = kmeans_pca.fit_predict(principal_components)
    
    ax = axes[i]
    
    if n_components == 1:
        ax.scatter(principal_components, np.zeros_like(principal_components), 
                   c=clusters_pca, cmap='viridis', s=50, alpha=0.8)
        ax.scatter(kmeans_pca.cluster_centers_[:, 0], np.zeros_like(kmeans_pca.cluster_centers_[:, 0]), 
                   color='red', marker='X', s=200, edgecolor='k', label='Centróides')
        ax.set_title(f'PCA: 1 Componente')
        ax.set_xlabel('Componente Principal 1')
        ax.set_yticks([]) 
    else: 
        ax.scatter(principal_components[:, 0], principal_components[:, 1], 
                   c=clusters_pca, cmap='viridis', s=50, alpha=0.8)
        ax.scatter(kmeans_pca.cluster_centers_[:, 0], kmeans_pca.cluster_centers_[:, 1], 
                   color='red', marker='X', s=200, edgecolor='k', label='Centróides')
        ax.set_title(f'PCA: 2 Componentes')
        ax.set_xlabel('Componente Principal 1')
        ax.set_ylabel('Componente Principal 2')
    
    ax.legend()
    ax.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()