
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


cancer_df = pd.read_csv("glass.csv")
cancer_df.info()

numeric_columns = cancer_df.drop(['Type'], axis=1)


tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(numeric_columns)

pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(numeric_columns)

plt.figure(figsize=(12, 5))

# Gráfico T-SNE
plt.subplot(1, 2, 1)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cancer_df['Type'].map({'M': 'red', 'B': 'blue'}))
plt.title('T-SNE')

# Gráfico PCA
plt.subplot(1, 2, 2)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=glass_df['Type'])
plt.title('PCA')
