import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# load embeddings
emb = np.load("query_embeddings.npy")

print("Embedding shape:", emb.shape)

# reduce dimensions to 2 using PCA
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(emb)

# plot
plt.figure(figsize=(8,6))
plt.scatter(emb_2d[:,0], emb_2d[:,1], s=5, alpha=0.6)
plt.title("PCA Visualization of Query Embeddings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()