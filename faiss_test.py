import time

import faiss
import numpy as np

# As a rule of thumb there is no consistent improvement of the k-means quantizer beyond 20 iterations and 1000 * k training points.
ncentroids = 10
niter = 60
verbose = True
n = 200000
d = 64
x = np.random.rand(n, d).astype("float32")
d = x.shape[1]
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
t1 = time.time()
kmeans.train(x)
# nearest centroid for each line vector in x in I. D contains the squared L2 distances
D, I = kmeans.index.search(x, 1)
# print(D, I)
# print(max(I), min(I))
print(kmeans.centroids.shape)

index = faiss.IndexFlatL2(d)
index.add(x)
D, I = index.search(kmeans.centroids, 15)
print(D.shape, I[:10])

t2 = time.time()
print(t2 - t1)
