import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Read the dataset
iris = datasets.load_iris()
df = pd.DataFrame(iris.data)
df.columns=['Sepal length','Sepal width','Petal length','Petal width']
print(df.head())
X = iris.data
y = iris.target

# Run the k-means clustering with k=4
km = KMeans(n_clusters=4, random_state=42)
km.fit_predict(X)
score = silhouette_score(X, km.labels_, metric='euclidean')
print("Silhouette score(with k= 4 ):  ", score)
print("-----------------------------------------------------\n")

# Run the k-means clustering with k=2,3,4,5,6
range_n_clusters = [2, 3, 4, 5, 6]
silhouette_avg = []
for num_clusters in range_n_clusters:

    # initialize kmeans
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(df)
    cluster_labels = kmeans.labels_

    silhouette_avg.append(silhouette_score(df, cluster_labels))

# Visualization
plt.plot(range_n_clusters, silhouette_avg, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette score')
plt.title('Silhouette analysis For Optimal k')
plt.show()

k = 2
for i in silhouette_avg:
    print("Silhouette score(with k=", k, "):", i)
    k += 1