import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import LabelEncoder
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA

# # Step 1: Load the dataset
# df = pd.read_csv("Mall_Customers.csv")

# # Step 2: Initial exploration
# print(df.head())
# print(df.info())
# print(df.describe())

# # Step 3: Encode 'Gender'
# df['Gender'] = LabelEncoder().fit_transform(df['Gender'])  # Male:1, Female:0

# # Step 4: Select features for clustering
# X = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# # Step 5: Elbow method to find optimal K
# inertia = []
# K_range = range(1, 11)

# for k in K_range:
#     model = KMeans(n_clusters=k, random_state=42)
#     model.fit(X)
#     inertia.append(model.inertia_)

# plt.figure(figsize=(8, 4))
# plt.plot(K_range, inertia, 'bo-')
# plt.xlabel('Number of Clusters K')
# plt.ylabel('Inertia')
# plt.title('Elbow Method For Optimal k')
# plt.show()

# # Step 6: Fit final KMeans model
# optimal_k = 5
# kmeans = KMeans(n_clusters=optimal_k, random_state=42)
# df['Cluster'] = kmeans.fit_predict(X)

# # Step 7: Visualize Clusters using PCA
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='Set1', s=100)
# plt.title("Customer Segments (PCA Reduced)")
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.legend(title="Cluster")
# plt.show()

# # Step 8: Analyze clusters
# print(df.groupby('Cluster').mean())

# Optional: Save the model
# import joblib
# joblib.dump(kmeans, "customer_segmentation_model.pkl")


df = pd.read_csv("Mall_Customers.csv")
# print(df.head())

df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

X = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

inertia = []
k_range = range(1, 11)

for k in k_range:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    inertia.append(model.inertia_)
    
plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel("Number of Clusters K")
plt.ylabel("Intertia")
plt.title("Elbow Method For Optimal k")
plt.show()

k= 5
kmeans = KMeans(n_clusters=k, random_state=42)
model = kmeans.fit(X)
df['Cluster'] = model.predict(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='Set1', s=100)
plt.title("Customer Segments (PCA Reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.show()


















