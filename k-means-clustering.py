import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Setup and Data Loading
# Ensure the plots directory exists
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load your dataset (update the filename if necessary)
df = pd.read_csv('kpop_cleaned_subset.csv') 

# 2. Feature Selection & Scaling
# We select numeric musical attributes for clustering
features = ['energy', 'danceability', 'acousticness', 'tempo', 'valence']
x = df[features].dropna() # Ensure no missing values

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# 3. Finding Optimal Clusters (Elbow Method)
inertia = []
K_range = range(1, 11)

for k in K_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(x_scaled)
    inertia.append(model.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Error)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.savefig('plots/k_means_elbow_method.png')
plt.close() # Close to save memory

# 4. Applying K-Means
# Based on common K-pop distributions, 3 or 4 clusters usually work well
optimal_k = 3 
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(x_scaled)

# 5. Visualizing and Saving Results
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df, 
    x='energy', 
    y='danceability', 
    hue='cluster', 
    palette='viridis', 
    alpha=0.7
)
plt.title(f'K-means Clustering (k={optimal_k}): Energy vs Danceability')
plt.legend(title='Cluster')
plt.savefig('plots/kmeans_clusters_energy_danceability.png')

plt.show()

# 6. Exporting the labeled data
df.to_csv('kpop_clustered_results.csv', index=False)

print("Analysis complete! Plots saved to the 'plots/' folder.")