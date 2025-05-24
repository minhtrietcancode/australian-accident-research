'''
- Call the function from task11_1.py to get the dataframe
- The visualization and clustering plan:
    + Step 1: Create a 3D scatter plot visualization - REMEMBER this is a 3D plot and the 
      x,y,z axis are the LIGHT_RISK, GEOMETRY_RISK, SURFACE_RISK respectively
      - Each point represents a unique combination of the three conditions
      - Use a color gradient from light to dark based on the SEVERE_ACCIDENT_RATE
      - This provides an overview of the risk distribution in the feature space
      --> save as task11_2_overview_scatter_plot.png 
      
    + Step 2: Determine optimal k for k-means clustering
      - Run k-means with different values of k
      - Plot Sum of Squared Errors against k to find the elbow point
      - Select the best k based on this analysis
      --> save as task11_2_elbow_plot.png
      
    + Step 3: Perform k-means clustering with the optimal k
      - Apply the algorithm to group similar combinations together
      - Create a new 3D scatter plot with colors representing cluster membership
      - Analyze the characteristics of each cluster to identify patterns
        --> save as task11_2_clustered_scatter_plot.png
      
    + The 3D feature space will use:
      - X-axis: LIGHT_RISK
      - Y-axis: GEOMETRY_RISK
      - Z-axis: SURFACE_RISK
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
from task11_1 import task11_1

# Call the function from task11_1.py to get the dataframe
df = task11_1()

# Make sure we have the required columns
required_columns = ['LIGHT_RISK', 'GEOMETRY_RISK', 'SURFACE_RISK', 'SEVERE_ACCIDENT_RATE']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column {col} not found in dataframe")

# Extract the features for clustering
X = df[['LIGHT_RISK', 'GEOMETRY_RISK', 'SURFACE_RISK']].values

# Standardize the features for better clustering results
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 1: Create a 3D scatter plot visualization with color gradient based on SEVERE_ACCIDENT_RATE
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create a scatter plot with color based on SEVERE_ACCIDENT_RATE
scatter = ax.scatter(
    X[:, 0],  # LIGHT_RISK
    X[:, 1],  # GEOMETRY_RISK
    X[:, 2],  # SURFACE_RISK
    c=df['SEVERE_ACCIDENT_RATE'],
    cmap='viridis',
    s=50,
    alpha=0.7
)

# Add colorbar to show the risk score scale
cbar = plt.colorbar(scatter)
cbar.set_label('SEVERE_ACCIDENT_RATE')

# Set labels and title
ax.set_xlabel('LIGHT_RISK')
ax.set_ylabel('GEOMETRY_RISK')
ax.set_zlabel('SURFACE_RISK')
ax.set_title('3D Scatter Plot of Risk Factors')

# Save the figure
plt.tight_layout()
plt.savefig('Task 11 Clustering Analysis/task11_2_overview_scatter_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Step 2: Determine optimal k for k-means clustering
# Calculate Sum of Squared Errors (SSE) for different values of k
sse = []
k_range = range(1, 11)  # Test k from 1 to 10

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

# Plot the Elbow method to find the optimal k
plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal k')

# Find the elbow point (simple approach - find point of maximum curvature)
# Calculate the rate of decrease
differences = np.diff(sse)
differences_of_differences = np.diff(differences)
elbow_index = np.argmax(differences_of_differences) + 1
optimal_k = k_range[elbow_index]

# Add a vertical line at the optimal k
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Task 11 Clustering Analysis/task11_2_elbow_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Step 3: Perform k-means clustering with the optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to the dataframe
df['cluster'] = cluster_labels

# Create a 3D scatter plot with colors representing cluster membership
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Create a colormap with distinct colors for each cluster
colors = cm.tab10(np.linspace(0, 1, optimal_k))

# Plot each cluster with a different color
for i in range(optimal_k):
    cluster_points = df[df['cluster'] == i]
    ax.scatter(
        cluster_points['LIGHT_RISK'],
        cluster_points['GEOMETRY_RISK'],
        cluster_points['SURFACE_RISK'],
        s=50,
        color=colors[i],
        label=f'Cluster {i}',
        alpha=0.7
    )

# Plot cluster centers
centers = kmeans.cluster_centers_
# Inverse transform to get back to original scale
centers_original = scaler.inverse_transform(centers)
ax.scatter(
    centers_original[:, 0],
    centers_original[:, 1],
    centers_original[:, 2],
    s=200,
    marker='X',
    color='red',
    label='Cluster Centers',
    alpha=1.0
)

# Set labels and title
ax.set_xlabel('LIGHT_RISK')
ax.set_ylabel('GEOMETRY_RISK')
ax.set_zlabel('SURFACE_RISK')
ax.set_title(f'K-Means Clustering with k={optimal_k}')
plt.legend()

# Save the figure
plt.tight_layout()
plt.savefig('Task 11 Clustering Analysis/task11_2_clustered_scatter_plot.png', dpi=300, bbox_inches='tight')

# Output the top rows for each cluster
for i in range(optimal_k):
    cluster_data = df[df['cluster'] == i].loc[df['COUNT'] >= 10]
    
    # Sort the cluster data by SEVERE_ACCIDENT_RATE in descending order
    cluster_data = cluster_data.sort_values(by='SEVERE_ACCIDENT_RATE', ascending=False)
    
    # Get only the top 9 rows
    cluster_data = cluster_data.head(10)
    
    # Save to CSV
    cluster_data.to_csv(f'Task 11 Clustering Analysis/task11_2_cluster_{i}.csv', index=False)

print("CSV files saved successfully!")



