#kmeanimport random
import numpy as np
import matplotlib.pyplot as plt

# Generate random 2D data
num_points = 100
x = np.random.randint(1, 100, num_points)
y = np.random.randint(1, 100, num_points)

# Combine x and y into a single array of points
data = np.array(list(zip(x, y)))
print(data)

# Plot the random data points before clustering
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', alpha=0.5)
plt.title('Random 2D Data Points Before Clustering')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()

# Function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
   return np.sqrt(np.sum((point1 - point2) ** 2))

# Function to assign clusters based on the closest centroid
def assign_clusters(data, centroids):
   clusters = [[] for _ in range(len(centroids))]
   for point in data:
       # Find the closest centroid
       distances = [euclidean_distance(point, centroid) for centroid in centroids]
       closest_centroid = np.argmin(distances)
       clusters[closest_centroid].append(point)
   return clusters

#Function to calculate new centroids as the mean of the points in each cluster
def calculate_centroids(clusters):
   new_centroids = []
   for cluster in clusters:
       if len(cluster) > 0:
           new_centroid = np.mean(cluster, axis=0)
           new_centroids.append(new_centroid)
       else:
           new_centroids.append(random.choice(data))  # Reinitialize centroid if no points assigned
   return new_centroids

# K-Means Algorithm Implementation
def kmeans(data, k, max_iterations=100):
   # Randomly select k initial centroids from the data
   centroids = random.sample(list(data), k)


   for iteration in range(max_iterations):
       print(f"Iteration {iteration + 1}:")


       # Assign clusters based on current centroids
       clusters = assign_clusters(data, centroids)


       # Calculate new centroids from clusters
       new_centroids = calculate_centroids(clusters)


       print("Current Centroids:", centroids)
       print("New Centroids:", new_centroids)


       # Check for convergence (if centroids do not change)
       if np.array_equal(new_centroids, centroids):
           print("Convergence reached.")
           break


       centroids = new_centroids


   return clusters, centroids

# Run K-means algorithm
k = 3  # Number of clusters
clusters, final_centroids = kmeans(data, k)

# Convert final_centroids to a NumPy array for plotting
final_centroids = np.array(final_centroids)

# Plot the clustered data points with centroids
plt.figure(figsize=(10, 6))
for i in range(k):
   cluster_points = np.array(clusters[i])
   plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')

plt.scatter(final_centroids[:, 0], final_centroids[:, 1], color='red', marker='X', s=200, label='Centroids')
plt.title('K-means Clustering Results')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()
