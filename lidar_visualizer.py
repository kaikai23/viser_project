import os
import laspy
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm

# Function to read all .las files in the current directory
def read_las_files(directory):
	las_files = [f for f in os.listdir(directory) if f.endswith('.las')]
	point_clouds = []
	N = 0
	for file in las_files:
		las = laspy.read(os.path.join(directory, file))
		points = np.vstack((las.x, las.y, las.z)).transpose()
		point_clouds.append(points)
		N += len(points)
		print(f'read {len(points)} points from {file}')
	print(f'Finish reading {N} points')
	return point_clouds


# Function to assign distinct colors to each point cloud
def assign_colors(point_clouds):
	colors = plt.get_cmap('tab20', len(point_clouds))  # Use a colormap with enough distinct colors
	colored_point_clouds = []
	for i, points in tqdm(enumerate(point_clouds), desc="assign colors"):
		color = colors(i)[:3]  # Get RGB (ignore alpha)
		colored_points = np.hstack((points, np.tile(color, (points.shape[0], 1))))  # Add color to each point
		colored_point_clouds.append(colored_points)
	return colored_point_clouds


# Function to visualize point clouds with Open3D
def visualize_point_clouds(colored_point_clouds):
	geometries = []
	for points in tqdm(colored_point_clouds, desc='visualize point clouds'):
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Set points
		pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6])  # Set colors
		geometries.append(pcd)

	# Visualize all point clouds together
	o3d.visualization.draw_geometries(geometries)


# Main script
if __name__ == "__main__":
	directory = '/Users/yifei/Downloads/huace/LAS'  # Current directory
	point_clouds = read_las_files(directory)
	colored_point_clouds = assign_colors(point_clouds)
	visualize_point_clouds(colored_point_clouds)