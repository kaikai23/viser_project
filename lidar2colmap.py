import os
import laspy
import numpy as np
from tqdm import tqdm


# Function to read a single .las file and extract points and colors
def read_las_file(file_path):
	las = laspy.read(file_path)
	points = np.vstack((las.x, las.y, las.z)).transpose()  # Extract (x, y, z) coordinates
	colors = np.vstack((las.red, las.green, las.blue)).transpose()  # Extract (R, G, B) colors
	return points, colors


# Function to read all .las files in a directory and extract points and colors
def read_las_files(directory):
	las_files = [f for f in os.listdir(directory) if f.endswith('.las')]
	all_points = []
	all_colors = []

	for file in tqdm(las_files, desc='Read las files'):
		file_path = os.path.join(directory, file)
		points, colors = read_las_file(file_path)
		all_points.append(points)
		all_colors.append(colors)

	# Combine points and colors from all files
	all_points = np.vstack(all_points)
	all_colors = np.vstack(all_colors)
	return all_points, all_colors


# Function to randomly downsample points and colors
def downsample_points(points, colors, target_num_points):
	num_points = points.shape[0]
	if num_points <= target_num_points:
		return points, colors  # No downsampling needed

	# Randomly select `target_num_points` indices
	indices = np.random.choice(num_points, target_num_points, replace=False)
	return points[indices], colors[indices]


# Function to convert point cloud to COLMAP format
def convert_to_colmap(points, colors, output_file):
	with open(output_file, 'w') as f:
		# Write header
		f.write("# 3D point list with one line of data per point:\n")
		f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
		f.write(f"# Number of points: {len(points)}, mean track length: 0.0\n")

		# Write each point
		for i, (point, color) in enumerate(zip(tqdm(points, desc='Write points'), colors)):
			point_id = i + 1  # POINT3D_ID (1-based index)
			x, y, z = point
			r, g, b = color
			error = -1  # Set ERROR to 0
			track = f"1 {i}"  # Dummy image ID and 2D point index
			f.write(f"{point_id} {x} {y} {z} {r} {g} {b} {error} {track}\n")


# Main script
if __name__ == "__main__":
	# Input path (can be a directory or a .las file)
	input_path = "/Users/yifei/Downloads/huace/LAS"  # Replace with your input path
	# Output COLMAP file
	output_file = "/Users/yifei/Downloads/huace/AT/内置相机/colmap/points3D.txt"
	# Target number of points after downsampling
	target_num_points = 1000000000  # Adjust this value as needed
	# Lidar origin
	# lidar_origin_x = 495146.0
	# lidar_origin_y = 2493792.0
	lidar_origin_x = 803716.403093
	lidar_origin_y = 2495795.418298
	lidar_origin_z = 2.655162

	# Check if the input path is a directory or a file
	if os.path.isdir(input_path):
		print(f"Processing all .las files in directory: {input_path}")
		points, colors = read_las_files(input_path)
	elif os.path.isfile(input_path) and input_path.endswith('.las'):
		print(f"Processing single .las file: {input_path}")
		points, colors = read_las_file(input_path)
	else:
		raise ValueError("Input path must be a directory or a .las file.")

	points[:, 0] -= lidar_origin_x
	points[:, 1] -= lidar_origin_y

	# Randomly downsample the points
	print(f"Downsampling from {points.shape[0]} points to {target_num_points} points...")
	points, colors = downsample_points(points, colors, target_num_points)

	# Convert to COLMAP format
	convert_to_colmap(points, colors, output_file)

	print(f"COLMAP 3D point file saved to {output_file}")
