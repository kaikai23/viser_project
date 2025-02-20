import numpy as np
import cv2
from viser.extras.colmap import read_cameras_text, read_images_text, read_points3D_text
from viser.extras.colmap import read_cameras_binary, read_images_binary, read_points3d_binary
from pathlib import Path
from tqdm import tqdm
from typing import Union


class Point3D:
	def __init__(self, id: int, xyz: np.ndarray, rgb: np.ndarray):
		self.id = id
		self.xyz = xyz
		self.rgb = rgb


def read_points3D_text(path: Union[str, Path]):
	"""
	see: src/base/reconstruction.cc
		void Reconstruction::ReadPoints3DText(const std::string& path)
		void Reconstruction::WritePoints3DText(const std::string& path)
	"""
	points3D = {}
	N = 0
	with open(path, "r") as fid:
		while True:
			line = fid.readline()
			if not line:
				break
			line = line.strip()
			if len(line) > 0 and line[0] != "#":
				elems = line.split()
				point3D_id = int(elems[0])
				xyz = np.array(tuple(map(float, elems[1:4])))
				rgb = np.array(tuple(map(int, elems[4:7])))
				points3D[point3D_id] = Point3D(
					id=point3D_id,
					xyz=xyz,
					rgb=rgb,
				)
			N += 1
			if N % 100000 == 0:
				print(N)
	return points3D


# Function to convert quaternion to rotation matrix (provided by COLMAP)
def qvec2rotmat(qvec):
	return np.array(
		[
			[
				1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
				2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
				2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
			],
			[
				2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
				1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
				2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
			],
			[
				2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
				2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
				1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
			],
		]
	)


# Main function to project points onto an image
def project_points_to_image(image_name, colmap_path, image_path):
	# Load COLMAP data
	cameras = read_cameras_binary(colmap_path / "cameras.bin")
	images = read_images_binary(colmap_path / "images.bin")
	points3d = read_points3d_binary(colmap_path / "points3D.bin")

	# Find the image data
	if image_name not in images:
		raise ValueError(f"Image {image_name} not found in images.txt")
	image_data = images[image_name]
	camera_id = image_data.camera_id
	camera = cameras[camera_id]

	# Intrinsic matrix
	fx, fy, cx, cy = camera.params[0], camera.params[1], camera.params[2], camera.params[3]
	K = np.array([[fx, 0, cx],
				  [0, fy, cy],
				  [0, 0, 1]])

	# Extrinsic parameters
	qvec = np.array([image_data.qw, image_data.qx, image_data.qy, image_data.qz])
	R = qvec2rotmat(qvec)  # Convert quaternion to rotation matrix
	t = np.array([image_data.tx, image_data.ty, image_data.tz])

	# Load image
	image = cv2.imread(image_path)

	# Project points
	for point_id, point in tqdm(points3d.items()):
		X, Y, Z = point.xyz
		color = point.rgb  # RGB color of the 3D point
		P = np.array([X, Y, Z, 1])  # Homogeneous coordinates
		p = K @ (R @ P[:3] + t)  # Projection
		p = p / p[2]  # Normalize by z
		u, v = int(p[0]), int(p[1])  # Convert to pixel coordinates

		# Draw point on image if within bounds
		if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
			cv2.circle(image, (u, v), 2, color.tolist(), -1)  # Use the point's color

	# Display image
	cv2.imshow('Projected Points', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# Example usage
image_name = '1732241439.392887.jpg'  # Replace with your image name
colmap_path = Path('/Users/yifei/Downloads/huace/AT/内置相机/colmap_bin')  # Replace with the path to your COLMAP data
image_path = '/Users/yifei/Downloads/huace/AT/内置相机/undistorted/1732241439.392887.jpg'  # Replace with the actual image path

project_points_to_image(image_name, colmap_path, image_path)
