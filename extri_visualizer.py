"""COLMAP visualizer

Visualize COLMAP sparse reconstruction outputs. To get demo data, see `./assets/download_colmap_garden.sh`.
"""

import random
import time
from pathlib import Path
from typing import List, Dict

import imageio.v3 as iio
import numpy as np
import tyro
import cv2
from tqdm.auto import tqdm

import viser
import viser.transforms as tf
from viser.extras.colmap import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)


def load_camera_from_yaml(
    intri_path: str, extri_path: str, camera_ids: List[str] = None
) -> Dict:
    """
    load intrinsics and extrinsics from yaml file, the intri yaml file should be like this:
        %YAML:1.0
        ---
        K_01: !!opencv-matrix
        rows: 3
        cols: 3
        dt: d
        data: [2714.137788, 0.000000, 911.738749, 0.000000, 2683.085000, 588.723669, 0.000000, 0.000000, 1.000000]
        dist_01: !!opencv-matrix
        rows: 1
        cols: 5
        dt: d
        data: [0.000000, 0.000000, 0.000000, -0.000000, 0.000000]
        names:
        - "01"

    if given camera_id, the function will only load the given camera's intrinsics and extrinsics, otherwise, it will load all the cameras' intrinsics and extrinsics

    Args:
        intri_path: path to the intrinsics yaml file
        extri_path: path to the extrinsics yaml file
        camera_id: camera id
    """
    fs_intri = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
    fs_extri = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)

    intri_data = {}
    extri_data = {}

    # 读取内参数据
    for name in fs_intri.root().keys():
        if name == "names":
            intri_data[name] = []
            for i in range(fs_intri.getNode(name).size()):
                intri_data[name].append(fs_intri.getNode(name).at(i).string())
            continue
        intri_data[name] = fs_intri.getNode(name).mat()

    # 读取外参数据
    for name in fs_extri.root().keys():
        if name == "names":
            extri_data[name] = []
            for i in range(fs_extri.getNode(name).size()):
                extri_data[name].append(fs_extri.getNode(name).at(i).string())
            continue
        extri_data[name] = fs_extri.getNode(name).mat()

    fs_intri.release()
    fs_extri.release()

    if "names" not in intri_data or "names" not in extri_data:
        raise ValueError("Invalid intrinsics or extrinsics yaml file")

    # check exist camera_id, should be in both intri_data and extri_data
    intri_camera_ids = intri_data["names"]
    extri_camera_ids = extri_data["names"]
    file_camera_ids = list(set(intri_camera_ids) & set(extri_camera_ids))

    if camera_ids is None:
        camera_ids = file_camera_ids
    else:
        # check if all camera_ids are in file_camera_ids
        if not all(camera_id in file_camera_ids for camera_id in camera_ids):
            raise ValueError(f"Invalid camera id: {camera_ids}")

    camera_params = {}
    # for each camera_id, load the intrinsics and extrinsics
    for camera_id in camera_ids:
        param = {}
        param["K"] = intri_data[f"K_{camera_id}"]
        param["dist"] = intri_data[f"dist_{camera_id}"]
        param["Rvec"] = extri_data[f"R_{camera_id}"]
        param["T"] = extri_data[f"T_{camera_id}"]
        camera_params[camera_id] = param

    return camera_params


def main(
    intri_path: Path,
    extri_path: Path,
    H: int,
    W: int,
) -> None:
    """Visualize COLMAP sparse reconstruction outputs.

    Args:
        colmap_path: Path to the COLMAP reconstruction directory.
        images_path: Path to the COLMAP images directory.
        downsample_factor: Downsample factor for the images.
    """
    server = viser.ViserServer(port=8002)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Load the colmap info.
    cameras = load_camera_from_yaml(intri_path, extri_path)
    gui_reset_up = server.gui.add_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
            [0.0, -1.0, 0.0]
        )

    gui_frames = server.gui.add_slider(
        "Max frames",
        min=1,
        max=len(cameras),
        step=1,
        initial_value=min(len(cameras), 100),
    )

    frames: List[viser.FrameHandle] = []

    def visualize_frames() -> None:
        """Send all COLMAP elements to viser for visualization. This could be optimized
        a ton!"""

        # Remove existing image frames.
        for frame in frames:
            frame.remove()
        frames.clear()

        # Interpret the images and cameras.
        img_ids = list(cameras.keys())
        random.shuffle(img_ids)
        img_ids = sorted(img_ids[: gui_frames.value])

        def attach_callback(
            frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle
        ) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        for img_id in tqdm(img_ids):
            # if '77' in img_id or '80' in img_id or '91' in img_id:
            #     continue
            if '77' in img_id:
                continue
            cam = cameras[img_id]

            T_world_camera = tf.SE3.from_rotation_and_translation(
                tf.SO3.from_matrix(cv2.Rodrigues(cam['Rvec'])[0]), cam['T'][:, 0]
            ).inverse()
            frame = server.scene.add_frame(
                f"/colmap/frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.1,
                axes_radius=0.005,
            )
            frames.append(frame)

            fy = cam['K'][1, 1]
            frustum = server.scene.add_camera_frustum(
                f"/colmap/frame_{img_id}/frustum",
                fov=2 * np.arctan2(H / 2, fy),
                aspect=W / H,
                scale=0.15,
                image=None,
            )
            attach_callback(frustum, frame)

    need_update = True

    @gui_frames.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    while True:
        if need_update:
            need_update = False
            visualize_frames()

        time.sleep(1e-3)


if __name__ == "__main__":
    tyro.cli(main)
