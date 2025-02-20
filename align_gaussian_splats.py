"""WebGL-based Gaussian splat rendering. This is still under developmentt."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TypedDict, Union

import numpy as np
import numpy.typing as npt
import tyro
from plyfile import PlyData, PlyElement

import viser
from viser import transforms as tf


class SplatFile(TypedDict):
    """Data loaded from an antimatter15-style splat file."""

    centers: npt.NDArray[np.floating]
    """(N, 3)."""
    rgbs: npt.NDArray[np.floating]
    """(N, 3). Range [0, 1]."""
    opacities: npt.NDArray[np.floating]
    """(N, 1). Range [0, 1]."""
    scales: npt.NDArray[np.floating]
    """(N, 3)"""
    wxyzs: npt.NDArray[np.floating]
    """(N, 4)"""
    covariances: npt.NDArray[np.floating]
    """(N, 3, 3)."""


def load_splat_file(splat_path: Path, center: bool = False) -> SplatFile:
    """Load an antimatter15-style splat file."""
    start_time = time.time()
    splat_buffer = splat_path.read_bytes()
    bytes_per_gaussian = (
        # Each Gaussian is serialized as:
        # - position (vec3, float32)
        3 * 4
        # - xyz (vec3, float32)
        + 3 * 4
        # - rgba (vec4, uint8)
        + 4
        # - ijkl (vec4, uint8), where 0 => -1, 255 => 1.
        + 4
    )
    assert len(splat_buffer) % bytes_per_gaussian == 0
    num_gaussians = len(splat_buffer) // bytes_per_gaussian

    # Reinterpret cast to dtypes that we want to extract.
    splat_uint8 = np.frombuffer(splat_buffer, dtype=np.uint8).reshape(
        (num_gaussians, bytes_per_gaussian)
    )
    scales = splat_uint8[:, 12:24].copy().view(np.float32)
    wxyzs = splat_uint8[:, 28:32] / 255.0 * 2.0 - 1.0
    Rs = tf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    centers = splat_uint8[:, 0:12].copy().view(np.float32)
    if center:
        centers -= np.mean(centers, axis=0, keepdims=True)
    print(
        f"Splat file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return {
        "centers": centers,
        # Colors should have shape (N, 3).
        "rgbs": splat_uint8[:, 24:27] / 255.0,
        "opacities": splat_uint8[:, 27:28] / 255.0,
        "scales": scales,
        "wxyzs": wxyzs,
        # Covariances should have shape (N, 3, 3).
        "covariances": covariances,
    }


def load_ply_file(ply_file_path: Path, center: bool = False) -> SplatFile:
    """Load Gaussians stored in a PLY file."""
    start_time = time.time()

    SH_C0 = 0.28209479177387814

    plydata = PlyData.read(ply_file_path)
    v = plydata["vertex"]
    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
    wxyzs = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    colors = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    opacities = 1.0 / (1.0 + np.exp(-v["opacity"][:, None]))

    Rs = tf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    if center:
        positions -= np.mean(positions, axis=0, keepdims=True)

    num_gaussians = len(v)
    print(
        f"PLY file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return {
        "centers": positions,
        "rgbs": colors,
        "opacities": opacities,
        "scales": scales,
        "wxyzs": wxyzs,
        "covariances": covariances,
    }


def main(splat_paths: tuple[Path, ...]) -> None:
    server = viser.ViserServer(port=8002)
    server.gui.configure_theme(dark_mode=True)
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

    # Initialize
    splats = []
    gs_handles = []
    need_update = []
    gui = []
    center_when_load = False
    for i in range(len(splat_paths)):
        splats.append(None)
        gs_handles.append(None)
        need_update.append(False)
        gui.append({})
    with server.gui.add_folder("Editable"):
        for i, splat_path in enumerate(splat_paths):
            with server.gui.add_folder(f"Splats_{i}"):
                if splat_path.suffix == ".splat":
                    splat_data = load_splat_file(splat_path, center=center_when_load)
                elif splat_path.suffix == ".ply":
                    splat_data = load_ply_file(splat_path, center=center_when_load)
                else:
                    raise SystemExit("Please provide a filepath to a .splat or .ply file.")
                splats[i] = splat_data

                server.scene.add_transform_controls(f"/{i}")
                gs_handles[i] = server.scene.add_gaussian_splats(
                    f"/{i}/gaussian_splats",
                    centers=splat_data["centers"],
                    rgbs=splat_data["rgbs"],
                    opacities=splat_data["opacities"],
                    covariances=splat_data["covariances"],
                )

                # Visible
                def attach_visible_callback(
                        gs_handle: viser.GaussianSplatHandle, vis_handle: viser.GuiCheckboxHandle
                ) -> None:
                    @vis_handle.on_update
                    def _(_):
                        gs_handle.visible = vis_handle.value
                gui[i]['visible'] = server.gui.add_checkbox("Visible", initial_value=True)
                attach_visible_callback(gs_handles[i], gui[i]['visible'])

                # Update Display Function
                def draw_splat_i(i: int) -> None:
                    # # Too Slow
                    # new_center = splats[i]["centers"] + np.array([gui[i]['pos_x'].value, gui[i]['pos_y'].value, gui[i]['pos_z'].value])
                    # gs_handles[i].remove()
                    # gs_handles[i] = server.scene.add_gaussian_splats(
                    #     f"/{i}/gaussian_splats",
                    #     centers=new_center,
                    #     rgbs=splats[i]["rgbs"],
                    #     opacities=splats[i]["opacities"],
                    #     covariances=splats[i]["covariances"],
                    # )
                    center = np.array([gui[i]['pos_x'].value,
                                           gui[i]['pos_y'].value,
                                           gui[i]['pos_z'].value])
                    wxyz = tf.SO3.from_rpy_radians(gui[i]['yaw'].value * np.pi / 180,
                                                   gui[i]['pitch'].value * np.pi / 180,
                                                   gui[i]['roll'].value * np.pi / 180).wxyz
                    server.scene.add_transform_controls(f"/{i}",
                                                        position=center,
                                                        wxyz=wxyz,
                                                        )
                def attach_draw_splat_callback(
                        handle: viser.GuiSliderHandle,
                        idx: int
                ) -> None:
                    @handle.on_update
                    def _(_):
                        draw_splat_i(idx)
                def scale_splat_i(i: int) -> None:
                    scales = splats[i]['scales'] * gui[i]['scale'].value
                    Rs = tf.SO3(splats[i]['wxyzs']).as_matrix()
                    covariances = np.einsum(
                        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
                    )
                    gs_handles[i].remove()
                    gs_handles[i] = server.scene.add_gaussian_splats(
                        f"/{i}/gaussian_splats",
                        centers=splats[i]['centers'] * gui[i]['scale'].value,
                        rgbs=splats[i]["rgbs"],
                        opacities=splats[i]["opacities"],
                        covariances=covariances,
                    )
                def attach_scale_splat_callback(
                        handle: viser.GuiNumberHandle,
                        idx: int
                ) -> None:
                    @handle.on_update
                    def _(_):
                        scale_splat_i(idx)
                # Scale
                gui[i]['scale'] = server.gui.add_number(
                    "scale",
                    initial_value=1.,
                    min=1e-6,
                    max=None,
                    step=0.1
                )
                attach_scale_splat_callback(gui[i]['scale'], i)
                # Position
                with server.gui.add_folder("Location"):
                    gui[i]['pos_x'] = server.gui.add_slider(
                        "pos_x", min=-10.0, max=10.0, step=0.05, initial_value=0.0
                    )
                    gui[i]['pos_y'] = server.gui.add_slider(
                        "pos_y", min=-10.0, max=10.0, step=0.05, initial_value=0.0
                    )
                    gui[i]['pos_z'] = server.gui.add_slider(
                        "pos_z", min=-10.0, max=10.0, step=0.05, initial_value=0.0
                    )
                    attach_draw_splat_callback(gui[i]['pos_x'], i)
                    attach_draw_splat_callback(gui[i]['pos_y'], i)
                    attach_draw_splat_callback(gui[i]['pos_z'], i)
                # Rotation
                with server.gui.add_folder("Rotation"):
                    gui[i]['yaw'] = server.gui.add_slider(
                        "yaw", min=-180.0, max=180.0, step=0.5, initial_value=0.0
                    )
                    gui[i]['pitch'] = server.gui.add_slider(
                        "pitch", min=-180.0, max=180.0, step=0.5, initial_value=0.0
                    )
                    gui[i]['roll'] = server.gui.add_slider(
                        "roll", min=-180.0, max=180.0, step=0.5, initial_value=0.0
                    )
                    attach_draw_splat_callback(gui[i]['yaw'], i)
                    attach_draw_splat_callback(gui[i]['pitch'], i)
                    attach_draw_splat_callback(gui[i]['roll'], i)
                # Save
                gui[i]['save'] = server.gui.add_button('Save Ply')
                def save_ply_i(idx):
                    """First Load Gaussians stored in a PLY file."""
                    ply_file_path = splat_paths[idx]
                    plydata = PlyData.read(ply_file_path)
                    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                                    np.asarray(plydata.elements[0]["y"]),
                                    np.asarray(plydata.elements[0]["z"])), axis=1)
                    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
                    f_dc = np.zeros((xyz.shape[0], 3))
                    f_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
                    f_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
                    f_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])
                    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
                    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
                    assert len(extra_f_names) == 45
                    f_rest = np.zeros((xyz.shape[0], len(extra_f_names)))
                    for ID, attr_name in enumerate(extra_f_names):
                        f_rest[:, ID] = np.asarray(plydata.elements[0][attr_name])
                    scale = np.stack((np.asarray(plydata.elements[0]["scale_0"]),
                                      np.asarray(plydata.elements[0]["scale_1"]),
                                      np.asarray(plydata.elements[0]["scale_2"])), axis=1)
                    wxyz = np.stack((np.asarray(plydata.elements[0]["rot_0"]),
                                     np.asarray(plydata.elements[0]["rot_1"]),
                                     np.asarray(plydata.elements[0]["rot_2"]),
                                     np.asarray(plydata.elements[0]["rot_3"]),), axis=1)
                    if center_when_load:
                        xyz -= np.mean(xyz, axis=0, keepdims=True)
                    gui_center = np.array([gui[idx]['pos_x'].value,
                                           gui[idx]['pos_y'].value,
                                           gui[idx]['pos_z'].value])
                    gui_wxyz = tf.SO3.from_rpy_radians(gui[idx]['yaw'].value * np.pi / 180,
                                                       gui[idx]['pitch'].value * np.pi / 180,
                                                       gui[idx]['roll'].value * np.pi / 180).wxyz
                    print('load ok')
                    # Apply changes
                    scale = scale + np.log(gui[idx]['scale'].value)
                    xyz = (tf.SO3(gui_wxyz).as_matrix() @ (gui[idx]['scale'].value*xyz).T).T + gui_center
                    rotation = tf.SO3(gui_wxyz).multiply(tf.SO3(wxyz)).wxyz
                    # Save
                    normals = np.zeros_like(xyz)
                    dtype_full = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2']
                    for k in range(45):
                        dtype_full.append(f'f_rest_{k}')
                    dtype_full.append('opacity')
                    dtype_full.append('scale_0')
                    dtype_full.append('scale_1')
                    dtype_full.append('scale_2')
                    dtype_full.append('rot_0')
                    dtype_full.append('rot_1')
                    dtype_full.append('rot_2')
                    dtype_full.append('rot_3')
                    dtype_full = [(attribute, 'f4') for attribute in dtype_full]
                    elements = np.empty(xyz.shape[0], dtype=dtype_full)
                    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
                    elements[:] = list(map(tuple, attributes))
                    el = PlyElement.describe(elements, 'vertex')
                    PlyData([el]).write(f'./Splat_{idx}.ply')
                def attach_save_ply_callback(
                        handle: viser.GuiButtonHandle,
                        idx: int
                ) -> None:
                    @handle.on_click
                    def _(_):
                        save_ply_i(idx)
                attach_save_ply_callback(gui[i]['save'], i)





    while True:
        for i in range(len(splat_paths)):
            if need_update[i]:
                draw_splat_i(i)
            time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)