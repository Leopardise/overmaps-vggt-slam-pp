#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import gtsam
import matplotlib.pyplot as plt
import torch
import open3d as o3d  # (import kept; not used directly here but left as before)
import viser
import viser.transforms as viser_tf
from termcolor import colored
from typing import Dict, List

from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from vggt_slam.loop_closure import ImageRetrieval
from vggt_slam.frame_overlap import FrameTracker
from vggt_slam.map import GraphMap
from vggt_slam.submap import Submap
from vggt_slam.h_solve import ransac_projective
from vggt_slam.gradio_viewer import TrimeshViewer


class Viewer:
    def __init__(self, port: int = 8080):
        print(f"Starting viser server on port {port}")
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        self.gui_show_frames = self.server.gui.add_checkbox("Show Cameras", initial_value=True)
        self.gui_show_frames.on_update(self._on_update_show_frames)

        self.submap_frames: Dict[int, List[viser.FrameHandle]] = {}
        self.submap_frustums: Dict[int, List[viser.CameraFrustumHandle]] = {}
        self.random_colors = np.random.randint(0, 256, size=(250, 3), dtype=np.uint8)

    def visualize_frames(self, extrinsics: np.ndarray, images_: np.ndarray, submap_id: int, image_scale: float = 0.5) -> None:
        if isinstance(images_, torch.Tensor):
            images_ = images_.cpu().numpy()

        if submap_id not in self.submap_frames:
            self.submap_frames[submap_id] = []
            self.submap_frustums[submap_id] = []

        S = extrinsics.shape[0]
        for img_id in range(S):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            frame_name = f"submap_{submap_id}/frame_{img_id}"
            frustum_name = f"{frame_name}/frustum"

            frame_axis = self.server.scene.add_frame(
                frame_name,
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frame_axis.visible = self.gui_show_frames.value
            self.submap_frames[submap_id].append(frame_axis)

            img = images_[img_id]
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)
            img_resized = cv2.resize(img, (int(w * image_scale), int(h * image_scale)), interpolation=cv2.INTER_AREA)

            frustum = self.server.scene.add_camera_frustum(
                frustum_name,
                fov=fov,
                aspect=w / h,
                scale=0.05,
                image=img_resized,
                line_width=3.0,
                color=self.random_colors[submap_id % len(self.random_colors)]
            )
            frustum.visible = self.gui_show_frames.value
            self.submap_frustums[submap_id].append(frustum)

    def _on_update_show_frames(self, _):
        visible = self.gui_show_frames.value
        for frames in self.submap_frames.values():
            for f in frames: f.visible = visible
        for frustums in self.submap_frustums.values():
            for fr in frustums: fr.visible = visible


class Solver:
    """VGGT-SLAM++ front-end: VGGT submaps + Sim(3) odometry (paper Sec. 3, App. A6)."""
    def __init__(self,
                 init_conf_threshold: float,
                 use_point_map: bool = False,
                 visualize_global_map: bool = False,
                 use_sim3: bool = True,
                 gradio_mode: bool = False,
                 vis_stride: int = 1,
                 vis_point_size: float = 0.001,
                 # NEW:
                 depth_min: float = 0.2,
                 depth_max: float = 50.0):

        self.init_conf_threshold = init_conf_threshold
        self.use_point_map = use_point_map
        self.gradio_mode = gradio_mode

        self.viewer = TrimeshViewer() if self.gradio_mode else Viewer()
        self.flow_tracker = FrameTracker()
        self.map = GraphMap()
        # Paper App. A6: Sim(3) geodesic optimisation, not SL(4).
        self.use_sim3 = True if use_sim3 else True
        from vggt_slam.graph_sim3 import PoseGraph
        self.graph = PoseGraph()

        # RGB SALAD retrieval is unused when w_loops = 0 (paper default).
        self.image_retrieval = ImageRetrieval(enable_salad=False)
        self._lc_already_added = set()
        self.current_working_submap = None

        self.first_edge = True
        self.T_w_kf_minus = None
        self.prior_pcd = None
        self.prior_conf = None

        self.vis_stride = vis_stride
        self.vis_point_size = vis_point_size

        # NEW: depth range for denoising (camera Z in meters)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)

        print("Starting viser server...")

    def get_graph_map(self):
        return self.map

    # ---- viz helpers ----
    def set_point_cloud(self, points_in_world_frame, points_colors, name, point_size):
        if self.gradio_mode:
            self.viewer.add_point_cloud(points_in_world_frame, points_colors)
        else:
            self.viewer.server.scene.add_point_cloud(
                name="pcd_" + name,
                points=points_in_world_frame,
                colors=points_colors,
                point_size=point_size,
                point_shape="circle",
            )

    def set_submap_point_cloud(self, submap):
        pts = submap.get_points_in_world_frame(stride=self.vis_stride)
        cols = submap.get_points_colors(stride=self.vis_stride)
        self.set_point_cloud(pts, cols, str(submap.get_id()), self.vis_point_size)

    def set_submap_poses(self, submap):
        extrinsics = submap.get_all_poses_world()
        if self.gradio_mode:
            for i in range(extrinsics.shape[0]):
                self.viewer.add_camera_pose(extrinsics[i])
        else:
            images = submap.get_all_frames()
            self.viewer.visualize_frames(extrinsics, images, submap.get_id())

    def export_3d_scene(self, output_path="output.glb"):
        return self.viewer.export(output_path)

    def update_all_submap_vis(self):
        for sm in self.map.get_submaps():
            self.set_submap_point_cloud(sm)
            self.set_submap_poses(sm)

    def update_latest_submap_vis(self):
        sm = self.map.get_latest_submap()
        self.set_submap_point_cloud(sm)
        self.set_submap_poses(sm)

    # ---- slam update ----
    def add_points(self, pred_dict):
        images = pred_dict["images"]
        extrinsics_cam = pred_dict["extrinsic"]     # world -> cam (3x4 per frame)
        intrinsics_cam = pred_dict["intrinsic"]
        detected_loops = pred_dict["detected_loops"]

        # Build point map + per-pixel confidence
        if self.use_point_map:
            world_points = pred_dict["world_points"]          # (S, H, W, 3)
            conf = pred_dict["world_points_conf"]             # (S, H, W)
        else:
            depth_map = pred_dict["depth"]                    # (S, H, W, 1) in meters (camera Z)
            conf = pred_dict["depth_conf"]                    # (S, H, W)
            world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)

        # --- NEW: depth-based denoising (operate on confidence) ---
        try:
            if not self.use_point_map and ("depth" in pred_dict):
                z_cam = pred_dict["depth"][..., 0]  # (S,H,W) meters already camera-Z
            else:
                # compute Z_cam = R[2]*Xw + t[2] for each submap frame
                S, H, W, _ = world_points.shape
                z_cam = np.empty((S, H, W), dtype=np.float32)
                for s in range(S):
                    R = extrinsics_cam[s, :, :3]   # world->cam
                    t = extrinsics_cam[s, :,  3]
                    X = world_points[s].reshape(-1, 3)
                    Z = (X @ R[2, :].T) + t[2]     # only row-2
                    z_cam[s] = Z.reshape(H, W)

            m = np.ones_like(z_cam, dtype=bool)
            if np.isfinite(self.depth_min): m &= (z_cam >= self.depth_min)
            if np.isfinite(self.depth_max): m &= (z_cam <= self.depth_max)
            conf = conf * m.astype(conf.dtype)  # zero-out out-of-range
        except Exception as _:
            # fail-safe: don't crash SLAM if something odd happens
            pass
        # --- /NEW ---

        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        cam_to_world = closed_form_inverse_se3(extrinsics_cam)  # cam->world (3x4 per frame)

        new_pcd_num = self.current_working_submap.get_id()

        if self.first_edge:
            self.first_edge = False
            self.prior_pcd = world_points[-1, ...].reshape(-1, 3)
            self.prior_conf = conf[-1, ...].reshape(-1)

            H_w_submap = np.eye(4)
            self.graph.add_homography(new_pcd_num, H_w_submap)
            self.graph.add_prior_factor(new_pcd_num, H_w_submap, self.graph.anchor_noise)
        else:
            prior_pcd_num = self.map.get_largest_key()
            prior_submap = self.map.get_submap(prior_pcd_num)

            current_pts = world_points[0, ...].reshape(-1, 3)
            # keep original "good_mask" logic
            good_mask = self.prior_conf > prior_submap.get_conf_threshold() * (
                conf[0, ...].reshape(-1) > prior_submap.get_conf_threshold()
            )

            if self.use_sim3:
                R_temp = prior_submap.poses[prior_submap.get_last_non_loop_frame_index()][0:3, 0:3]
                t_temp = prior_submap.poses[prior_submap.get_last_non_loop_frame_index()][0:3, 3]
                T_temp = np.eye(4); T_temp[0:3, 0:3] = R_temp; T_temp[0:3, 3] = t_temp
                T_temp = np.linalg.inv(T_temp)
                scale_factor = np.mean(
                    np.linalg.norm((T_temp[0:3, 0:3] @ self.prior_pcd[good_mask].T).T + T_temp[0:3, 3], axis=1)
                    / np.linalg.norm(current_pts[good_mask], axis=1)
                )
                print(colored("scale factor", 'green'), scale_factor)
                H_relative = np.eye(4); H_relative[0:3, 0:3] = R_temp; H_relative[0:3, 3] = t_temp
                world_points *= scale_factor
                cam_to_world[:, 0:3, 3] *= scale_factor
            else:
                H_relative = ransac_projective(current_pts[good_mask], self.prior_pcd[good_mask])

            H_w_submap = prior_submap.get_reference_homography() @ H_relative

            non_lc_frame = self.current_working_submap.get_last_non_loop_frame_index()
            pts_cam0_camn = world_points[non_lc_frame, ...].reshape(-1, 3)
            self.prior_pcd = pts_cam0_camn
            self.prior_conf = conf[non_lc_frame, ...].reshape(-1)

            self.graph.add_homography(new_pcd_num, H_w_submap)
            self.graph.add_between_factor(prior_pcd_num, new_pcd_num, H_relative, self.graph.relative_noise)
            print("added between factor", prior_pcd_num, new_pcd_num)

        # store in submap
        self.current_working_submap.set_reference_homography(H_w_submap)
        self.current_working_submap.add_all_poses(cam_to_world)
        self.current_working_submap.add_all_points(world_points, colors, conf, self.init_conf_threshold, intrinsics_cam)
        self.current_working_submap.set_conf_masks(conf)

        # loop closures (unchanged)
        for index, loop in enumerate(detected_loops):
            assert loop.query_submap_id == self.current_working_submap.get_id()
            loop_index = self.current_working_submap.get_last_non_loop_frame_index() + index + 1

            if self.use_sim3:
                pose_world_detected = self.map.get_submap(loop.detected_submap_id).get_pose_subframe(loop.detected_submap_frame)
                pose_world_query = self.current_working_submap.get_pose_subframe(loop_index)
                pose_world_detected = gtsam.Pose3(pose_world_detected)
                pose_world_query = gtsam.Pose3(pose_world_query)
                H_relative_lc = pose_world_detected.between(pose_world_query).matrix()
            else:
                points_world_detected = self.map.get_submap(loop.detected_submap_id).get_frame_pointcloud(loop.detected_submap_frame).reshape(-1, 3)
                points_world_query = self.current_working_submap.get_frame_pointcloud(loop_index).reshape(-1, 3)
                H_relative_lc = ransac_projective(points_world_query, points_world_detected)

            self.graph.add_between_factor(loop.detected_submap_id, loop.query_submap_id, H_relative_lc, self.graph.relative_noise)
            self.graph.increment_loop_closure()
            print("added loop closure factor", loop.detected_submap_id, loop.query_submap_id)

        self.map.add_submap(self.current_working_submap)

    def run_predictions(self, image_names, model, max_loops):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        images = load_and_preprocess_images(image_names).to(device)
        print(f"Preprocessed images shape: {images.shape}")

        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

        new_pcd_num = self.map.get_largest_key() + 1
        new_submap = Submap(new_pcd_num)
        new_submap.add_all_frames(images)
        new_submap.set_frame_ids(image_names)
        # Paper: w_loops = 0; loop closure is handled by the DEM/AnyLoc back-end.
        detected_loops = []
        if max_loops and int(max_loops) > 0:
            if not self.image_retrieval.enable_salad:
                self.image_retrieval = ImageRetrieval(enable_salad=True)
            new_submap.set_all_retrieval_vectors(
                self.image_retrieval.get_all_submap_embeddings(new_submap)
            )
            detected_loops = self.image_retrieval.find_loop_closures(
                self.map, new_submap, max_loop_closures=max_loops
            )
            if len(detected_loops) > 0:
                print(colored("detected_loops", "yellow"), detected_loops)
        retrieved_frames = self.map.get_frames_from_loops(detected_loops)

        new_submap.set_last_non_loop_frame_index(images.shape[0] - 1)
        if len(retrieved_frames) > 0:
            image_tensor = torch.stack(retrieved_frames)
            images = torch.cat([images, image_tensor], dim=0)
            new_submap.add_all_frames(images)

        self.current_working_submap = new_submap

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype, enabled=torch.cuda.is_available()):
                predictions = model(images)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        predictions["detected_loops"] = detected_loops

        for k in list(predictions.keys()):
            if isinstance(predictions[k], torch.Tensor):
                predictions[k] = predictions[k].float().cpu().numpy() if predictions[k].shape[0] != 1 else predictions[k].float().cpu().numpy().squeeze(0)

        return predictions

    def apply_external_loops_if_any(self, csv_path: str, optimize_each: bool = True) -> int:
        """Insert DEM/AnyLoc loop edges (paper Sec. 3, Eq. 4) and re-optimise Sim(3)."""
        from vggt_slam.anyloc_csv import add_loop_edges_from_csv
        if not csv_path:
            return 0
        return add_loop_edges_from_csv(self, csv_path, optimize_each=optimize_each)
