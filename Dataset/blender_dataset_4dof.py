import glob
import json
import os
import math
import msgpack
import msgpack_numpy as m

from loguru import logger

import copy

from os import path as osp
from typing import Dict
from unicodedata import name
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2

from Dataset.globals import DATASET_DIR
from Dataset.preprocessing import resize_img_pair
from Utils.data_utils import (pose_inv, bbox_from_mask, crop, calculate_intrinsic_for_crop, calculate_rot_delta, 
                get_keypoint_indices, project_pointcloud, calculate_intrinsic_for_new_resolution, rot2rotvec)
from Utils.training_utils import encode_rotation_matrix

# from .debug_utils import estimate_correspondences, estimate_correspondences_diff_intr

m.patch()

class BlenderDataset_4dof(Dataset):

    def __init__(self,
                 label_type: str,
                 crop_margin: float = 0.05, #0.3
                 n_bins: int = 18,
                 debug: bool = False) -> None:
        
        self.dataset_dir = DATASET_DIR
        self.label_type = label_type

        self.idx = None
        self.crop_margin = crop_margin
        self.bin_number = n_bins
        self.debug_mode = debug

        self.scene_names = np.sort(glob.glob(os.path.join(self.dataset_dir, 'scene_*')))

    def __len__(self):
        return len(self.scene_names)

    def load_scene(self, scene_filename) -> dict:
        with open(scene_filename, "rb") as data_file:
            byte_data = data_file.read()
            data: dict = msgpack.unpackb(byte_data)

        # for k in data.keys():
        #     print(k)

        # print(data["colors"].shape)
        # print(data["depth"].shape)
        # print(data["colors"])
        # print(data["depth"])

        data["depth"][0] = data["depth"][0] * (data["depth"][0] < 5.0)
        data["depth"][1] = data["depth"][1] * (data["depth"][1] < 5.0)

        # print(data["depth"][0].dtype)
        # # print(data["depth"][0])
        # print(np.min(data["depth"][0]), np.max(data["depth"][0]))
        # plt.figure()
        # plt.imshow(data["colors"][0])
        # plt.figure()
        # plt.imshow(data["depth"][0])
        # plt.show()
        
        data.update({
            "rgb_0": data["colors"][0],
            "rgb_1": data["colors"][1],
            "depth_0": data["depth"][0] * data["cp_main_obj_segmaps"][0],
            "depth_1": data["depth"][1] * data["cp_main_obj_segmaps"][1]
        })
        data.pop("colors")
        data.pop("depth")
        return data
    
    def load_random_scene(self):
        self.idx = np.random.randint(0, len(self))
        scene_dir = os.path.join(self.dataset_dir, f"scene_{str(self.idx).zfill(7)}")
        return self.load_scene(scene_dir)

    def crop_object(self, data: dict):
        crop_data = {}

        rgb0 = data["rgb_0"].copy()
        depth0 = data["depth_0"].copy()
        segmap0 = data["cp_main_obj_segmaps"][0].copy()

        rgb1 = data["rgb_1"].copy()
        depth1 = data["depth_1"].copy()
        segmap1 = data["cp_main_obj_segmaps"][1].copy()

        bbox0 = bbox_from_mask(segmap0, margin=self.crop_margin)
        bbox1 = bbox_from_mask(segmap1, margin=self.crop_margin)

        rgb0, crop_data["depth_0"], crop_data["seg_0"], bbox0 = crop(bbox0, rgb0, depth0, segmap0)
        rgb1, crop_data["depth_1"], crop_data["seg_1"], bbox1 = crop(bbox1, rgb1, depth1, segmap1)

        crop_data["rgb_0"] = rgb0 / 255
        crop_data["rgb_1"] = rgb1 / 255
        
        # crop_data["intrinsics_0"] = calculate_intrinsic_for_crop(
        #     RENDERING_INTRINSIC.copy(), top=bbox0[1], left=bbox0[0]
        # )
        # crop_data["intrinsics_1"] = calculate_intrinsic_for_crop(
        #     RENDERING_INTRINSIC.copy(), top=bbox1[1], left=bbox1[0]
        # )
        crop_data["intrinsics_0"] = calculate_intrinsic_for_crop(
            data["intrinsic"].copy(), top=bbox0[1], left=bbox0[0]
        )
        crop_data["intrinsics_1"] = calculate_intrinsic_for_crop(
            data["intrinsic"].copy(), top=bbox1[1], left=bbox1[0]
        )

        resize_img_pair(crop_data, modality=6)

        return crop_data
    
    def project_pointclouds(self, crop_data: dict):
        K = crop_data['intrinsics_0']
        depth = crop_data["depth_0"]
        keypoints = get_keypoint_indices((depth.shape[1], depth.shape[0]))
        # print(keypoints.shape)
        projected_keypoints = project_pointcloud(keypoints, depth, K, 'm')
        crop_data['proj_0'] = projected_keypoints.reshape((depth.shape[1], depth.shape[0], 3)).transpose(2,0,1).astype(np.float32)
        # print(crop_data['proj_0'][crop_data["seg_0"]])

        K = crop_data['intrinsics_1']
        depth = crop_data["depth_1"]
        keypoints = get_keypoint_indices((depth.shape[1], depth.shape[0]))
        projected_keypoints = project_pointcloud(keypoints, depth, K, 'm')
        crop_data['proj_1'] = projected_keypoints.reshape((depth.shape[1], depth.shape[0], 3)).transpose(2,0,1).astype(np.float32)

    def get_transformation(self, data: dict):
        T_WC = data["T_WC_opencv"]
        T_CW = pose_inv(T_WC)
        T_C1 = T_CW @ data["T_WO_frame_1"]
        T_0C = pose_inv(data["T_WO_frame_0"]) @ T_WC
        T_delta_cam = T_C1 @ T_0C
        T_delta_base = T_WC @ T_delta_cam @ T_CW
        rotvec = rot2rotvec(T_delta_base[:3, :3])
        delta_magnitude = np.linalg.norm(rotvec)
        axis = rotvec / delta_magnitude
        z_axis = np.array([0,0,1])
        if np.dot(z_axis, axis) < 0:
            delta_magnitude *= -1
        target_position = data["T_WO_frame_1"][:3,3]
        t_W0 = data["T_WO_frame_0"][:3,3]
        return delta_magnitude * 180/np.pi, target_position, t_W0, T_delta_cam
    
    def get_rotation_label(self, angle):
        if self.label_type == 'class':
            angle += 45 #make sure the whole interval [-45, 45] is positive
            class_index = int(angle // (90/self.bin_number))
            class_labels = np.zeros(self.bin_number, dtype=np.float32)
            class_labels[class_index] = 1.0
            return class_labels
        elif self.label_type == 'reg':
            return angle

    def __getitem__(self, idx):
        # Check length of Dataset is respected
        # self.idx = np.random.randint(0,2)
        self.idx = idx
        if self.idx >= len(self):
            raise IndexError
        
        # scene_dir = os.path.join(self.dataset_dir, f"scene_{str(self.idx).zfill(7)}")
        # data = self.load_scene(scene_dir)

        is_valid_scene = False
        while not is_valid_scene:
            try:
                # Get all the data for current scene
                scene_dir = self.scene_names[self.idx]
                data = self.load_scene(scene_dir)
                crop_data = self.crop_object(data)        
                self.project_pointclouds(crop_data)
                rot_magnitude, target_position, t_w0, full_T_delta = self.get_transformation(data)
                assert rot_magnitude < 45.1, "The rotation magnitude is actually above 45 degrees"
                rot_labels = self.get_rotation_label(rot_magnitude)
                is_valid_scene = True
            except Exception as e:
                logger.warning(f"[SCENE {self.idx}] The following exception was found: \n{e}")
                # data = self.load_random_scene()
                self.idx = np.random.randint(0, len(self))

        crop_data["rgb_0"] *= crop_data["seg_0"]
        crop_data["proj_0"] *= crop_data["seg_0"]
        crop_data["rgb_1"] *= crop_data["seg_1"]
        crop_data["proj_1"] *= crop_data["seg_1"]

        # print(crop_data["seg_0"].shape)
        # print(crop_data["rgb_0"][np.repeat(crop_data["seg_0"][None], 3, axis=0)])

        # crop_data["rgb_0"][np.repeat(crop_data["seg_0"][None], 3, axis=0)] = np.random.rand()
        # crop_data["proj_0"] *= crop_data["seg_0"]
        # crop_data["rgb_1"][np.repeat(crop_data["seg_1"][None], 3, axis=0)] = np.random.rand(3)
        # crop_data["proj_1"] *= crop_data["seg_1"]

        # background = np.random.rand(*tuple(crop_data["rgb_0"].shape)) * (~crop_data["seg_0"]) #random
        # # background = np.ones(crop_data["rgb_0"].shape) * -1 * (~crop_data["seg_0"])
        # crop_data["rgb_0"] *= crop_data["seg_0"]
        # crop_data["rgb_0"] += background
        # crop_data["proj_0"] *= crop_data["seg_0"]
        # background = np.random.rand(*tuple(crop_data["rgb_0"].shape)) * (~crop_data["seg_1"]) #random
        # # background = np.ones(crop_data["rgb_0"].shape) * -1 * (~crop_data["seg_1"])
        # crop_data["rgb_1"] *= crop_data["seg_1"]
        # crop_data["rgb_1"] += background
        # crop_data["proj_1"] *= crop_data["seg_1"]

        # import sys
        # np.set_printoptions(threshold=sys.maxsize)

        # print(crop_data["proj_0"][0][crop_data["proj_0"][0] != 0])
        # print("\n\n\n\n")
        # print(crop_data["proj_0"][1][crop_data["proj_0"][0] != 0])
        # print("\n\n\n\n")
        # print(crop_data["proj_0"][2][crop_data["proj_0"][0] != 0])

        # check_dim = 0
        # proj0 = (np.expand_dims(crop_data["proj_0"][check_dim,:,:], -1) - np.min(crop_data["proj_0"][check_dim,:,:]))
        # proj0 /= np.max(proj0)*255
        # proj0 *= crop_data["seg_0"][:,:,None]
        # proj1 = (np.expand_dims(crop_data["proj_1"][check_dim,:,:], -1) - np.min(crop_data["proj_1"][check_dim,:,:]))
        # proj1 /= np.max(proj1)*255
        # proj1 *= crop_data["seg_1"][:,:,None]

        # # plt.figure()
        # # plt.imshow(crop_data["rgb_0"].transpose(1,2,0))
        # # plt.figure()
        # # plt.imshow(crop_data["rgb_1"].transpose(1,2,0))
        # plt.figure()
        # # print("min 0: ", np.min(proj0))
        # # print("min 1: ", np.min(proj1))
        # # print("max 0: ", np.max(proj0))
        # # print("max 1: ", np.max(proj1))
        # plt.imshow(proj0)
        # plt.figure()
        # plt.imshow(proj1)
        # print(np.max(crop_data["rgb_0"]))
        # plt.figure()
        # plt.imshow(crop_data["rgb_0"].transpose(1,2,0))
        # plt.figure()
        # plt.imshow(crop_data["rgb_1"].transpose(1,2,0))
        # plt.show()

        if self.debug_mode:
            data = {
                'd0': crop_data['depth_0'] * crop_data["seg_0"],
                'd1': crop_data['depth_1'] * crop_data["seg_1"],
                'intrinsics0': crop_data['intrinsics_0'],
                'intrinsics1': crop_data['intrinsics_1'],
                'T_delta': full_T_delta.astype(np.float64),
                'T_WC_opencv': data["T_WC_opencv"],
                'T_CW_opencv': pose_inv(data["T_WC_opencv"])
            }
        else:
            data = {}

        data.update({
            'rgb0': crop_data["rgb_0"].astype(np.float32),   # (1, h, w)
            'vmap0': crop_data["proj_0"],   # (h, w)
            'rgb1': crop_data["rgb_1"].astype(np.float32),
            'vmap1': crop_data["proj_1"], # NOTE: maybe int32?
            'obj0_centre': t_w0.astype(np.float32),
            'rot_label': rot_labels,
            't_label': target_position.astype(np.float32),
            'dataset_name': 'Blender',
            'scene_id': self.idx,
            'pair_id': 0,
            'pair_names': (f"scene_{self.idx}_0",
                           f"scene_{self.idx}_1")
        })


        return data

if __name__ == "__main__":

    from Utils.se3_tools import so3_log
    import open3d as o3d
    from copy import deepcopy

    def img_to_o3d_pcd(depth: np.ndarray, intrinsic_matrix: np.ndarray, rgb=None):
        if depth.dtype == np.int32:
            depth = depth.astype(np.uint16)
        assert depth.dtype == np.uint16, f'The depth image must be \'mm\' stored in the dtype np.uint16 and not {depth.dtype}'
        if rgb is not None:
            assert rgb.dtype == np.uint8, f'The RGB image must be stored in the dtype np.uint8 and not {depth.rgb}'

        intrinsic_matrix_o3d = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_matrix_o3d.intrinsic_matrix = intrinsic_matrix
        depth_o3d = o3d.geometry.Image(depth.copy())
        if rgb is not None:
            rgb_o3d = o3d.geometry.Image(rgb.copy())
            rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(color=rgb_o3d, depth=depth_o3d,
                                                                        depth_scale=1000,
                                                                        convert_rgb_to_intensity=False)
            pcd_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_o3d, intrinsic=intrinsic_matrix_o3d,
                                                                    extrinsic=np.eye(4))
        else:
            pcd_o3d = o3d.geometry.PointCloud.create_from_depth_image(depth=depth_o3d, intrinsic=intrinsic_matrix_o3d,
                                                                    extrinsic=np.eye(4))
        return pcd_o3d
    




    dataset = BlenderDataset_4dof('class', n_bins=90)
    # print(dataset.get_rotation_label(0.5))
    # print(np.argmax(dataset.get_rotation_label(-0.5)))

    print(f"\nThe dataset length is: {len(dataset)}")

    # while True:
    #     for point in dataset:
            # print(f"Scene id: {point['scene_id']}\nObject id: {point['pair_id']}\n")
            # depth = (point['d0'] * 1000).astype(np.uint16)
            # pcd1 = img_to_o3d_pcd(depth, point['intrinsics0'])
            # pcd1.paint_uniform_color([1, 0.706, 0])
            # pcd1.transform(point['T_delta'])
            # depth = (point['d1'] * 1000).astype(np.uint16)
            # pcd2 = img_to_o3d_pcd(depth, point['intrinsics1'])
            # pcd2.paint_uniform_color([0, 0.651, 0.929])
            # o3d.visualization.draw([pcd1, pcd2])

    for point in dataset:
        # for key in point.keys():
        #     if isinstance(point[key], np.ndarray):
        #         tp = point[key].dtype
        #     else:
        #         tp = type(point[key])
        #     print(f"{key}: {tp}")
        print("\n")

    # n_samples = 4
    # labels = np.zeros((n_samples, 3))
    # for i in range(n_samples):
    #     print(i)
    #     data = dataset[i]
    #     print(data['label'])
    #     labels[i,:] = data['label'][None]
    # print(np.mean(labels, axis=0)) #[ 0.0282289   0.00679863 -0.01870937]


    # proj0s = np.zeros((1))
    # proj1s = np.zeros((1))
    # for i in range(200):
    #     print(i)
    #     data = dataset[i]
    #     seg = data['d0'] != 0
    #     # print(data['d0'].shape)
    #     print(f"1: {np.min(data['d0'][seg])}, {np.max(data['d0'][seg])}")
    #     proj0s = np.concatenate((proj0s, data['d0'][seg]))
    #     seg = data['d1'] != 0
    #     print(f"2: {np.min(data['d1'][seg])}, {np.max(data['d1'][seg])}\n")
    #     proj1s = np.concatenate((proj1s, data['d1'][seg]))
    # print("Done. The metrics are:")
    # print(f"1x: {np.mean(proj0s, axis=0)} +/- {np.std(proj0s, axis=0)}")
    # print(f"2x: {np.mean(proj1s, axis=0)} +/- {np.std(proj1s, axis=0)}")



    # proj0s = np.zeros((1,3))
    # proj1s = np.zeros((1,3))
    # for i in range(200):
    #     print(i)
    #     data = dataset[i]
    #     seg = data['vmap0'][0] != 0
    #     print(f"x: {np.min(data['vmap0'][0][seg, None])}, {np.max(data['vmap0'][0][seg, None])}")
    #     print(f"y: {np.min(data['vmap0'][1][seg, None])}, {np.max(data['vmap0'][1][seg, None])}")
    #     print(f"z: {np.min(data['vmap0'][2][seg, None])}, {np.max(data['vmap0'][2][seg, None])}\n")
    #     proj_data = np.concatenate((data['vmap0'][0][seg, None], data['vmap0'][1][seg, None], data['vmap0'][2][seg, None]), axis=1)
    #     proj0s = np.concatenate((proj0s, proj_data))
    #     seg = data['vmap1'][0] != 0
    #     proj_data = np.concatenate((data['vmap1'][0][seg, None], data['vmap1'][1][seg, None], data['vmap1'][2][seg, None]), axis=1)
    #     proj1s = np.concatenate((proj1s, proj_data))
    # print("Done. The metrics are:")
    # print(f"1x: {np.mean(proj0s[1:,0], axis=0)} +/- {np.std(proj0s[1:,0], axis=0)}")
    # print(f"1y: {np.mean(proj0s[1:,1], axis=0)} +/- {np.std(proj0s[1:,1], axis=0)}")
    # print(f"1z: {np.mean(proj0s[1:,2], axis=0)} +/- {np.std(proj0s[1:,2], axis=0)}")

    # print(f"2x: {np.mean(proj1s[1:,0], axis=0)} +/- {np.std(proj1s[1:,0], axis=0)}")
    # print(f"2y: {np.mean(proj1s[1:,1], axis=0)} +/- {np.std(proj1s[1:,1], axis=0)}")
    # print(f"2z: {np.mean(proj1s[1:,2], axis=0)} +/- {np.std(proj1s[1:,2], axis=0)}")

    # while True:
    #     for i in range(len(dataset)):
    #         dataset[i]
    #     print("\n")