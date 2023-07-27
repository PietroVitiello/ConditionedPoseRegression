import glob
import json
import os
import math
import msgpack
import msgpack_numpy as m
from copy import deepcopy

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
from Dataset.data_augmentation import DataAugmentator
from Utils.data_utils import (pose_inv, bbox_from_mask, crop, calculate_intrinsic_for_crop, calculate_rot_delta, 
                get_segmented_keypoints, project_pointcloud, calculate_intrinsic_for_new_resolution, rot2rotvec)
from Utils.training_utils import encode_rotation_matrix

# from .debug_utils import estimate_correspondences, estimate_correspondences_diff_intr

m.patch()

class BlenderDataset_4dof_pcd(Dataset):

    def __init__(self,
                 label_type: str,
                 crop_margin: float = 0.05, #0.3
                 n_bins: int = 18,
                 do_augmentation: bool = True,
                 is_validation: bool = False,
                 train_split: float = 0.98,
                 debug: bool = False) -> None:
        
        self.dataset_dir = DATASET_DIR
        self.label_type = label_type

        self.idx = None
        self.crop_margin = crop_margin
        self.bin_number = n_bins
        self.debug_mode = debug

        self.is_validation = is_validation
        self.train_split = train_split
        self.val_split = 1 - train_split
        self.do_augmentation = do_augmentation
        self.augmentator = DataAugmentator()

        self.scene_names = np.sort(glob.glob(os.path.join(self.dataset_dir, 'scene_*')))
        self.filter_correct_datapoints()

    def filter_correct_datapoints(self):
        if self.is_validation:
            length = int(len(self.scene_names) * 2 * self.val_split)
            self.scene_names = self.scene_names[-length:]
        else:
            length = int(len(self.scene_names) * 2 * self.train_split)
            self.scene_names = self.scene_names[:length]

    def __len__(self):
        if self.is_validation:
            return int(len(self.scene_names) * 2 * self.val_split)
        else:
            return int(len(self.scene_names) * 2 * self.train_split)
        # return 80

    def load_scene(self, scene_filename) -> dict:
        with open(scene_filename, "rb") as data_file:
            byte_data = data_file.read()
            data: dict = msgpack.unpackb(byte_data)

        data = deepcopy(data)
        # kernel = np.ones((5, 5), dtype=np.float32)
        # # # kernel = np.ones((3,3), dtype=np.float32) / 9
        # # # kernel = np.ones((2,2), dtype=np.float32) / 4

        # # cv2.imshow("Normal", data["cp_main_obj_segmaps"][0].astype(np.float32))

        # data["seg_0"] = cv2.filter2D(data["cp_main_obj_segmaps"][0].astype(np.float32), -1, kernel)
        # data["seg_1"] = cv2.filter2D(data["cp_main_obj_segmaps"][1].astype(np.float32), -1, kernel)
        # data["seg_0"] = (data["seg_0"] >= 1)
        # data["seg_1"] = (data["seg_1"] >= 1)

        # print(data["seg_0"].shape)
        # print(data["seg_0"].dtype)

        # cv2.imshow("Eroded", data["seg_0"].astype(np.float32))
        # cv2.waitKey(0)

        data["seg_0"] = data["cp_main_obj_segmaps"][0].astype(bool)
        data["seg_1"] = data["cp_main_obj_segmaps"][1].astype(bool)

        # plt.figure()
        # plt.imshow(data["colors"][0])

        if self.do_augmentation:
            data["colors"], data["depth"] = self.augmentator.augment_data(data["colors"], data["depth"])

        data["depth"][0] = data["depth"][0] * (data["depth"][0] < 5.0) * data["seg_0"]
        data["depth"][1] = data["depth"][1] * (data["depth"][1] < 5.0) * data["seg_1"]
        
        data.update({
            "rgb_0": data["colors"][0].astype(np.float32),
            "rgb_1": data["colors"][1].astype(np.float32),
            "depth_0": data["depth"][0].astype(np.float32),
            "depth_1": data["depth"][1].astype(np.float32),
            "intrinsics_0": data["intrinsic"],
            "intrinsics_1": data["intrinsic"],
        })

        data["T_WC_opencv"][2,3] -= 0.5
        data["T_WO_frame_0"][2,3] -= 0.5
        data["T_WO_frame_1"][2,3] -= 0.5

        data.pop("colors")
        data.pop("depth")
        data.pop("cp_main_obj_segmaps")
        data.pop("intrinsic")

        # plt.figure()
        # plt.imshow(data["rgb_0"].astype(np.int16))
        # plt.figure()
        # plt.imshow(data["rgb_1"].astype(np.int16))
        # plt.show()
        
        return data
    
    def load_random_scene(self):
        self.idx = np.random.randint(0, len(self))
        scene_dir = os.path.join(self.dataset_dir, f"scene_{str(self.idx).zfill(7)}")
        return self.load_scene(scene_dir)

    def get_filtered_depth_ids(self, depth: np.ndarray, seg: np.ndarray):
        n_smallest = 500
        std_scale = 4
        seg_flat_depth = depth.reshape(-1)[seg.reshape(-1)]
        # print(seg_flat_depth.shape)

        semi_sorted_args = np.argpartition(-seg_flat_depth[:], n_smallest) #get arg of 50 largest values
        largest_values = seg_flat_depth[semi_sorted_args[:n_smallest]]
        # print(f"\n\n\nLargest: {largest_values}")

        mean, std = np.mean(largest_values), np.std(largest_values)
        filter_args = seg_flat_depth[:] > (mean + std_scale * std)
        # print(f"\nFiltered args: {len(seg_flat_depth) - len(seg_flat_depth[~filter_args])}\n\n")

        semi_sorted_args = np.argpartition(seg_flat_depth[:], n_smallest) #get arg of 50 smallest values
        smallest_values = seg_flat_depth[semi_sorted_args[:n_smallest]]
        # print(f"\n\n\Smallest: {smallest_values}")
        
        mean, std = np.mean(smallest_values), np.std(smallest_values)
        filter_args += seg_flat_depth[:] < (mean - std_scale * std)
        # print(f"\nFiltered args: {len(seg_flat_depth) - len(seg_flat_depth[~filter_args])}\n\n")

        # keep_args = seg_flat_depth > 0.001
        # print(f"\n\nFiltered args: {len(seg_flat_depth) - len(keep_args)}\n\n")
        return ~filter_args

    
    def get_pointclouds(self, crop_data: dict):
        K = crop_data["intrinsics_0"]
        depth = crop_data["depth_0"]
        keypoints = get_segmented_keypoints(crop_data['seg_0'])
        projected_keypoints = project_pointcloud(keypoints, depth, K, 'm')
        # crop_data['pc0_keep_id'] = self.get_filtered_pointcloud_ids(projected_keypoints)
        crop_data['pc0_keep_id'] = self.get_filtered_depth_ids(crop_data["depth_0"], crop_data["seg_0"])
        projected_keypoints = np.concatenate((projected_keypoints, np.ones((projected_keypoints.shape[0], 1))), axis=1).transpose(1,0)
        crop_data['pc0'] = (crop_data["T_WC_opencv"] @ projected_keypoints).astype(np.float32)
        crop_data['pc0'] = crop_data["pc0"].transpose(1,0)[:,:3]
        # print(np.min(crop_data['pc0'][:,2]), np.max(crop_data['pc0'][:,2]))

        K = crop_data["intrinsics_1"]
        depth = crop_data["depth_1"]
        keypoints = get_segmented_keypoints(crop_data['seg_1'])
        projected_keypoints = project_pointcloud(keypoints, depth, K, 'm')
        # print(np.mean(projected_keypoints, axis=0))
        # crop_data['pc1_keep_id'] = self.get_filtered_pointcloud_ids(projected_keypoints)
        crop_data['pc1_keep_id'] = self.get_filtered_depth_ids(crop_data["depth_1"], crop_data["seg_1"])
        projected_keypoints = np.concatenate((projected_keypoints, np.ones((projected_keypoints.shape[0], 1))), axis=1).transpose(1,0)
        crop_data['pc1'] = (crop_data["T_WC_opencv"] @ projected_keypoints).astype(np.float32)
        # crop_data['pc1'] = np.expand_dims(crop_data["pc1"], axis=0).transpose(0,2,1)
        crop_data['pc1'] = crop_data["pc1"].transpose(1,0)[:,:3]
        # print(np.mean(crop_data["pc1"], axis=0))

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
        # print(np.dot(z_axis, axis))
        if np.dot(z_axis, axis) < 0:
            delta_magnitude *= -1
        target_position = data["T_WO_frame_1"][:3,3]
        t_W0 = data["T_WO_frame_0"][:3,3]
        return delta_magnitude * 180/np.pi, target_position, t_W0, T_delta_cam
    
    def get_rotation_label(self, angle, flip:float = 1):
        angle *= flip
        if self.label_type == 'class':
            angle += 45 #make sure the whole interval [-45, 45] is positive
            class_index = int(angle // (90/self.bin_number))
            class_labels = np.zeros(self.bin_number, dtype=np.float32)
            class_labels[class_index] = 1.0
            return class_labels
        elif self.label_type == 'reg':
            return (angle / 45).astype(np.float32)[...,None]

    def __getitem__(self, idx):
        # Check length of Dataset is respected
        # self.idx = np.random.randint(0,2)
        if idx >= len(self):
            raise IndexError
        # idx = np.random.randint(0,6) * 2
        self.idx = idx // 2
        flip_object_pairs = (idx % 2)*-2 + 1  #1 if even, -1 if odd
        
        # scene_dir = os.path.join(self.dataset_dir, f"scene_{str(self.idx).zfill(7)}")
        # data = self.load_scene(scene_dir)

        is_valid_scene = False
        while not is_valid_scene:
            try:
                # Get all the data for current scene
                scene_dir = self.scene_names[self.idx]
                data = self.load_scene(scene_dir)    
                self.get_pointclouds(data)
                rot_magnitude, target_position, t_W0, full_T_delta = self.get_transformation(data)
                assert rot_magnitude < 45.1, "The rotation magnitude is actually above 45 degrees"
                rot_label = self.get_rotation_label(rot_magnitude, flip_object_pairs)
                is_valid_scene = True
            except Exception as e:
                logger.warning(f"[SCENE {self.idx}] The following exception was found: \n{e}")
                # data = self.load_random_scene()
                idx = np.random.randint(0, len(self))
                self.idx = idx // 2

        data["rgb_0"] = data["rgb_0"] / 255 * 2 - 1
        data["rgb_1"] = data["rgb_1"] / 255 * 2 - 1
        # print(np.min(crop_data["rgb_0"]), np.max(crop_data["rgb_0"]))
        # crop_data["pc0"] = crop_data["pc0"] / 0.5 * 2
        # crop_data["rgb_1"] = crop_data["rgb_1"] / 255 * 2 - 1

        rgb_data = data["rgb_0"].transpose(1,2,0).reshape(-1,3)
        rgb_data = rgb_data[data["seg_0"].reshape(-1),:]
        data["pc0"] = np.concatenate((data["pc0"], rgb_data), axis=1)

        rgb_data = data["rgb_1"].transpose(1,2,0).reshape(-1,3)
        rgb_data = rgb_data[data["seg_1"].reshape(-1),:]
        data["pc1"] = np.concatenate((data["pc1"], rgb_data), axis=1)

        data["pc0"] = data["pc0"][data['pc0_keep_id'],:]
        data["pc1"] = data["pc1"][data['pc1_keep_id'],:]

        n_points = 2048
        sample_args = np.random.randint(low=0, high=data["pc0"].shape[0], size=n_points)
        data["pc0"] = data["pc0"][sample_args,:]
        sample_args = np.random.randint(low=0, high=data["pc1"].shape[0], size=n_points)
        data["pc1"] = data["pc1"][sample_args,:]

        # # print(np.min(data['pc1'][:,2]), np.max(data['pc1'][:,2]))
        # pcd_error_z = np.min(data['pc1'][:,2]) - np.min(data['pc0'][:,2])
        # # print(pcd_error_z)
        # gt_error_z = target_position[2] - t_W0[2]
        # # print(gt_error_z)
        # print(f"\n ####### Discrepancy: {(gt_error_z - pcd_error_z) < 0.01}")
        # if (gt_error_z - pcd_error_z) > 0.01:
        #     print(f"\n ####### Discrepancy: {gt_error_z - pcd_error_z}")

        # pointcloud_min_z = np.min((np.min(data["pc0"][:,2]), np.min(data["pc1"][:,2])))
        pointcloud_min_z = min(np.min(data["pc0"][:,2]), np.min(data["pc1"][:,2]))
        if pointcloud_min_z < 0.05:
            sample_new_z = np.random.uniform(-0.05, 0.05)
            displace = sample_new_z - pointcloud_min_z
            data["pc0"][:,2] += displace
            data["pc1"][:,2] += displace
            target_position[2] += displace
            t_W0[2] += displace
            # print(f"Displace: {displace}")

        data["pc0"] = np.concatenate((data["pc0"], data["pc0"][:,:3]), axis=1)
        data["pc1"] = np.concatenate((data["pc1"], data["pc1"][:,:3]), axis=1)

        # print(f"DATASET:\nObj0: {(t_W0 - np.mean(data['pc0'][:,:3], axis=0))*100} \nObj1: {(target_position - np.mean(data['pc1'][:,:3], axis=0))*100}")

        # print("Object 0")
        # print(np.mean(data["pc0"][:, :3], axis=0))
        # print(t_W0)

        # print("\nObject 1")
        # print(np.mean(data["pc1"][:, :3], axis=0))
        # print(target_position)



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
            processed_data = {
                # 'd0': data['depth_0'] * data["seg_0"],
                # 'd1': data['depth_1'] * data["seg_1"],
                # 'intrinsics0': data['intrinsics_0'],
                # 'intrinsics1': data['intrinsics_1'],
                # 'T_delta': full_T_delta.astype(np.float64),
                # 'T_WC_opencv': data["T_WC_opencv"],
                # 'T_CW_opencv': pose_inv(data["T_WC_opencv"])
                "min_z": min(np.min(data["pc0"][:,2]), np.min(data["pc1"][:,2])),
                "max_z": max(np.max(data["pc0"][:,2]), np.max(data["pc1"][:,2]))
            }
        else:
            processed_data = {}

        if idx % 2 == 0:
            processed_data.update({
                'pc0': data["pc0"].transpose(1,0).astype(np.float32),   # (1, 6, p)
                'pc1': data["pc1"].transpose(1,0).astype(np.float32),
                'obj0_centre': t_W0.astype(np.float32),
                'rot_label': rot_label,
                't_label': target_position.astype(np.float32),
                'extr': data["T_WC_opencv"],
                'dataset_name': 'Blender',
                'scene_id': self.idx,
                'pair_id': 0,
                'pair_names': (f"scene_{self.idx}_00",
                            f"scene_{self.idx}_01")
            })
        else:
            processed_data.update({
                'pc0': data["pc1"].transpose(1,0).astype(np.float32),   # (1, 6, p)
                'pc1': data["pc0"].transpose(1,0).astype(np.float32),
                'obj0_centre': target_position.astype(np.float32),
                'rot_label': rot_label,
                't_label': t_W0.astype(np.float32),
                'extr': data["T_WC_opencv"],
                'dataset_name': 'Blender',
                'scene_id': self.idx,
                'pair_id': 1,
                'pair_names': (f"scene_{self.idx}_10",
                            f"scene_{self.idx}_11")
            })


        return processed_data

if __name__ == "__main__":

    from Utils.se3_tools import so3_log
    import open3d as o3d
    from copy import deepcopy
    import time

    def get_transformation(data):
        angles_z = data["rot_label"] * (np.pi / 4)
        T = np.eye(4)
        cosines = np.cos(angles_z)
        sines = np.sin(angles_z)
        T[0,0] = cosines
        T[1,1] = cosines
        T[0,1] = -sines
        T[1,0] = sines
        t_rotated = T[:3,:3] @ data["obj0_centre"][...,None]
        T[:3,3] = data["t_label"] - t_rotated[:,0]
        return T

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
    
    def compare_o3d_filter(pc: np.ndarray):
        start = time.time()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=15.0)
        cl, ind = pcd.remove_radius_outlier(nb_points=3, radius=0.005)
        pcd_f = pcd.select_by_index(ind)
        print(f"Elapsed time: {time.time() - start}")
        pcd.paint_uniform_color([1, 0.706, 0])
        pcd_f.paint_uniform_color([0, 0.651, 0.929])
        o3d.visualization.draw([pcd, pcd_f])
    
    def get_o3d_pointcloud(pc: np.ndarray):
        # start = time.time()
        pc = pc.transpose(1,0)[:,:3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=15.0)
        # cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
        # pcd_f = pcd.select_by_index(ind)
        # pcd = np.asarray(pcd_f.points)
        # print(f"Elapsed time: {time.time() - start}")
        # print(pcd.shape)
        # pcd.paint_uniform_color([1, 0.706, 0])
        # pcd_f.paint_uniform_color([0, 0.651, 0.929])
        # o3d.visualization.draw([pcd, pcd_f])
        return pcd
    


    dataset = BlenderDataset_4dof_pcd('reg', n_bins=90, debug=True)
    # print(dataset.get_rotation_label(0.5))
    # print(np.argmax(dataset.get_rotation_label(-0.5)))

    print(f"\nThe dataset length is: {len(dataset)}")

    # # Visualise the two pointclouds
    # while True:
    #     for point in dataset:
    #         print(f"Scene id: {point['scene_id']}\nObject id: {point['pair_id']}\n")
    #         pcd1 = get_o3d_pointcloud(point["pc0"].copy())
    #         pcd1.paint_uniform_color([1, 0.706, 0])
    #         # pcd1.transform(point['T_delta'])
    #         pcd2 = get_o3d_pointcloud(point["pc1"].copy())
    #         pcd2.paint_uniform_color([0, 0.651, 0.929])
    #         o3d.visualization.draw([pcd1, pcd2])

    # # Visualise one of the two poinclouds in the filtered and unfiltered case
    # while True:
    #     for i, point in enumerate(dataset):
    #         print(f"Scene id: {point['scene_id']}\nObject id: {point['pair_id']}\n")
    #         pcd1 = compare_o3d_filter(point["pc0"].copy()[:,:3])

    # Visualise transformed pointclopud and target pointcloud
    while True:
        for point in dataset:
            print(f"Scene id: {point['scene_id']}\nObject id: {point['pair_id']}\n")
            pcd1 = get_o3d_pointcloud(point["pc0"].copy())
            pcd1.paint_uniform_color([1, 0.706, 0])
            T_delta = get_transformation(point)
            pcd1.transform(T_delta)
            pcd2 = get_o3d_pointcloud(point["pc1"].copy())
            pcd2.paint_uniform_color([0, 0.651, 0.929])
            o3d.visualization.draw([pcd1, pcd2])

    for point in dataset:
        # for key in point.keys():
        #     if isinstance(point[key], np.ndarray):
        #         tp = point[key].dtype
        #     else:
        #         tp = type(point[key])
        #     print(f"{key}: {tp}")
        print(f"\n{point['pc0'].shape}, {point['pc1'].shape}")
        # print(point['pc0'][-50:,:])

    # n_samples = int(1e4)
    # z_min = []
    # z_max = []
    # for i in range(n_samples):
    #     print(i)
    #     data = dataset[i]
    #     z_min.append(data['min_z'])
    #     z_max.append(data['max_z'])
    # print(f"Min: {np.mean(np.array(z_min))}") # -0.16425932943820953   0.01897597685456276
    # print(f"Max: {np.mean(np.array(z_max))}") #  0.16841661930084229   0.3516519367694855
    # plt.figure()
    # plt.hist(z_min)
    # plt.figure()
    # plt.hist(z_max)
    # plt.show()






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