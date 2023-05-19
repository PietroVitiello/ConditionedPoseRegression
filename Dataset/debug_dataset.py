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

class DebugDataset(Dataset):

    def __init__(self,
                 use_masks: bool = False,
                 crop_margin: float = 0.3,
                 resize_modality: int = 0,
                 segment_object: bool = False,
                 filter_data: bool = False) -> None:
        self.dataset_dir = DATASET_DIR
        self.idx = None

        self.use_masks = use_masks
        self.crop_margin = crop_margin
        self.resize_modality = resize_modality
        self.segment_object = segment_object
        self.filter_data = filter_data

    def __len__(self):
        return 50000

    def __getitem__(self, idx):
        self.idx = idx
        if self.idx >= len(self):
            raise IndexError
        
        colors = np.array([
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ], dtype=np.float32)

        sample = np.random.randint(low=0, high=3)

        rgb0 = np.ones((3,256,256), dtype=np.float32) * colors[sample][:,None,None]
        rgb1 = rgb0.copy()
        vmap0 = rgb0.copy()
        vmap1 = rgb0.copy()
        label = np.zeros(3)
        label[sample] = 1

        data = {
            'rgb0': rgb0,   # (3, h, w)
            'vmap0': vmap0,   # (h, w)
            'rgb1': rgb1,
            'vmap1': vmap1, # NOTE: maybe int32?
            'label': label,
            'dataset_name': 'Debug',
            'scene_id': self.idx,
            'pair_id': 0,
            'pair_names': (f"scene_{self.idx}_0",
                           f"scene_{self.idx}_1")
        }

        return data

if __name__ == "__main__":

    from Utils.se3_tools import so3_log
    import open3d as o3d
    from copy import deepcopy

    dataset = DebugDataset(use_masks=True, resize_modality=5)

    print(f"\nThe dataset length is: {len(dataset)}")

    # while True:
    #     for point in dataset:
    #         print(f"Scene id: {point['scene_id']}\nObject id: {point['pair_id']}\n")
    #         depth = (point['d0'] * 1000).astype(np.uint16)
    #         pcd1 = img_to_o3d_pcd(depth, point['intrinsics0'])
    #         pcd1.paint_uniform_color([1, 0.706, 0])
    #         pcd1.transform(point['T_delta'])
    #         depth = (point['d1'] * 1000).astype(np.uint16)
    #         pcd2 = img_to_o3d_pcd(depth, point['intrinsics1'])
    #         pcd2.paint_uniform_color([0, 0.651, 0.929])
    #         o3d.visualization.draw([pcd1, pcd2])

    for point in dataset:
        print("\n", point['label'])
        for key in point.keys():
            if isinstance(point[key], np.ndarray):
                tp = point[key].dtype
            else:
                tp = type(point[key])
            print(f"{key}: {tp}")

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
