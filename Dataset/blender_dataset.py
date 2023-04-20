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

from .globals import DATASET_DIR
from Utils.data_utils import (pose_inv, bbox_from_mask, crop, calculate_intrinsic_for_crop, 
                get_keypoint_indices, project_pointcloud)
from Utils.training_utils import encode_rotation_matrix
from .preprocessing import resize_img_pair

# from .debug_utils import estimate_correspondences, estimate_correspondences_diff_intr

m.patch()

class BlenderDataset(Dataset):

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
        return len(np.sort(glob.glob(os.path.join(self.dataset_dir, 'scene_*'))))
        # return len(next(os.walk(self.dataset_dir))[1])

    def load_scene(self, scene_dir) -> dict:
        scene_filename = scene_dir + '.msgpack'
        with open(scene_filename, "rb") as data_file:
            byte_data = data_file.read()
            data: dict = msgpack.unpackb(byte_data)
        for k in data.keys():
            print(k)

        # print(data["colors"].shape)
        # print(data["depth"].shape)
        # print(data["colors"])
        # print(data["depth"])
        
        data.update({
            "rgb_0": data["colors"][0],
            "rgb_1": data["colors"][1],
            "depth_0": data["depth"][0] * 1000,
            "depth_1": data["depth"][1] * 1000
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
        crop_data["intrinsics_0"] = calculate_intrinsic_for_crop(
            data["intrinsic"].copy(), top=bbox0[1], left=bbox0[0]
        )
        crop_data["intrinsics_1"] = calculate_intrinsic_for_crop(
            data["intrinsic"].copy(), top=bbox1[1], left=bbox1[0]
        )

        resize_img_pair(crop_data, self.resize_modality)

        return crop_data
    
    def project_pointclouds(self, crop_data: dict):
        K = crop_data['intrinsics_0']
        depth = crop_data["depth_0"]
        keypoints = get_keypoint_indices((depth.shape[1], depth.shape[0]))
        # print(keypoints.shape)
        projected_keypoints = project_pointcloud(keypoints, depth, K, 'mm')
        crop_data['proj_0'] = projected_keypoints.reshape((depth.shape[1], depth.shape[0], 3)).transpose(2,0,1)
        # print(crop_data['proj_0'][crop_data["seg_0"]])

        K = crop_data['intrinsics_1']
        depth = crop_data["depth_1"]
        keypoints = get_keypoint_indices((depth.shape[1], depth.shape[0]))
        projected_keypoints = project_pointcloud(keypoints, depth, K, 'mm')
        crop_data['proj_1'] = projected_keypoints.reshape((depth.shape[1], depth.shape[0], 3)).transpose(2,0,1)

    def get_rotation_label(self, data):
        T_C0 = pose_inv(data["T_WC_opencv"]) @ data["T_WO_frame_0"]
        T_1C = pose_inv(data["T_WO_frame_1"]) @ data["T_WC_opencv"]
        T_delta = T_C0 @ T_1C
        return encode_rotation_matrix(T_delta[:3, :3])

    def __getitem__(self, idx):
        # Check length of Dataset is respected
        # self.idx = np.random.randint(0,2)
        self.idx = idx
        if self.idx >= len(self):
            raise IndexError
        
        # Get all the data for current scene
        scene_dir = os.path.join(self.dataset_dir, f"scene_{str(self.idx).zfill(7)}")
        data = self.load_scene(scene_dir)

        # try:
        crop_data = self.crop_object(data)        
        self.project_pointclouds(crop_data)
        R_delta = self.get_rotation_label(data)
        # except Exception as e:
        #     logger.warning(f"The following exception was found: \n{e}")
        #     data = self.load_random_scene()

        crop_data["rgb_0"] *= crop_data["seg_0"]
        crop_data["proj_0"] *= crop_data["seg_0"]
        crop_data["rgb_1"] *= crop_data["seg_1"]
        crop_data["proj_1"] *= crop_data["seg_1"]

        data = {
            'rgb0': crop_data["rgb_0"].astype(np.float32),   # (1, h, w)
            'vmap0': crop_data["proj_0"],   # (h, w)
            'rgb1': crop_data["rgb_1"].astype(np.float32),
            'vmap1': crop_data["proj_1"], # NOTE: maybe int32?
            'encoded_rot': R_delta,
            'dataset_name': 'Blender',
            'scene_id': self.idx,
            'pair_id': 0,
            'pair_names': (f"scene_{self.idx}_0",
                           f"scene_{self.idx}_1")
        }
        return data

if __name__ == "__main__":

    dataset = BlenderDataset(use_masks=True, resize_modality=5)

    print(f"\nThe dataset length is: {len(dataset)}")

    while True:
        for point in dataset:
            print(f"Scene id: {point['scene_id']}\nObject id: {point['pair_id']}\n")

    # for point in dataset:
    #     print("\n")
    #     for key in point.keys():
    #         if isinstance(point[key], np.ndarray):
    #             tp = point[key].dtype
    #         else:
    #             tp = type(point[key])
    #         print(f"{key}: {tp}")

    dataset[2]

    # print(len(dataset))

    # while True:
    #     for i in range(len(dataset)):
    #         dataset[i]
    #     print("\n")