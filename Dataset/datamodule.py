import os
import math
from collections import abc
from loguru import logger
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from os import path as osp
from pathlib import Path
from joblib import Parallel, delayed

import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    DistributedSampler,
    RandomSampler,
    dataloader,
    random_split
)

from Dataset.realworld_data_4dof_pcd import RealWorldDataset_4dof_pcd

from Dataset.debug_dataset import DebugDataset
from Dataset.blender_dataset import BlenderDataset
from Dataset.blender_dataset_class import BlenderDatasetClassification
from Dataset.blender_dataset_1dof import BlenderDataset_1dof
from Dataset.blender_dataset_1dof_reg import BlenderDataset_1dof_reg
from Dataset.bd_1dof_clNreg import BlenderDataset_1dof_classNreg
from Dataset.blender_dataset_4dof import BlenderDataset_4dof
from Dataset.bdataset_4dof_pc import BlenderDataset_4dof_pcd


class BlenderDataModule(pl.LightningDataModule):
    """ 
    For distributed training, each training process is assgined
    only a part of the training scenes to reduce memory overhead.
    """
    def __init__(self, args,
                 train_split: float = 0.98,
                 modality: int = 0):
        super().__init__()

        # 2. dataset config
        # general options
        # self.min_overlap_score_test = config.DATASET.MIN_OVERLAP_SCORE_TEST  # 0.4, omit data with overlap_score < min_overlap_score
        # self.min_overlap_score_train = config.DATASET.MIN_OVERLAP_SCORE_TRAIN
        # self.augment_fn = build_augmentor(config.DATASET.AUGMENTATION_TYPE)  # None, options: [None, 'dark', 'mobile']

        # 3.loader parameters
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }
        self.val_loader_params = {
            'batch_size': args.batch_size_val,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }
        
        # (optional) RandomSampler for debugging

        # misc configurations
        # self.parallel_load_data = getattr(args, 'parallel_load_data', False)
        self.seed = 66  # 66

        self.train_split = train_split
        self.modality = modality
        self.args = args
        self.training_dataset, self.validation_dataset = None, None
        self.real_dataset = self.initialise_rw_dataset()

    def initialise_dataset(self, is_validation: bool = False):
        if self.modality == 0:
            return BlenderDataset(self.args.use_masks, self.args.crop_margin, self.args.resize_modality,
                                  self.args.segment_object, self.args.filter_dataset)
        elif self.modality == 1:
            return BlenderDatasetClassification(self.args.crop_margin, self.args.class_bins)
            # return DebugDataset(args.use_masks, args.crop_margin, args.resize_modality,
            #                     args.segment_object, args.filter_dataset)
        elif self.modality == 2:
            return BlenderDataset_1dof(self.args.crop_margin, self.args.class_bins)
        elif self.modality == 3:
            return BlenderDataset_1dof_reg(self.args.crop_margin, self.args.class_bins)
        elif self.modality == 4:
            return BlenderDataset_1dof_classNreg(self.args.crop_margin, self.args.class_bins)
        elif self.modality == 5:
            return BlenderDataset_4dof('class', self.args.crop_margin, self.args.class_bins)
        elif self.modality in [6, 11]:
            return BlenderDataset_4dof_pcd('class', self.args.crop_margin, self.args.class_bins, train_split=self.train_split,
                                            do_augmentation=~is_validation, is_validation=is_validation)
        elif self.modality in [7, 8, 9, 10, 12]:
            return BlenderDataset_4dof_pcd('reg', self.args.crop_margin, self.args.class_bins, train_split=self.train_split,
                                            do_augmentation=~is_validation, is_validation=is_validation)
        
    def initialise_rw_dataset(self):
        if self.modality in [6, 11]:
            return RealWorldDataset_4dof_pcd('class', self.args.crop_margin, self.args.class_bins)
        elif self.modality in [7, 8, 9, 10, 12]:
            return RealWorldDataset_4dof_pcd('reg', self.args.crop_margin, self.args.class_bins)
        return None

    def setup(self, stage=None):
        """
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """

        assert stage in ['fit', 'test'], "stage must be either fit or test"

        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
        except AssertionError as ae:
            self.world_size = 1
            self.rank = 0
            logger.warning(str(ae) + " (set wolrd_size=1 and rank=0)")

        self.training_dataset = self.initialise_dataset()
        self.validation_dataset = self.initialise_dataset(is_validation=True)
        logger.info(f'[rank:{self.rank}] Train & Val Dataset loaded!')

        self.real_dataset = self.initialise_rw_dataset()
        if self.real_dataset is not None:
            logger.info(f'[rank:{self.rank}] Real World Dataset loaded!')     

    def train_dataloader(self):
        """ Build training dataloader for ScanNet / MegaDepth. """
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Train Sampler and DataLoader re-init (should not re-init between epochs!).')
        return DataLoader(self.training_dataset, **self.train_loader_params)
    
    def val_dataloader(self):
        """ Build validation dataloader for ScanNet / MegaDepth. """
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Val Sampler and DataLoader re-init.')
        if self.real_dataset is None:
            return DataLoader(self.validation_dataset, **self.val_loader_params)
        else:
            return [DataLoader(self.validation_dataset, **self.val_loader_params),
                    DataLoader(self.real_dataset, **self.val_loader_params)]

# class BlenderDataModule(pl.LightningDataModule):
#     """ 
#     For distributed training, each training process is assgined
#     only a part of the training scenes to reduce memory overhead.
#     """
#     def __init__(self, args,
#                  train_split: float = 0.98,
#                  modality: int = 0):
#         super().__init__()

#         # 2. dataset config
#         # general options
#         # self.min_overlap_score_test = config.DATASET.MIN_OVERLAP_SCORE_TEST  # 0.4, omit data with overlap_score < min_overlap_score
#         # self.min_overlap_score_train = config.DATASET.MIN_OVERLAP_SCORE_TRAIN
#         # self.augment_fn = build_augmentor(config.DATASET.AUGMENTATION_TYPE)  # None, options: [None, 'dark', 'mobile']

#         # 3.loader parameters
#         self.train_loader_params = {
#             'batch_size': args.batch_size,
#             'shuffle': True,
#             'num_workers': args.num_workers,
#             'pin_memory': getattr(args, 'pin_memory', True)
#         }
#         self.val_loader_params = {
#             'batch_size': args.batch_size_val,
#             'shuffle': False,
#             'num_workers': args.num_workers,
#             'pin_memory': getattr(args, 'pin_memory', True)
#         }
        
#         # (optional) RandomSampler for debugging

#         # misc configurations
#         # self.parallel_load_data = getattr(args, 'parallel_load_data', False)
#         self.seed = 66  # 66

#         self.train_split = train_split
#         self.full_dataset = self.initialise_dataset(modality, args)
#         self.training_dataset, self.validation_dataset = None, None

#     def initialise_dataset(self, modality, args):
#         if modality == 0:
#             return BlenderDataset(args.use_masks, args.crop_margin, args.resize_modality,
#                                   args.segment_object, args.filter_dataset)
#         elif modality == 1:
#             return BlenderDatasetClassification(args.crop_margin, args.class_bins)
#             # return DebugDataset(args.use_masks, args.crop_margin, args.resize_modality,
#             #                     args.segment_object, args.filter_dataset)
#         elif modality == 2:
#             return BlenderDataset_1dof(args.crop_margin, args.class_bins)
#         elif modality == 3:
#             return BlenderDataset_1dof_reg(args.crop_margin, args.class_bins)
#         elif modality == 4:
#             return BlenderDataset_1dof_classNreg(args.crop_margin, args.class_bins)
#         elif modality == 5:
#             return BlenderDataset_4dof('class', args.crop_margin, args.class_bins)
#         elif modality == 6:
#             return BlenderDataset_4dof_pcd('class', args.crop_margin, args.class_bins)
#         elif modality in [7, 8, 9, 10]:
#             return BlenderDataset_4dof_pcd('reg', args.crop_margin, args.class_bins)

#     def setup(self, stage=None):
#         """
#         Setup train / val / test dataset. This method will be called by PL automatically.
#         Args:
#             stage (str): 'fit' in training phase, and 'test' in testing phase.
#         """

#         assert stage in ['fit', 'test'], "stage must be either fit or test"

#         try:
#             self.world_size = dist.get_world_size()
#             self.rank = dist.get_rank()
#             logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
#         except AssertionError as ae:
#             self.world_size = 1
#             self.rank = 0
#             logger.warning(str(ae) + " (set wolrd_size=1 and rank=0)")

#         train_samples = round(self.train_split * len(self.full_dataset))
#         val_samples = len(self.full_dataset) - train_samples
#         self.training_dataset, self.validation_dataset = random_split(
#             self.full_dataset, [train_samples, val_samples]
#         )
#         logger.info(f'[rank:{self.rank}] Train & Val Dataset loaded!')

#     def train_dataloader(self):
#         """ Build training dataloader for ScanNet / MegaDepth. """
#         logger.info(f'[rank:{self.rank}/{self.world_size}]: Train Sampler and DataLoader re-init (should not re-init between epochs!).')
#         return DataLoader(self.training_dataset, **self.train_loader_params)
    
#     def val_dataloader(self):
#         """ Build validation dataloader for ScanNet / MegaDepth. """
#         logger.info(f'[rank:{self.rank}/{self.world_size}]: Val Sampler and DataLoader re-init.')
#         return DataLoader(self.validation_dataset, **self.val_loader_params)
