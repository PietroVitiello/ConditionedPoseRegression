import math
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

from Utils.misc import get_rank_zero_only_logger, setup_gpus
from profiler import build_profiler
from pl_model import PL_Model
from Dataset.datamodule import BlenderDataModule

loguru_logger = get_rank_zero_only_logger(loguru_logger)

def parse_args():
    def str2bool(v: str) -> bool:
        return v.lower() in ("true", "1")
    # init a custom parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-model', '--model_name', type=str, default="transformer", required=False,
        help='Name of the model architecture we are training')
    parser.add_argument(
        '-name', '--run_name', type=str, default=None, required=True,
        help='Name of the run')
    parser.add_argument(
        '-mode', '--modality', type=int, default=2, required=False,
        help='The training modality')
    parser.add_argument(
        '-resize_m', '--resize_modality', type=int, default=5, required=False,
        help='Set the modality used to resize the images. Options: [0-5]')
    parser.add_argument(
        '-margin', '--crop_margin', type=float, default=0.05, required=False,
        help='The margin to put around the cropped objects expressed in fraction [0-1]')
    parser.add_argument(
        '-split', '--train_split', type=float, default=0.98, required=False,
        help='The training split in the dataset')
    parser.add_argument(
        '-bins', '--class_bins', type=int, default=18, required=False,
        help='The number of bins to use in the classification task')
    parser.add_argument(
        '-wandb', '--use_wandb', action='store_true',
        help='Whether to upload the training information to weights and biases')
    parser.add_argument(
        '-bs', '--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument(
        '-bs_val', '--batch_size_val', type=int, default=None, help='validation set batch_size per gpu')
    parser.add_argument(
        '-nw', '--num_workers', type=int, default=0)
    parser.add_argument(
        '-lr', '--learning_rate', type=float, default=None, required=False,
        help='The starting learning rate')
    parser.add_argument(
        '-optim', '--optimizer_name', type=str, default='adamw', required=False,
        help='Name of the chosen optimizer')
    parser.add_argument(
        '-sched', '--scheduler_name', type=str, default='plateau', required=False,
        help='Name of the chosen scheduler')
    parser.add_argument(
        '-no_warm', '--no_warmup', action='store_true',
        help='Whether to have learning rate warm up or not')
    parser.add_argument(
        '--pin_memory', type=lambda x: bool(strtobool(x)),
        nargs='?', default=False, help='whether loading data to pinned memory or not')
    parser.add_argument(
        '--exp_name', type=str, default='trying_out', help='Name of the experiment and of checkpoint folder')
    
    # parser.add_argument(
    #     '--data_cfg_path', type=str, help='data config path', default="configs/data/scannet_trainval.py")
    # parser.add_argument(
    #     '--main_cfg_path', type=str, help='main config path', default="configs/aspan/indoor/aspan_train.py")
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only ASpanFormer')
    
    parser.add_argument(
        '--disable_ckpt', action='store_true',
        help='disable checkpoint saving (useful for debugging).')
    parser.add_argument(
        '--profiler_name', type=str, default=None,
        help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--parallel_load_data', action='store_true',
        help='load datasets in with multiple processes.')    
    parser.add_argument(
        '--mode', type=str, default='vanilla',
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only ASpanFormer')
    parser.add_argument(
        '--ini', type=str2bool, default=False,
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only ASpanFormer')
    parser.add_argument(
        '-mask', '--use_masks', action='store_true',
        help='Whether to upload the training information to weights and biases')
    parser.add_argument(
        '-seg', '--segment_object', action='store_true',
        help='Whether to fully segment the objects in the input images (True) or leave the background (False)')
    parser.add_argument(
        '-filter', '--filter_dataset', action='store_true',
        help='Whether to filter the dataset by removing more complex datapoints')

    parser = pl.Trainer.add_argparse_args(parser)
    '''
    Useful Trainer arguments:
        - log_every_n_steps
        - max_epochs
        - gpus
    '''
    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()
    # rank_zero_only(pprint.pprint)(vars(args))

    pl.seed_everything(66)  # reproducibility
    # TODO: Use different seeds for each dataloader workers
    # This is needed for data augmentation

    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    if args.batch_size_val is None:
        args.batch_size_val = args.batch_size

    trainer_world_size = 1
    trainer_gradient_clipping = 0.8

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_Model(
        args,
        profiler=profiler,
    )
    loguru_logger.info(f"ASpanFormer LightningModule initialized!")

    # lightning data
    # data_module = MultiSceneDataModule(args, config)
    data_module = BlenderDataModule(args, train_split=args.train_split, modality=args.modality)
    loguru_logger.info(f"ASpanFormer DataModule initialized!")

    # TensorBoard Logger
    logger = TensorBoardLogger(
        save_dir='logs', name=args.run_name, default_hp_metric=False)
    ckpt_dir = Path(logger.log_dir) / 'checkpoints'

    # Callbacks
    # TODO: update ModelCheckpoint to monitor multiple metrics
    training_validation_interval = 3000
    train_w_loss_callback = ModelCheckpoint(monitor='training_window_loss', verbose=True, save_top_k=3, mode='min',
                                            save_last=False,
                                            every_n_train_steps=150,
                                            dirpath=str(ckpt_dir),
                                            filename='{epoch}-{step}-{training_window_loss:.4f}')
    # train_last_loss_callback = ModelCheckpoint(monitor='training_last_loss', verbose=True, save_top_k=3, mode='min',
    #                                            save_last=False,
    #                                            every_n_train_steps=150,
    #                                            dirpath=str(ckpt_dir),
    #                                            filename='{epoch}-{step}-{training_last_loss:.4f}')
    val_loss_callback = ModelCheckpoint(monitor='val_loss', verbose=True, save_top_k=3, mode='min',
                                       save_last=True,
                                       every_n_train_steps=training_validation_interval,
                                       dirpath=str(ckpt_dir),
                                       filename='{epoch}-{val_loss:.4f}')
    val_ori_callback = ModelCheckpoint(monitor='val_ori_error', verbose=True, save_top_k=3, mode='min',
                                       save_last=True,
                                       every_n_train_steps=training_validation_interval,
                                       dirpath=str(ckpt_dir),
                                       filename='{epoch}-{val_ori_error:.4f}')

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        # training
        callbacks.append(train_w_loss_callback)
        # validation
        callbacks.append(val_loss_callback)
        callbacks.append(val_ori_callback)
        # callbacks.append(val_auc_callback)

    # Lightning Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        plugins=DDPPlugin(find_unused_parameters=False,
                          num_nodes=args.num_nodes,
                          sync_batchnorm=trainer_world_size > 0),
        gradient_clip_val=trainer_gradient_clipping,
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=trainer_world_size > 0,
        replace_sampler_ddp=False,  # use custom sampler
        reload_dataloaders_every_epoch=False,  # avoid repeated samples!
        weights_summary='full',
        profiler=profiler,
        val_check_interval=training_validation_interval
        )
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()

