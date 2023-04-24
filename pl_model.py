from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path
import wandb

import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from losses import rotation_loss
from optimizers import build_optimizer, build_scheduler
from profiler import PassThroughProfiler
from Utils.training_utils import calculate_rot_error

from Transformer.resT_v1 import ResNet_Transformer


class PL_Model(pl.LightningModule):
    def __init__(self,
                 model_name,
                 lr: float,
                 epochs: int,
                 bs: int,
                 optimizer_name: str,
                 scheduler_name: str,
                 do_warmup: bool = False,
                 run_name = None,
                 pretrained_ckpt=None,
                 profiler=None,
                 dump_dir=None,
                 use_wandb: bool = True):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        self.profiler = profiler or PassThroughProfiler()

        # Matcher: LoFTR
        self.model = self.model_choice(model_name)
        self.loss = rotation_loss()
        self.optimizer = None # Will be set later in a lightning function

        self.lr = lr
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.do_warmup = do_warmup

        # Pretrained weights
        print(pretrained_ckpt)
        if pretrained_ckpt:
            print('load')
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            msg=self.model.load_state_dict(state_dict, strict=False)
            print(msg)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
        
        # Testing
        self.dump_dir = dump_dir

        run_name = model_name if run_name is None else run_name
        self.use_wandb = use_wandb
        if self.global_rank == 0 and use_wandb:
            entity = "head-dome"
            print(f'Connecting with {entity} on wandb')
            wandb.init(
                project="headcam-dome",
                name=run_name,
                entity=entity,
                reinit=True,
                tags=["DirectRegression", model_name]
            )
            wandb.config = {
                "learning_rate": lr,
                "epochs": epochs,
                "batch_size": bs
            }
            # wandb.watch(self.model, log='all', log_freq=1)

        self.train_step = 0
        self.val_step = 0
        self.train_loss = np.array([np.inf]) 

    def wandb_log_epochs(self, data: dict, stage: str='train'):
        if stage == 'train':
            wandb.log({"epoch": self.current_epoch,
                       "train_step": self.train_step,
                       "train_loss_stepwise": self.train_loss[-1],
                       "training_window_loss": np.mean(self.train_loss[1:]),
                       "lr": self.optimizer.param_groups[0]["lr"],
                    })
            self.train_step += 1
            
        elif stage == 'val':
            try:
                train_loss = np.mean(self.train_loss[1:])
            except IndexError:
                train_loss = self.train_loss
            data.update({
                "epoch": self.current_epoch,
                "val_step" : self.val_step,
                "train_rotation_loss": train_loss,
            })
            wandb.log(data)
            self.val_step += 1
        else:
            print("[WANDB] Incorrect logging stage")
        data.clear()

    def model_choice(self, model_name):
        if model_name == "resnet":
            return 0
        elif model_name == "transformer":
            return ResNet_Transformer()
        else:
            raise Exception("The chosen model is not supported")
        
    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.optimizer_name, self.lr)
        scheduler = build_scheduler(self.scheduler_name, optimizer)
        self.optimizer = optimizer
        return [optimizer], [scheduler]
    
    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        if self.do_warmup:
            warmup_step = 500
            if self.trainer.global_step < warmup_step:
                base_lr = 0 * self.lr #self.config.TRAINER.WARMUP_RATIO * self.lr
                lr = base_lr + \
                    (self.trainer.global_step / warmup_step) * \
                    abs(self.lr - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        # update lr stepwise
        # scheduler = self.lr_schedulers()
        # scheduler.step()
    
    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute pose estimation"):
            self.model(batch)
        
        with self.profiler.profile("Compute losses"):
            self.loss(batch)
    
    def on_train_epoch_start(self) -> None:
        self.train_loss = np.zeros(1)
   
    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        self.train_loss = np.append(self.train_loss, batch['loss'].detach().cpu().numpy())
        
        # logging
        if self.global_step % self.trainer.log_every_n_steps == 0:

            window_average_loss = np.mean(self.train_loss[1:])
            self.log("training_window_loss", window_average_loss)
            self.log("training_last_loss", self.train_loss[-1])

            if self.trainer.global_rank == 0:
                if self.use_wandb:
                    self.wandb_log_epochs(dict())

            self.train_loss = self.train_loss[0]
                
        return batch['loss']
    
    def on_train_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        calculate_rot_error(batch)
        return [batch['loss'].detach().cpu(),
                batch['ori_error']]
        
    def validation_epoch_end(self, outputs):
        # handle multiple validation sets
        # print('outputs', outputs)
        multi_outputs = np.array([np.array(output) for output in outputs])
        # print(multi_outputs)
        losses = multi_outputs[:,0]
        ori_errors = multi_outputs[:,1]

        val_data = {"val_rotation_loss": np.mean(losses),
                    "val_ori_errors": np.mean(ori_errors)}

        self.log("val_loss", val_data["val_rotation_loss"])                            
        if self.trainer.global_rank == 0 and self.use_wandb:
            self.wandb_log_epochs(val_data, stage='val')

    def test_step(self, batch, batch_idx):
        return 0

    def test_epoch_end(self, outputs):
        return 0
