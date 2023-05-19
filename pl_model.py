from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path
import wandb

import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from losses import rotation_loss, rotvec_loss, cross_entropy_loss, MSE_loss, CE_MSE_loss, KL_Divergence
from optimizers import build_optimizer, build_scheduler
from profiler import PassThroughProfiler
from Utils.training_utils import calculate_rot_error_from_matrix, calculate_rot_error_from_class, calculate_rot_error_from_classreg

from CNN.cnn_cv1 import SimpleCNN
from Transformer.resT_v1 import ResNet_Transformer
from Transformer.deep_resT_v1 import Deep_ResNet_Transformer
from Transformer.resT_class_v1 import ResNet_Transformer_Classification
from Transformer.pt_resT_cv1 import PreTrained_RNTransformer_Class
from Transformer.resT_class_v2 import ResNet_Transformer_Classification as ResNet_Transformer_Cv2
from CNN.resnet_class_v1 import ResNet_Classification
from CNN.dense_resnet_cv1 import DenseResNet_Classification
from CNN.dresv1_pretrain import PreTrained_DRes_Classification
from CNN.dres_class_reg_v1 import DRes_Class_N_Reg
from CNN.dres_vertex_cv1 import DenseResNet_VertexMap_Classification
from CNN.dres_colvertex_cv1 import DRNet_ColorVertexMap_Class
from CNN.full_resnet_v1 import ResNet


class PL_Model(pl.LightningModule):
    def __init__(self,
                 args,
                 profiler=None,
                 dump_dir=None,):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        self.profiler = profiler or PassThroughProfiler()

        # Matcher: LoFTR
        self.model = self.model_choice(args)
        self.class_bins = args.class_bins
        self.loss = self.loss_choice(args.modality)
        self.optimizer = None # Will be set later in a lightning function

        self.lr = args.learning_rate
        self.optimizer_name = args.optimizer_name
        self.scheduler_name = args.scheduler_name
        self.do_warmup = ~ args.no_warmup
        self.modality = args.modality

        # Pretrained weights
        print(args.ckpt_path)
        if args.ckpt_path:
            print('load')
            state_dict = torch.load(args.ckpt_path, map_location='cpu')['state_dict']
            msg=self.model.load_state_dict(state_dict, strict=False)
            print(msg)
            logger.info(f"Load \'{args.ckpt_path}\' as pretrained checkpoint")
        
        # Testing
        self.dump_dir = dump_dir

        run_name = args.model_name if args.run_name is None else args.run_name
        self.use_wandb = args.use_wandb
        if self.global_rank == 0 and args.use_wandb:
            entity = "head-dome"
            print(f'Connecting with {entity} on wandb')
            wandb.init(
                project="headcam-dome",
                name=run_name,
                entity=entity,
                reinit=True,
                tags=["DirectRegression", args.model_name]
            )
            wandb.config = {
                "learning_rate": args.learning_rate,
                "epochs": args.max_epochs,
                "batch_size": args.batch_size
            }
            wandb.watch(self.model, log='all', log_freq=1)

        self.train_step = 0
        self.val_step = 0
        self.train_loss = np.array([np.inf])
        self.train_ori_err = np.array([np.inf])

    def wandb_log_epochs(self, data: dict, stage: str='train'):
        if stage == 'train':
            wandb.log({"epoch": self.current_epoch,
                       "train_step": self.train_step,
                       "train_loss_stepwise": self.train_loss[-1],
                       "training_window_loss": np.mean(self.train_loss[1:]),
                       "train_ori_error": self.train_ori_err[-1],
                       "trainW_ori_error": np.mean(self.train_ori_err[1:]),
                       "trainW_ori_std": np.std(self.train_ori_err[1:]),
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

    def loss_choice(self, modality):
        if modality == 0:
            return rotation_loss()
        elif modality == 1:
            return cross_entropy_loss()
        elif modality == 2:
            # return cross_entropy_loss()
            return KL_Divergence(n_steps=80000, bacthes_per_step=100, initial_steps=5)
        elif modality == 3:
            return MSE_loss()
        elif modality == 4:
            return CE_MSE_loss()

    def model_choice(self, args):
        if args.model_name == "simple_cnn":
            return SimpleCNN(args.class_bins)
        elif args.model_name == "resnet":
            if args.modality == 0:
                return ResNet()
            elif args.modality in [1, 2]:
                return ResNet_Classification(args.class_bins)
        elif args.model_name == "dense_res":
            return DenseResNet_Classification(args.class_bins)
        elif args.model_name == "pretr_dres":
            if args.modality == 2:
                return PreTrained_DRes_Classification(args.class_bins)
            elif args.modality == 3:
                return PreTrained_DRes_Classification(n_classes=1)
            elif args.modality == 4:
                return DRes_Class_N_Reg(args.class_bins)
        elif args.model_name == "dense_res_vmap":
            return DenseResNet_VertexMap_Classification(args.class_bins)
        elif args.model_name == "dense_res_cvmap":
            return DRNet_ColorVertexMap_Class(args.class_bins)
        elif args.model_name == "transformer":
            if args.modality == 0:
                return ResNet_Transformer()
            elif args.modality in [1, 2]:
                return ResNet_Transformer_Classification(args.class_bins)
        elif args.model_name == "pretr_transf":
            if args.modality == 2:
                return PreTrained_RNTransformer_Class(args.class_bins)
        elif args.model_name == "transformer2":
            return ResNet_Transformer_Cv2(args.class_bins)
        elif args.model_name == "deep_transf":
            return Deep_ResNet_Transformer()
        else:
            raise Exception("The chosen model is not supported")
        
    def calculate_rot_error(self, batch):
        if self.modality == 0:
            return calculate_rot_error_from_matrix(batch)
        elif self.modality == 1:
            return calculate_rot_error_from_class(batch, 45/self.class_bins)
        elif self.modality == 2:
            return calculate_rot_error_from_class(batch, 90/self.class_bins)
        elif self.modality in [3, 4]:
            return calculate_rot_error_from_classreg(batch)
        
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
            warmup_step = 200 #100 #500
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

        # print(f"\nBatch outputs:{batch['pred']}")
        # print(f"Batch outputs:{batch['label']}")
        outputs_batch = torch.argmax(batch['pred'].detach(), dim=1)
        labels_batch = torch.argmax(batch['label'].detach(), dim=1)
        print(f"\nBatch outputs: {outputs_batch}")
        print(f"Batch labels: {labels_batch}")
        print(f"Difference: {torch.abs(outputs_batch - labels_batch)}")
        print(f"\nBatch loss: {batch['loss']}")
    
    def on_train_epoch_start(self) -> None:
        self.train_loss = np.zeros(1)
        self.train_ori_err = np.zeros(1)
   
    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        self.calculate_rot_error(batch)

        self.train_loss = np.append(self.train_loss, batch['loss'].detach().cpu().numpy())
        self.train_ori_err = np.append(self.train_ori_err, batch['ori_error'])
        
        # logging
        if self.global_step % self.trainer.log_every_n_steps == 0:

            window_average_loss = np.mean(self.train_loss[1:])
            self.log("training_window_loss", window_average_loss)

            if self.trainer.global_rank == 0:
                if self.use_wandb:
                    self.wandb_log_epochs(data=dict())

            self.train_loss = self.train_loss[0]
            self.train_ori_err = self.train_ori_err[0]
                
        return batch['loss']
    
    def on_train_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        self.calculate_rot_error(batch)
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
                    "val_ori_error": np.mean(ori_errors),
                    "val_ori_std": np.std(ori_errors)}

        self.log("val_loss", val_data["val_rotation_loss"])
        self.log("val_ori_error", val_data["val_ori_error"])
        if self.trainer.global_rank == 0 and self.use_wandb:
            self.wandb_log_epochs(val_data, stage='val')

    def test_step(self, batch, batch_idx):
        return 0

    def test_epoch_end(self, outputs):
        return 0
