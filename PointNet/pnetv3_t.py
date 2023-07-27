import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from PointNet.pointnet2_msg_enc1 import PointNet2_Encoder_v1

import open3d as o3d
import numpy as np

class PointNetv3_tReg(nn.Module):

    def __init__(self, only_translation: bool = False) -> None:
        super(PointNetv3_tReg, self).__init__()

        # self.encoder = PointNet2_Encoder_v1(output_dim=256, data_dim=6)
        self.encoder = PointNet2_Encoder_v1(output_dim=256, data_dim=3)
        self.res_translation_perceptron = nn.Linear(in_features=512, out_features=3, bias=True)
        self.only_translation = only_translation
        # self.initialise_pretrained("pnetv3_t_1/version_0/checkpoints/epoch=0-step=1799-training_window_loss=0.0009.ckpt")
        
    def initialise_pretrained(self, ckpt_dir):
        ckpt_dir = os.path.join(Path(__file__).parent.parent, 'logs', ckpt_dir)
        state_dict = torch.load(ckpt_dir, map_location='cpu')['state_dict']
        own_state = self.state_dict()
        for name, param in state_dict.items():
            name = name.split(".", 1)[1]
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            print(f"Loading {name} parameters")
            own_state[name].copy_(param)

    def viz_pointcloud(self, pcd0, pcd1):
        def get_o3d_pointcloud(pc: torch.Tensor):
            pc = pc.permute(0,2,1)[0,:,:3].detach().cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc)
            return pcd
        
        # T_delta = np.eye(4)
        pcd0 = get_o3d_pointcloud(pcd0)
        pcd0.paint_uniform_color([1, 0.706, 0])
        # T_delta[:3,:3] = R
        # pcd1.transform(T_delta)
        pcd1 = get_o3d_pointcloud(pcd1)
        pcd1.paint_uniform_color([0, 0.651, 0.929])
        o3d.visualization.draw([pcd0, pcd1])
        
    def rotate_pointcloud(self, pcd, rot_pred):
        angles_z = rot_pred * (torch.pi / 4)
        bs = rot_pred.shape[0]
        R = torch.eye(3, device=rot_pred.device).unsqueeze(0)
        R = R.repeat(bs, 1, 1)
        cosines = torch.cos(angles_z)
        sines = torch.sin(angles_z)
        R[:,0,0] = cosines
        R[:,1,1] = cosines
        R[:,0,1] = -sines
        R[:,1,0] = sines

        pcd[:,:3,:] = ((R[:,None,:,:] @ pcd.permute(0,2,1)[:,:,:3,None]))[:,:,:,0].permute(0,2,1)
        return pcd

    def forward(self, batch: dict, rot_pred: float = None) -> torch.Tensor:
        ''' rot_pred has to be a number between -1 and 1,
            where -1 and 1 represent -45 and +45 degrees respectively'''
        pcd0 = batch["pc0"].clone()
        if self.only_translation:
            rot_pred = torch.rand(pcd0.shape[0], device=pcd0.device) * 2 - 1
            batch['rot_pred'] = rot_pred
        if rot_pred is not None:
            pcd0 = self.rotate_pointcloud(pcd0, rot_pred)

        # self.viz_pointcloud(batch["pc0"], batch["pc1"])
        # self.viz_pointcloud(pcd0, batch["pc1"])

        # pcd0 = torch.cat((pcd0, torch.mean(pcd0[:,:3,:], dim=2)[...,None].expand(-1, -1, pcd0.shape[2])), dim=1)
        # pcd1 = torch.cat((batch["pc1"], torch.mean(batch["pc1"][:,:3,:], dim=2)[...,None].expand(-1, -1, pcd0.shape[2])), dim=1)

        # pcd0[:,-3:,:] = pcd0[:,:3,:]

        encoded_live = self.encoder(pcd0)
        encoded_bottleneck = self.encoder(batch["pc1"])

        out = torch.concat((encoded_live, encoded_bottleneck), dim=1)
        residual_translation_pred = self.res_translation_perceptron(out)

        # pcd0_ = pcd0.clone()
        # pcd0_[:,:3,:] += (torch.mean(batch["pc1"].detach()[:,:3,:], dim=2) - torch.mean(pcd0.detach()[:,:3,:], dim=2) + residual_translation_pred)[...,None].expand(-1, -1, pcd0.shape[2])
        # pcd0_[:,:3,:] += (torch.mean(batch["pc1"].detach()[:,:3,:], dim=2) - torch.mean(pcd0.detach()[:,:3,:], dim=2))[...,None].expand(-1, -1, pcd0.shape[2])

        # self.viz_pointcloud(batch["pc1"].detach(), pcd0_)

        batch["rotated_pc0_centre"] = torch.mean(pcd0[:,:3,:], dim=2)
        batch["res_translation_pred"] = residual_translation_pred
        # batch["res_translation_pred"] = residual_translation_pred * 0

        return residual_translation_pred


if __name__ == "__main__":
    # rand_live = torch.randint(low=-1, high=1, size=(8, 6, 1000)).to(dtype=torch.float32)
    # rand_bottle = torch.randint(low=-1, high=1, size=(8, 6, 1000)).to(dtype=torch.float32)

    rand_live = torch.rand(size=(2, 9, 1000)).to(dtype=torch.float32)
    rand_live[0,:3,0] = torch.tensor([1,0,0])
    rand_live[1,:3,1] = torch.tensor([0,0,1])
    print(rand_live[:,:,:2])

    rand_bottle = torch.rand(size=(2, 9, 1000)).to(dtype=torch.float32)

    batch = {
        "pc0": rand_live,
        "pc1": rand_bottle,
    }

    model = PointNetv3_tReg(True)
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    out = model(batch, torch.tensor([2, 2]))
    print(out.shape)
    # print(out)
