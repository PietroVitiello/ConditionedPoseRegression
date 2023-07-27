import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from PointNet.pointnet2_msg_enc1 import PointNet2_Encoder_v1

import open3d as o3d
import numpy as np

class PointNetv2_tReg(nn.Module):

    def __init__(self) -> None:
        super(PointNetv2_tReg, self).__init__()

        self.encoder = PointNet2_Encoder_v1(output_dim=256, data_dim=6)
        self.aux_mean_mlp = nn.Linear(in_features=256, out_features=3, bias=True)
        self.fusion = nn.Sequential(nn.Linear(in_features=512, out_features=256, bias=True),
                                    # nn.BatchNorm1d(256),
                                    nn.InstanceNorm1d(256),
                                    nn.LeakyReLU(inplace=True))
        self.mlp2 = nn.Sequential(nn.Linear(in_features=256, out_features=128, bias=True),
                                #   nn.BatchNorm1d(128),
                                  nn.InstanceNorm1d(128),
                                  nn.LeakyReLU(inplace=True))
        self.mlp3 = nn.Sequential(nn.Linear(in_features=128, out_features=3, bias=True),
                                  nn.Identity(inplace=True))
        # self.initialise_pretrained("pnet_v1_2/version_1/checkpoints/epoch=0-step=899-training_window_loss=1.5641.ckpt")
        
    def initialise_pretrained(self, ckpt_dir):
        ckpt_dir = os.path.join(Path(__file__).parent.parent, 'logs', ckpt_dir)
        state_dict = torch.load(ckpt_dir, map_location='cpu')['state_dict']
        own_state = self.state_dict()
        # print(own_state.keys(), "\n\n")
        for name, param in state_dict.items():
            if name.split(".")[1] == "t_model":
                name = name.split(".", 2)[2]
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
        if rot_pred is not None:
            pcd0 = self.rotate_pointcloud(pcd0, rot_pred)

        # self.viz_pointcloud(batch["pc0"], batch["pc1"])
        # self.viz_pointcloud(pcd0, batch["pc1"])

        pcd0 = torch.cat((pcd0, torch.mean(pcd0[:,:3,:], dim=2)[...,None].expand(-1, -1, pcd0.shape[2])), dim=1)
        pcd1 = torch.cat((batch["pc1"], torch.mean(batch["pc1"][:,:3,:], dim=2)[...,None].expand(-1, -1, pcd0.shape[2])), dim=1)

        encoded_live = self.encoder(pcd0)
        encoded_bottleneck = self.encoder(pcd1)

        obj0_centre_pred = self.aux_mean_mlp(encoded_live)
        obj1_centre_pred = self.aux_mean_mlp(encoded_bottleneck)

        # out = encoded_live - encoded_bottleneck
        out = torch.concat((encoded_live, encoded_bottleneck), dim=1)
        out = self.fusion(out)
        out = self.mlp2(out)
        out = self.mlp3(out)

        # pcd0[0,:3,:] += out[0,:,None]
        # self.viz_pointcloud(pcd0, batch["pc1"])

        batch['t_pred'] = out

        return out, obj0_centre_pred, obj1_centre_pred


if __name__ == "__main__":
    # rand_live = torch.randint(low=-1, high=1, size=(8, 6, 1000)).to(dtype=torch.float32)
    # rand_bottle = torch.randint(low=-1, high=1, size=(8, 6, 1000)).to(dtype=torch.float32)

    rand_live = torch.rand(size=(2, 6, 1000)).to(dtype=torch.float32)
    rand_live[0,:3,0] = torch.tensor([1,0,0])
    rand_live[1,:3,1] = torch.tensor([0,0,1])
    print(rand_live[:,:,:2])

    rand_bottle = torch.rand(size=(2, 6, 1000)).to(dtype=torch.float32)

    batch = {
        "pc0": rand_live,
        "pc1": rand_bottle,
    }

    model = PointNetv2_tReg()
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    out = model(batch, torch.tensor([2, 2]))
    print(out.shape)
    # print(out)
