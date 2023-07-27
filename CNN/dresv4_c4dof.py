import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from CNN.backbone_v3 import ResNet_Encoder, ResNet_Block_2, ResNet_Block_2_later, ConvBlock

class DRes_v3_RCl_tReg(nn.Module):

    def __init__(self, n_classes: int = 18) -> None:
        super(DRes_v3_RCl_tReg, self).__init__()
        self.n_classes = n_classes

        self.encoder = ResNet_Encoder()
        self.fusion = ConvBlock(512, 256, downsample=False)
        self.resnet = self.resnet_end()

        self.mlp1 = nn.Sequential(nn.Linear(in_features=512, out_features=256, bias=True),
                                  nn.SELU(inplace=True))
        self.mlp2 = nn.Sequential(nn.Linear(in_features=256, out_features=128, bias=True),
                                  nn.SELU(inplace=True))
        self.rot_class_mlp = nn.Sequential(nn.Linear(in_features=128, out_features=n_classes, bias=True),
                                  nn.Identity(inplace=True))
        self.t_reg_mlp = nn.Sequential(nn.Linear(in_features=135, out_features=3, bias=True),
                                  nn.Identity(inplace=True))
        
        self.eval_angles = Parameter(torch.linspace(-40.5, 40.5, 10, requires_grad=False))
        
        # self.initialise_pretrained("dresv3_c1_1/version_0/checkpoints/epoch=1-val_loss=0.9276.ckpt")
        self.initialise_pretrained("dresv3_ptc4_1ce/version_0/checkpoints/epoch=0-val_loss=2.9146.ckpt")
        
    # def initialise_pretrained(self, ckpt_dir):
    #     ckpt_dir = os.path.join(Path(__file__).parent.parent, 'logs', ckpt_dir)
    #     state_dict = torch.load(ckpt_dir, map_location='cpu')['state_dict']
    #     own_state = self.state_dict()
    #     for name, param in state_dict.items():
    #         # if name not in own_state:
    #         #      continue
    #         valid_component = True
    #         for layer_name in ["mlp1", "mlp2", "mlp3"]:
    #             if layer_name in name:
    #                 valid_component = False

    #         if valid_component:
    #             if isinstance(param, Parameter):
    #                 # backwards compatibility for serialized parameters
    #                 param = param.data
    #             print(f"Loading {name} parameters")
    #             name = name[6:] #remove 'model.'
    #             own_state[name].copy_(param)

    def get_pointcloud_centres(self, T_wc, vmap1, vmap2):
        def get_pointcloud_centre(vmap):
            centre = torch.mean()

    
    def initialise_pretrained(self, ckpt_dir):
        ckpt_dir = os.path.join(Path(__file__).parent.parent, 'logs', ckpt_dir)
        state_dict = torch.load(ckpt_dir, map_location='cpu')['state_dict']
        own_state = self.state_dict()
        for name, param in state_dict.items():
            # if name not in own_state:
            #      continue
            valid_component = True
            for layer_name in ["mlp3"]:
                if layer_name in name:
                    valid_component = False

            if valid_component:
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                print(f"Loading {name} parameters")
                name = name[6:] #remove 'model.'
                own_state[name].copy_(param)

    def resnet_end(self):
        return nn.Sequential(
            ResNet_Block_2(256, 512),
            ResNet_Block_2(512, 512),
            ResNet_Block_2(512, 1024),
            ResNet_Block_2(1024, 1024),
            ResNet_Block_2_later(1024, 512)
        )

    def forward(self, batch: dict) -> torch.Tensor:
        encoded_live = self.encoder(batch["rgb0"], batch["vmap0"])
        encoded_bottleneck = self.encoder(batch["rgb1"], batch["vmap1"])

        # print(f"Encoded live:\n{encoded_live[:,0,int(encoded_live.shape[2]/2),:]}\n")
        # print(f"Encoded bottleneck:\n{encoded_bottleneck[:,0,int(encoded_bottleneck.shape[2]/2),:]}\n")

        # out = encoded_live - encoded_bottleneck
        out = torch.concat((encoded_live, encoded_bottleneck), dim=1)
        out = self.fusion(out)
        # print(f"Fusion:\n{out[:,0,int(out.shape[2]/2),:]}\n")
        out = self.resnet(out).squeeze(2).squeeze(2)
        # print(f"ResNet:\n{out}\n")

        out = self.mlp1(out)
        out = self.mlp2(out)
        rot_class = self.rot_class_mlp(out)

        pc1_c, pc2_c = self.get_pointcloud_centres(batch['extr'], batch["vmap0"], batch["vmap1"])
        pred_rot = self.get_pred_rot(rot_class)
        rot_out = torch.concat((pred_rot, out), dim=1)
        t_reg = self.t_reg_mlp(rot_out)

        batch['rot_pred'] = rot_class
        batch['t_pred'] = t_reg
        return rot_class, t_reg


if __name__ == "__main__":
    rand_live = torch.randint(low=0, high=255, size=(8, 3, 256, 256)).to(dtype=torch.float32)
    rand_bottle = torch.randint(low=0, high=255, size=(8, 3, 256, 256)).to(dtype=torch.float32)

    batch = {
        "rgb0": rand_live,
        "vmap0": rand_live,
        "rgb1": rand_bottle,
        "vmap1": rand_bottle,
    }

    model = DRes_v3_RCl_tReg()
    print("Parameter cound: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    out = model(batch)
    print(out[0].shape, out[1].shape)
    # print(out)
