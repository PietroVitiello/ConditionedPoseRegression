import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from CNN.backbone import ResNet_Backbone, ConvBlock, ResNet_Block_2

class DRes_v1_RCl_tReg(nn.Module):

    def __init__(self, n_classes: int = 18) -> None:
        super(DRes_v1_RCl_tReg, self).__init__()

        self.encoder = ResNet_Backbone()
        self.fusion = ConvBlock(512, 256, downsample=False)
        # self.fusion = ConvBlock(256, 256, downsample=False)
        self.resnet = self.resnet_end()

        self.mlp1 = nn.Sequential(nn.Linear(in_features=512, out_features=256, bias=True),
                                 nn.SELU(inplace=True))
        self.mlp2 = nn.Sequential(nn.Linear(in_features=256, out_features=128, bias=True),
                                 nn.SELU(inplace=True))
        self.rot_class_mlp = nn.Sequential(nn.Linear(in_features=128, out_features=n_classes, bias=True),
                                  nn.Identity(inplace=True))
        self.t_reg_mlp = nn.Sequential(nn.Linear(in_features=128, out_features=3, bias=True),
                                  nn.Identity(inplace=True))

    def resnet_end(self):
        return nn.Sequential(
            ResNet_Block_2(256, 512),
            ResNet_Block_2(512, 1024),
            ResNet_Block_2(1024, 2048),
            ResNet_Block_2(2048, 1024),
            ResNet_Block_2(1024, 512)
        )

    def forward(self, batch: dict) -> torch.Tensor:
        encoded_live = self.encoder(batch["rgb0"], batch["vmap0"])
        encoded_bottleneck = self.encoder(batch["rgb1"], batch["vmap1"])

        print(f"Encoded live:\n{encoded_live[:,0,int(encoded_live.shape[2]/2),:]}\n")
        print(f"Encoded bottleneck:\n{encoded_bottleneck[:,0,int(encoded_bottleneck.shape[2]/2),:]}\n")

        # out = encoded_live - encoded_bottleneck
        out = torch.concat((encoded_live, encoded_bottleneck), dim=1)
        out = self.fusion(out)
        print(f"Fusion:\n{out[:,0,int(out.shape[2]/2),:]}\n")
        out = self.resnet(out).squeeze(2).squeeze(2)

        out = self.mlp1(out)
        out = self.mlp2(out)
        rot_class = self.rot_class_mlp(out)
        t_reg = self.t_reg_mlp(out)

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

    model = DRes_v1_RCl_tReg(90)
    print("Parameter cound: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    out = model(batch)
    print(out[0].shape, out[1].shape)
    # print(out)




