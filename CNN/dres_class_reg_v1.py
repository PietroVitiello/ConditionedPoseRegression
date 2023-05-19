import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from CNN.backbone import ResNet_Backbone, ConvBlock, ResNet_Block_2

class DRes_Class_N_Reg(nn.Module):

    def __init__(self, n_classes: int = 18) -> None:
        super(DRes_Class_N_Reg, self).__init__()

        self.encoder = ResNet_Backbone()
        self.fusion = ConvBlock(512, 256, downsample=False)
        # self.fusion = ConvBlock(256, 256, downsample=False)
        self.resnet = self.resnet_end()

        self.mlp1 = nn.Sequential(nn.Linear(in_features=512, out_features=256, bias=True),
                                  nn.LeakyReLU(inplace=True))
        self.mlp2 = nn.Sequential(nn.Linear(in_features=256, out_features=128, bias=True),
                                  nn.LeakyReLU(inplace=True))
        self.mlp3 = nn.Sequential(nn.Linear(in_features=128, out_features=n_classes, bias=True),
                                  nn.Identity(inplace=True))
        self.class_scaling = nn.Parameter(torch.linspace(-1,1,90, requires_grad=False).unsqueeze(0))
        self.mlp4 = nn.Sequential(nn.LeakyReLU(inplace=True),
                                  nn.Linear(in_features=n_classes, out_features=1, bias=True),
                                  nn.Identity(inplace=True))
        
        # self.initialise_pretrained("dense_res_4/version_0/checkpoints/best_val.ckpt")
        self.initialise_pretrained("pretrained_dres_1/version_0/checkpoints/epoch=2-val_loss=1.8490.ckpt")
        
    def initialise_pretrained(self, ckpt_dir):
        ckpt_dir = os.path.join(Path(__file__).parent.parent, 'logs', ckpt_dir)
        state_dict = torch.load(ckpt_dir, map_location='cpu')['state_dict']
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            print(f"Loading {name} parameters")
            name = name[6:] #remove 'model.'
            own_state[name].copy_(param)

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
        
        # out = self.mlp1(out)
        # out = self.mlp2(out)
        # out = self.mlp3(out)

        out = self.mlp1(out)
        out = self.mlp2(out)
        class_preds = self.mlp3(out)
        out = self.mlp4(class_preds * self.class_scaling)

        batch['class_pred'] = class_preds
        batch['pred'] = out
        return out


if __name__ == "__main__":
    rand_live = torch.randint(low=0, high=255, size=(8, 3, 256, 256)).to(dtype=torch.float32)
    rand_bottle = torch.randint(low=0, high=255, size=(8, 3, 256, 256)).to(dtype=torch.float32)

    batch = {
        "rgb0": rand_live,
        "vmap0": rand_live,
        "rgb1": rand_bottle,
        "vmap1": rand_bottle,
    }

    model = DRes_Class_N_Reg(90)
    print("Parameter cound: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    out = model(batch)
    print(out.shape)
    # print(out)




