import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from PointNet.pointnet2_msg_enc2 import PointNet2_Encoder_v2

class PointNetv2_RReg(nn.Module):

    def __init__(self) -> None:
        super(PointNetv2_RReg, self).__init__()

        self.encoder = PointNet2_Encoder_v2(output_dim=256, data_dim=6)
        self.fusion = nn.Sequential(nn.Linear(in_features=512, out_features=256, bias=True),
                                    # nn.BatchNorm1d(256),
                                    nn.InstanceNorm1d(256),
                                    nn.LeakyReLU(inplace=True))
        self.mlp2 = nn.Sequential(nn.Linear(in_features=256, out_features=128, bias=True),
                                #   nn.BatchNorm1d(128),
                                  nn.InstanceNorm1d(128),
                                  nn.LeakyReLU(inplace=True))
        self.mlp3 = nn.Sequential(nn.Linear(in_features=128, out_features=1, bias=True),
                                  nn.Identity(inplace=True))
        
        # self.initialise_pretrained("pnetv2_rot_4/version_0/checkpoints/epoch=0-val_ori_error=6.4754.ckpt")
        self.initialise_pretrained("pnetv2_rotreg_2/version_0/checkpoints/last.ckpt")
        
    def initialise_pretrained(self, ckpt_dir):
        ckpt_dir = os.path.join(Path(__file__).parent.parent, 'logs', ckpt_dir)
        state_dict = torch.load(ckpt_dir, map_location='cpu')['state_dict']
        own_state = self.state_dict()
        # print(own_state.keys(), "\n\n")
        for name, param in state_dict.items():
            name = name.split(".", 1)[1]
            valid_component = True
            # for layer_name in ["fusion", "mlp2", "mlp3"]:
            #     if layer_name == name.split(".")[0]:
            #         valid_component = False

            if valid_component:
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                print(f"Loading {name} parameters")
                own_state[name].copy_(param)

    def forward(self, batch: dict) -> torch.Tensor:
        encoded_live = self.encoder(batch["pc0"])
        encoded_bottleneck = self.encoder(batch["pc1"])

        # out = encoded_live - encoded_bottleneck
        out = torch.concat((encoded_live, encoded_bottleneck), dim=1)
        out = self.fusion(out)
        out = self.mlp2(out)
        out = self.mlp3(out)

        batch['rot_pred'] = out
        return out


if __name__ == "__main__":
    rand_live = torch.randint(low=-1, high=1, size=(8, 9, 1000)).to(dtype=torch.float32)
    rand_bottle = torch.randint(low=-1, high=1, size=(8, 9, 1000)).to(dtype=torch.float32)

    batch = {
        "pc0": rand_live,
        "pc1": rand_bottle,
    }

    model = PointNetv2_RReg()
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    out = model(batch)
    print(out.shape)
    # print(out)
