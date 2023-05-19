import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from CNN.backbone import ResNet_Backbone, ConvBlock, ResNet_Block_2

class ResNet_Classification(nn.Module):

    def __init__(self) -> None:
        super(ResNet_Classification, self).__init__()

        self.encoder = ResNet_Backbone()

        self.mlp1 = nn.Sequential(nn.Linear(in_features=256, out_features=128, bias=True),
                                  nn.SELU(inplace=True))
        self.mlp2 = nn.Sequential(nn.Linear(in_features=128, out_features=64, bias=True),
                                  nn.SELU(inplace=True))
        self.mlp3 = nn.Sequential(nn.Linear(in_features=64, out_features=3, bias=True),
                                  nn.Identity(inplace=True))

        # self.mlp1 = nn.Sequential(nn.Linear(in_features=256, out_features=256, bias=True),
        #                          nn.SELU(inplace=True))
        # self.mlp2 = nn.Sequential(nn.Linear(in_features=256, out_features=128, bias=True),
        #                          nn.SELU(inplace=True))
        # self.mlp3 = nn.Sequential(nn.Linear(in_features=128, out_features=128, bias=True),
        #                          nn.SELU(inplace=True))
        # self.mlp4 = nn.Sequential(nn.Linear(in_features=128, out_features=128, bias=True),
        #                          nn.SELU(inplace=True))
        # self.mlp5 = nn.Sequential(nn.Linear(in_features=128, out_features=200, bias=True),
        #                          nn.Softmax(dim=1))

    #     self._init_weights()

    # def _init_weights(self):
    #     r"""Initiate parameters in the transformer model."""
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)

    def forward(self, batch: dict) -> torch.Tensor:
        encoded_live = self.encoder(batch["rgb0"], batch["vmap0"])

        print(f"Encoded live:\n{encoded_live[:,0,int(encoded_live.shape[2]/2),:]}\n")

        print(encoded_live.shape)
        
        # out = self.mlp1(out)
        # out = self.mlp2(out)
        # out = self.mlp3(out)

        out = self.mlp1(out)
        out = self.mlp2(out)
        out = self.mlp3(out)
        # out = self.mlp4(out)
        # out = self.mlp5(out)

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

    model = ResNet_Classification()
    print("Parameter cound: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    out = model(batch)
    print(out.shape)
    # print(out)




