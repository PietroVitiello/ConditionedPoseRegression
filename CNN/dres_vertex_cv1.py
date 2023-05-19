import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from CNN.backbone import ResNet_VertexMap_Backbone, ConvBlock, ResNet_Block_2

class DenseResNet_VertexMap_Classification(nn.Module):

    def __init__(self, n_classes: int = 18) -> None:
        super(DenseResNet_VertexMap_Classification, self).__init__()

        self.encoder = ResNet_VertexMap_Backbone()
        self.fusion = ConvBlock(512, 256, downsample=False)
        # self.fusion = ConvBlock(256, 256, downsample=False)
        self.resnet = self.resnet_end()

        self.mlp1 = nn.Sequential(nn.Linear(in_features=512, out_features=256, bias=True),
                                 nn.SELU(inplace=True))
        self.mlp2 = nn.Sequential(nn.Linear(in_features=256, out_features=128, bias=True),
                                 nn.SELU(inplace=True))
        self.mlp3 = nn.Sequential(nn.Linear(in_features=128, out_features=n_classes, bias=True),
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

    def resnet_end(self):
        return nn.Sequential(
            ResNet_Block_2(256, 512),
            ResNet_Block_2(512, 1024),
            ResNet_Block_2(1024, 2048),
            ResNet_Block_2(2048, 1024),
            ResNet_Block_2(1024, 512)
        )

    def forward(self, batch: dict) -> torch.Tensor:
        encoded_live = self.encoder(batch["vmap0"])
        encoded_bottleneck = self.encoder(batch["vmap1"])

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

    model = DenseResNet_VertexMap_Classification()
    print("Parameter cound: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    out = model(batch)
    print(out.shape)
    # print(out)




