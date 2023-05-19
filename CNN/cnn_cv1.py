import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from CNN.backbone import ResNet_Backbone, ConvBlock, ResNet_Block_2

class SimpleCNN(nn.Module):

    def __init__(self, n_classes: int = 18) -> None:
        super(SimpleCNN, self).__init__()

        self.conv1 = self.conv_block(6, 64)
        self.conv2 = self.conv_block(64, 128)
        self.conv3 = self.conv_block(128, 128)
        self.conv4= self.conv_block(128, 256) # (B, C, 32, 32)

        self.conv5 = self.conv_block(512, 512)
        self.conv6 = self.conv_block(512, 256)
        self.conv7 = self.conv_block(256, 256) # (B, C, 4, 4)

        self.mlp1 = nn.Sequential(nn.Linear(in_features=1024, out_features=256, bias=True),
                                 nn.LeakyReLU())
        self.mlp2 = nn.Sequential(nn.Linear(in_features=256, out_features=128, bias=True),
                                 nn.LeakyReLU())
        self.mlp3 = nn.Sequential(nn.Linear(in_features=128, out_features=n_classes, bias=True),
                                 nn.Identity())

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

    def conv_block(self, ch_in, ch_out):
        return nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, padding=1),
            # nn.BatchNorm2d(num_features=ch_out),
            # nn.InstanceNorm2d(num_features=ch_out),
            # nn.SELU(inplace=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def resnet_end(self):
        return nn.Sequential(
            ResNet_Block_2(256, 512),
            ResNet_Block_2(512, 512),
            ResNet_Block_2(512, 1024),
            ResNet_Block_2(1024, 512),
            ResNet_Block_2(512, 256)
        )
    
    def encode(self, x):
        return self.conv4(self.conv3(self.conv2(self.conv1(x))))

    def forward(self, batch: dict) -> torch.Tensor:
        live = torch.concat((batch["rgb0"], batch["vmap0"]), dim=1)
        bottleneck = torch.concat((batch["rgb1"], batch["vmap1"]), dim=1)

        live = self.encode(live)
        bottleneck = self.encode(bottleneck)

        # out = encoded_live - encoded_bottleneck
        out = torch.concat((live, bottleneck), dim=1)
        out = self.conv7(self.conv6(self.conv5(out)))
        # print(f"Fusion:\n{out[:,0,int(out.shape[2]/2),:]}\n")
        out = torch.flatten(out, 1)
        
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

    model = SimpleCNN()
    print("Parameter cound: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    out = model(batch)
    print(out.shape)
    # print(out)




