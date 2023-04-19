from .transformer_blocks import CA_Block_LayerNormBefore as CA_Block, SA_Block_LayerNormBefore as SA_Block
from CNN.resnet_blocks import ResNet_Block

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class Deep_ResNet_Transformer_2_RGB(nn.Module):

    def __init__(self, rescale_255: bool = True) -> None:
        super(Deep_ResNet_Transformer_2_RGB, self).__init__()
        self.rescale_255 = rescale_255

        self.encoder = nn.Sequential(
            ResNet_Block(3, 128),
            ResNet_Block(128, 256, 256)
        )

        self.pos_encoding = Parameter(
            data = torch.randn(1,256,256),
            requires_grad = True
        ) 

        self.ca = CA_Block(seq_len=256, embed_dim=256, num_heads=4)
        self.sa1 = SA_Block(seq_len=256, embed_dim=256, num_heads=4)
        self.sa2 = SA_Block(seq_len=256, embed_dim=256, num_heads=4)
        self.sa3 = SA_Block(seq_len=256, embed_dim=256, num_heads=4)
        self.sa4 = SA_Block(seq_len=256, embed_dim=256, num_heads=4)
        self.sa5 = SA_Block(seq_len=256, embed_dim=256, num_heads=4)

        self.mlp1 = nn.Sequential(nn.Linear(in_features=256, out_features=128, bias=True),
                                 nn.SELU(inplace=True))
        self.mlp2 = nn.Sequential(nn.Linear(in_features=128, out_features=64, bias=True),
                                 nn.SELU(inplace=True))
        self.mlp3 = nn.Sequential(nn.Linear(in_features=64, out_features=9, bias=True),
                                 nn.Identity(inplace=True))

        self._init_weights()

    def _init_weights(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, live_images: torch.Tensor, bottleneck_images: torch.Tensor) -> torch.Tensor:
        encoded_live = self.encoder(pre_process_rgb(live_images, size=256, rescale_255=self.rescale_255)).permute(0, 2, 3, 1).view(-1, 256, 256)
        encoded_bottleneck = self.encoder(pre_process_rgb(bottleneck_images, size=256, rescale_255=self.rescale_255)).permute(0, 2, 3, 1).view(-1, 256, 256)
        encoded_live = encoded_live + self.pos_encoding
        encoded_bottleneck = encoded_bottleneck + self.pos_encoding

        attention = self.ca(encoded_live, encoded_bottleneck)
        attention = self.sa1(attention)
        attention = self.sa2(attention)
        attention = self.sa3(attention)
        attention = self.sa4(attention)
        attention = self.sa5(attention)

        out = torch.mean(attention, dim=1)
        out = self.mlp1(out)
        out = self.mlp2(out)
        out = self.mlp3(out)
        return out


if __name__ == "__main__":
    rand_live = torch.randint(low=0, high=255, size=(8, 3, 128, 128))
    rand_bottle = torch.randint(low=0, high=255, size=(8, 3, 128, 128))
    # rand_live = torch.randint(low=-1, high=1, size=(8, 3, 128, 128))
    # rand_bottle = torch.randint(low=-1, high=1, size=(8, 3, 128, 128))
    # print(rand_live)
    # print(rand_bottle)

    model = Deep_ResNet_Transformer_2_RGB()
    out = model(rand_live, rand_bottle)
    print(out.shape)
    print(out)

