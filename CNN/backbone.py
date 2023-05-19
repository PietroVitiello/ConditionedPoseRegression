import torch
import torch.nn as nn

from CNN.resnet_blocks import ResNet_Block_0, ResNet_Block_2, ResNet_Block_4, ConvBlock

class ResNet_Backbone(nn.Module):

    def __init__(self) -> None:
        super(ResNet_Backbone, self).__init__()
        self.rgb_head = self.encoding_head()
        self.pos_head = self.encoding_head()
        self.fusion = ConvBlock(512, 256, downsample=False)
        self.resnet = self.resnet_end()

    def encoding_head(self):
        return nn.Sequential(
            ResNet_Block_2(3, 128),
            ResNet_Block_2(128, 256, 256)
        )

    def resnet_end(self):
        return nn.Sequential(
            ResNet_Block_2(256, 512),
            ResNet_Block_0(512, 256, 256)
        )

    def forward(self, rgb, vmap):
        rgb_enc = self.rgb_head(rgb)
        vmap_enc = self.pos_head(vmap)

        out = torch.concat((rgb_enc, vmap_enc), dim=1)
        out = self.fusion(out)

        return self.resnet(out)
    
class ResNet_Backbone_512(nn.Module):

    def __init__(self) -> None:
        super(ResNet_Backbone_512, self).__init__()
        self.rgb_head = self.encoding_head()
        self.pos_head = self.encoding_head()
        self.fusion = ConvBlock(512, 256, downsample=False)
        self.resnet = self.resnet_end()

    def encoding_head(self):
        return nn.Sequential(
            ResNet_Block_2(3, 128),
            ResNet_Block_2(128, 256, 256)
        )

    def resnet_end(self):
        return nn.Sequential(
            ResNet_Block_2(256, 512),
            ResNet_Block_0(512, 512, 512)
        )

    def forward(self, rgb, vmap):
        rgb_enc = self.rgb_head(rgb)
        vmap_enc = self.pos_head(vmap)

        out = torch.concat((rgb_enc, vmap_enc), dim=1)
        out = self.fusion(out)

        return self.resnet(out)
    
class ResNet_Backbone_colvertex(nn.Module):

    def __init__(self) -> None:
        super(ResNet_Backbone_colvertex, self).__init__()
        self.encoder = self.encoding_head()
        self.resnet = self.resnet_end()

    def encoding_head(self):
        return nn.Sequential(
            ResNet_Block_2(6, 128),
            ResNet_Block_2(128, 512, 256)
        )

    def resnet_end(self):
        return nn.Sequential(
            ResNet_Block_2(512, 1024, 512),
            ResNet_Block_0(1024, 256, 512)
        )

    def forward(self, rgb, vmap):
        colored_vmap = torch.concat((rgb, vmap), dim=1)
        out = self.encoder(colored_vmap)
        return self.resnet(out)
    
class ResNet_VertexMap_Backbone(nn.Module):

    def __init__(self) -> None:
        super(ResNet_VertexMap_Backbone, self).__init__()
        self.pos_head = self.encoding_head()
        self.resnet = self.resnet_end()

    def encoding_head(self):
        return nn.Sequential(
            ResNet_Block_2(3, 128),
            ResNet_Block_2(128, 256, 256)
        )

    def resnet_end(self):
        return nn.Sequential(
            ResNet_Block_2(256, 512, 256),
            ResNet_Block_0(512, 256, 256)
        )

    def forward(self, vmap):
        vmap_enc = self.pos_head(vmap)
        return self.resnet(vmap_enc)
    

if __name__ == "__main__":
    rand_rgb = torch.randint(low=0, high=255, size=(8, 3, 256, 256)).to(dtype=torch.float32)
    rand_vmap = torch.randint(low=0, high=255, size=(8, 3, 256, 256)).to(dtype=torch.float32)

    model = ResNet_VertexMap_Backbone()
    print("Parameter cound: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    out = model(rand_rgb, rand_vmap)
    print(out.shape)
    # print(out)

        

