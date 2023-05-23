import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, downsample: bool=False) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm2d(num_features=ch_out),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, padding=1)
        )
        self.downsample = None
        if downsample:
            # self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
    
class squeeze_excite(nn.Module):
    def __init__(self, frame_size: int, n_channels: int, ratio: int = 16) -> None:
        super().__init__()
        self.global_pool = nn.AvgPool2d(kernel_size=frame_size)
        self.squeeze = nn.Sequential(
            nn.Linear(n_channels, n_channels//ratio),
            nn.ReLU()
        )
        self.excite = nn.Sequential(
            nn.Linear(n_channels//ratio, n_channels),
            nn.Sigmoid()
        )        

    def forward(self, in_block):
        x = self.global_pool(in_block)
        x = self.squeeze(x)
        x = self.excite(x)
        return torch.matmul(in_block, x)
    
class Single_ResNet_Block(nn.Module):
    def __init__(self, ch_in: int, ch_out:int, downsample:bool = False) -> None:
        super().__init__()
        self.block = ConvBlock(ch_in, ch_out, downsample=downsample)
        self.residual_layer = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1)

    def forward(self, x: torch.Tensor):
        out = self.block(x)
        res = self.residual_layer(x)
        return out + res
    
class ResNet_Block(nn.Module):
    def __init__(self, ch_in: int, ch_out:int, downsample:bool) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(ch_in, ch_out, downsample=downsample),
            ConvBlock(ch_out, ch_out)
        )
        self.residual_layer = nn.Sequential(
            nn.InstanceNorm2d(num_features=ch_in),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1)
        )
        self.downsample = None
        if downsample:
            # self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        

    def forward(self, x: torch.Tensor):
        out = self.block(x)
        res = self.residual_layer(x)
        if self.downsample is not None:
            res = self.downsample(res)
        return out + res
    
class ResNet_Block_0(ResNet_Block):
    def __init__(self, ch_in: int, ch_out:int) -> None:
        super().__init__(ch_in, ch_out, downsample=False)

class ResNet_Block_2(ResNet_Block):
    def __init__(self, ch_in: int, ch_out:int) -> None:
        super().__init__(ch_in, ch_out, downsample=True)

class ResNet_Encoder(nn.Module):

    def __init__(self) -> None:
        super(ResNet_Encoder, self).__init__()
        self.rgb_head = self.encoding_head()
        self.pos_head = self.encoding_head()
        self.fusion = Single_ResNet_Block(512, 256)
        self.resnet = self.resnet_end()

    def encoding_head(self):
        return nn.Sequential(
            nn.InstanceNorm2d(num_features=3),
            nn.Conv2d(3, 128, kernel_size=1),
            ResNet_Block_0(128, 128),
            ResNet_Block_2(128, 256)
        )

    def resnet_end(self):
        return nn.Sequential(
            ResNet_Block_2(256, 512),
            ResNet_Block_2(512, 512),
        )

    def forward(self, rgb, vmap):
        rgb_enc = self.rgb_head(rgb)
        vmap_enc = self.pos_head(vmap)

        out = torch.concat((rgb_enc, vmap_enc), dim=1)
        out = self.fusion(out)

        return self.resnet(out)



if __name__ == "__main__":
    rand_live = torch.randint(low=0, high=255, size=(8, 3, 256, 256), dtype=torch.float32)
    rand_bottle = torch.randint(low=0, high=255, size=(8, 3, 256, 256), dtype=torch.float32)
    # print(rand_live)
    # print(rand_bottle)

    model = ResNet_Encoder()
    out = model(rand_live, rand_bottle)
    print(out.shape)

