import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, downsample: bool=True) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, padding=1),
            # nn.BatchNorm2d(num_features=ch_out),
            nn.InstanceNorm2d(num_features=ch_out),
            # nn.SELU(inplace=True),
            nn.LeakyReLU(inplace=True)
        )
        self.downsample = None
        if downsample:
            self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        # print("X:", x.shape)
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
    
class ResNet_Block_0(nn.Module):
    def __init__(self, ch_in: int, ch_out:int, ch_between:int = None) -> None:
        super().__init__()
        ch_between = int(ch_out/2) if ch_between==None else ch_between
        self.block = nn.Sequential(
            ConvBlock(ch_in, ch_between, downsample=False),
            ConvBlock(ch_between, ch_out, False)
        )
        self.residual_layer = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1),
            # nn.BatchNorm2d(num_features=ch_out)
            nn.InstanceNorm2d(num_features=ch_out)
        )
        self.final_activation = nn.SELU(inplace=True)

    def forward(self, x: torch.Tensor):
        out = self.block(x)
        res = self.residual_layer(x)
        return self.final_activation(out + res)

class ResNet_Block_2(nn.Module):
    def __init__(self, ch_in: int, ch_out:int, ch_between:int = None) -> None:
        super().__init__()
        ch_between = int(ch_out/2) if ch_between==None else ch_between
        self.block = nn.Sequential(
            ConvBlock(ch_in, ch_between, downsample=False),
            ConvBlock(ch_between, ch_out, True)
        )
        self.residual_layer = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1),
            # nn.BatchNorm2d(num_features=ch_out),
            nn.InstanceNorm2d(num_features=ch_out),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.final_activation = nn.SELU(inplace=True)

    def forward(self, x: torch.Tensor):
        out = self.block(x)
        res = self.residual_layer(x)
        return self.final_activation(out + res)
    
class ResNet_Block_4(nn.Module):
    def __init__(self, ch_in: int, ch_out:int, ch_between:int = None) -> None:
        super().__init__()
        ch_between = int(ch_out/2) if ch_between==None else ch_between
        self.block = nn.Sequential(
            ConvBlock(ch_in, ch_between, downsample=True),
            ConvBlock(ch_between, ch_out, True)
        )
        self.residual_layer = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1),
            # nn.BatchNorm2d(num_features=ch_out),
            nn.InstanceNorm2d(num_features=ch_out),
            nn.AvgPool2d(kernel_size=4, stride=4),
        )
        self.final_activation = nn.SELU(inplace=True)

    def forward(self, x: torch.Tensor):
        out = self.block(x)
        res = self.residual_layer(x)
        return self.final_activation(out + res)
    
    



if __name__ == "__main__":
    rand_live = torch.randint(low=0, high=255, size=(8, 3, 128, 128))
    rand_bottle = torch.randint(low=0, high=255, size=(8, 3, 128, 128))
    # print(rand_live)
    # print(rand_bottle)

    model = ResNet_Block_4()
    out = model(rand_live, rand_bottle)
    print(out.shape)
    print(out)

