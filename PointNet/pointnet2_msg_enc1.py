import torch
import torch.nn as nn
import torch.nn.functional as F
from PointNet.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class PointNet2_Encoder_v1(nn.Module):
    def __init__(self, output_dim, data_dim=0):
        super(PointNet2_Encoder_v1, self).__init__()
        self.data_present = data_dim > 0
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], data_dim, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.mlp1 = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.InstanceNorm1d(512),
            nn.Dropout(0.2),
            # nn.LeakyReLU(inplace=True)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.InstanceNorm1d(256),
            nn.Dropout(0.2),
            # nn.LeakyReLU(inplace=True)
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(256, output_dim),
            # nn.BatchNorm1d(output_dim),
            nn.InstanceNorm1d(output_dim),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, xyz):
        bs, _, _ = xyz.shape
        if self.data_present:
            data = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            data = None
        l1_xyz, l1_points = self.sa1(xyz, data)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        out = l3_points.view(bs, 1024)
        out = self.mlp1(out)
        out = self.mlp2(out)
        out = self.mlp3(out)
        # x = F.log_softmax(x, -1)

        return out


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
    
if __name__ == "__main__":
    pnet = PointNet2_Encoder_v1(256, 3)
    print("Parameter cound: ", sum(p.numel() for p in pnet.parameters() if p.requires_grad))

    x = torch.rand((8,6,1000))

    out = pnet(x)
    print(out.shape)


