import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from PointNet.pnetv1_rot import PointNetv1_RCl
from PointNet.pnetv2_t import PointNetv2_tReg

class PointNetv1(nn.Module):

    def __init__(self, n_classes: int = 18) -> None:
        super(PointNetv1, self).__init__()
        self.n_classes = n_classes

        self.rot_model = PointNetv1_RCl(n_classes)
        self.t_model = PointNetv2_tReg()

    def get_pred_rot(self, class_pred: torch.Tensor):
            bin_size = 2 / self.n_classes
            rot_pred = (torch.argmax(class_pred.detach(), dim=1) + 0.5) * bin_size - 1
            return rot_pred
    
    def forward(self, batch: dict) -> torch.Tensor:
        rot_class_pred = self.rot_model(batch)
        rot_pred = self.get_pred_rot(rot_class_pred)
        t_pred, obj0_centre_pred, obj1_centre_pred = self.t_model(batch, rot_pred)

        batch['rot_pred'] = rot_class_pred
        batch['t_pred'] = t_pred
        batch['obj0_centre_pred'] = obj0_centre_pred
        batch['obj1_centre_pred'] = obj1_centre_pred
        return rot_class_pred, t_pred
    

if __name__ == "__main__":
    rand_live = torch.randint(low=-1, high=1, size=(8, 6, 1000)).to(dtype=torch.float32)
    rand_bottle = torch.randint(low=-1, high=1, size=(8, 6, 1000)).to(dtype=torch.float32)

    batch = {
        "pc0": rand_live,
        "pc1": rand_bottle,
    }

    model = PointNetv1(18)
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    out = model(batch)
    print(out[0].shape, out[1].shape)
    # print(out)
