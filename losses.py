import torch
import torch.nn as nn

class cosine_loss():
    
    def __init__(self, reduction: str = 'mean') -> None:
        self.reduction = reduction
        self.cos = nn.CosineSimilarity(dim=1)

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduction == 'mean':
            return torch.mean(x)

    def __call__(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        normalised_pred = pred / torch.norm(pred)
        cos_sim = self.cos(normalised_pred, label)
        cos_sim = self.reduce(cos_sim)
        return - cos_sim
    

class rotation_loss():
    
    def __init__(self, reduction: str = 'mean') -> None:
        self.cos = cosine_loss(reduction)

    def __call__(self, batch: dict):
        x_pred, y_pred = torch.split(batch['pred'], [3,3], dim=-1)
        x_label, y_label = torch.split(batch['label'], [3,3], dim=-1)
        x_cosine = self.cos(x_pred, x_label)
        y_cosine = self.cos(y_pred, y_label)
        batch['loss'] = x_cosine + y_cosine +2

    # def __call__(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    #     x_pred, y_pred = torch.split(pred, [3,3], dim=-1)
    #     x_label, y_label = torch.split(label, [3,3], dim=-1)
    #     x_cosine = self.cos(x_pred, x_label)
    #     y_cosine = self.cos(y_pred, y_label)
    #     return x_cosine + y_cosine +2

class rotvec_loss():
    
    def __init__(self, reduction: str = 'mean') -> None:
        self.mse = nn.MSELoss(reduction=reduction)
        self.cos = cosine_loss(reduction)

    def get_rotvec_parameters(self, rotvec):
        magnitude = torch.linalg.norm(rotvec)
        axis = rotvec / magnitude
        return magnitude, axis

    def __call__(self, batch: dict):
        pred = batch['pred']
        label = batch['label']
        mag_pred, axis_pred = self.get_rotvec_parameters(pred)
        mag_label, axis_label = self.get_rotvec_parameters(label)
        mag_loss = self.mse(mag_pred, mag_label)
        axis_loss = self.cos(axis_pred, axis_label)
        batch['loss'] = mag_loss + axis_loss + 1

class cross_entropy_loss():
    def __init__(self) -> None:
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, batch: dict):
        batch['loss'] = self.loss(batch['pred'], batch['label'])