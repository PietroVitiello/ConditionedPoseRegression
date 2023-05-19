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

class MSE_loss():
    def __init__(self) -> None:
        self.loss = nn.MSELoss()

    def __call__(self, batch: dict):
        batch['loss'] = self.loss(batch['pred'], batch['label'])

class CE_MSE_loss():
    def __init__(self) -> None:
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def __call__(self, batch: dict):
        class_loss = self.ce(batch['class_pred'], batch['class_label'])
        reg_loss = self.mse(batch['pred'], batch['label'])
        batch['loss'] = class_loss + reg_loss

class KL_Divergence():
    def __init__(self,
                 n_steps: int,
                 bacthes_per_step: int = 50,
                 n_classes: int = 90,
                 initial_steps: int = 0) -> None:
        self.step = 0
        self.std = 5
        n_updates = int(n_steps // bacthes_per_step)
        self.step_std_update = (self.std - 0.001) / n_updates
        self.n_classes = n_classes
        self.initial_steps = bacthes_per_step * initial_steps
        self.max_step_n = n_steps + self.initial_steps
        self.batches_per_step = bacthes_per_step

        self.kl = nn.KLDivLoss(reduction='none', log_target = False)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def get_dist(self, means: float):
        bs = means.shape[0]
        distributions = torch.zeros((bs, self.n_classes), device=means.device)
        points = torch.linspace(-45.5, 45.5, self.n_classes+1, device=means.device)
        for b_id in range(bs):
            pdf = torch.distributions.normal.Normal(means[b_id], self.std)
            cdf_up = pdf.cdf(torch.Tensor(points[1:]))
            cdf_down = pdf.cdf(torch.Tensor(points[:-1]))
            distributions[b_id,:] = cdf_up - cdf_down
        return distributions
    
    def update_std(self):
        if self.step < self.initial_steps:
            pass
        elif self.step < self.max_step_n:
            self.step += 1
            if self.step % self.batches_per_step:
                self.std -= self.step_std_update

    def __call__(self, batch: dict):
        pred_dist = self.log_softmax(batch['pred'])
        gt_means = torch.argmax(batch['label'], dim=1) - 44.5
        gt_dist = self.get_dist(gt_means)
        self.update_std()
        divergence = self.kl(pred_dist, gt_dist)
        batch['loss'] = torch.mean(torch.sum(divergence, dim=1))
        return batch['loss']
    
if __name__ == '__main__':
    kldiv = KL_Divergence(100, 2)

    # means = 0
    means = torch.tensor([-10, 0, 20])

    print(kldiv.get_dist(means))
    print(kldiv.get_dist(means)[0,45])
    print(torch.sum(kldiv.get_dist(means)))
    print(len(kldiv.get_dist(means)))

    # print(kldiv.get_dist(0).exp())

    # print(kldiv.get_dist(0).exp()[45])
    
    input_ = torch.rand((4,90))
    labels = torch.rand((4,90))
    print(kldiv({'pred': input_, 'label': labels}))