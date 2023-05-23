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
                 n_batches: int,
                 bacthes_per_step: int = 50,
                 n_classes: int = 90,
                 initial_steps: int = 0) -> None:
        self.iteration = 0
        self.std = 1 #5
        n_updates = int(n_batches // bacthes_per_step)
        self.step_std_update = (self.std - 0.001) / n_updates
        self.initial_iter = bacthes_per_step * initial_steps
        self.max_step_n = n_batches + self.initial_iter
        self.batches_per_step = bacthes_per_step
        self.n_classes = n_classes
        self.bin_size = 90/self.n_classes

        self.kl = nn.KLDivLoss(reduction='none', log_target = False)
        self.log_softmax = nn.LogSoftmax(dim=1)
        

    def get_dist(self, means: float):
        bs = means.shape[0]
        distributions = torch.zeros((bs, self.n_classes), device=means.device)
        points = torch.linspace(-45, 45, self.n_classes+1, device=means.device)
        for b_id in range(bs):
            pdf = torch.distributions.normal.Normal(means[b_id], self.std)
            cdf_up = pdf.cdf(torch.Tensor(points[1:]))
            cdf_down = pdf.cdf(torch.Tensor(points[:-1]))
            distributions[b_id,:] = cdf_up - cdf_down
        return distributions
    
    def update_std(self):
        if self.iteration < self.initial_iter:
            pass
        elif self.iteration < self.max_step_n:
            self.iteration += 1
            if self.iteration % self.batches_per_step:
                self.std -= self.step_std_update

    def __call__(self, batch: dict):
        pred_dist = self.log_softmax(batch['pred'])
        gt_means = (torch.argmax(batch['label'], dim=1) + 0.5) * self.bin_size - 45
        gt_dist = self.get_dist(gt_means)
        self.update_std()
        divergence = self.kl(pred_dist, gt_dist)
        batch['loss'] = torch.mean(torch.sum(divergence, dim=1))
        return batch['loss']
    

class Loss_4Dof():
    def __init__(self,
                 rot_type: str,
                 kl_n_batches: int,
                 kl_bacthes_per_step: int = 50,
                 n_classes: int = 90,
                 kl_initial_steps: int = 0) -> None:
        
        self.rot_loss = KL_Divergence(kl_n_batches, kl_bacthes_per_step, n_classes, kl_initial_steps)
        self.mse = nn.MSELoss()
        self.rot_type = rot_type
        self.n_classes = n_classes
        self.bin_size = 90/self.n_classes

    def get_T_matrix(self, rot_pred: torch.Tensor, t_pred: torch.Tensor):
        if self.rot_type == 'reg':
            angles_z = rot_pred.detach()
        elif self.rot_type == 'class':
            angles_z = (torch.argmax(rot_pred.detach(), dim=1) + 0.5) * self.bin_size - 45
            angles_z = torch.deg2rad(angles_z)
        bs = rot_pred.shape[0]
        T = torch.eye(4, device=rot_pred.device).unsqueeze(0)
        T = T.repeat(bs, 1, 1)
        T[:,:3,3] = t_pred
        cosines = torch.cos(angles_z)
        sines = torch.sin(angles_z)
        T[:,0,0] = cosines
        T[:,1,1] = cosines
        T[:,0,1] = -sines
        T[:,1,0] = sines
        return T
    
    def get_pred_pos(self, obj_centres, rot_pred, t_pred):
        T = self.get_T_matrix(rot_pred, t_pred)
        centres_h = torch.cat((obj_centres, torch.ones(obj_centres.shape[0],1,device=obj_centres.device)), dim=1)
        pred_position = T @ centres_h.unsqueeze(2)
        return pred_position[:,:3,0]

    def __call__(self, batch: dict):
        rotation_loss = self.rot_loss({
            'pred': batch['rot_pred'],
            'label': batch['rot_label']
            })
        predicted_position = self.get_pred_pos(
            batch['obj0_centre'], batch['rot_pred'], batch['t_pred'])
        translation_loss = self.mse(predicted_position, batch['t_label'])
        batch['t_error'] = torch.mean(torch.norm(predicted_position - batch['t_label'], dim=1)).detach().cpu().numpy()
        batch['loss'] = rotation_loss + 10 * translation_loss
        print(f"Rotation Loss: {rotation_loss}")
        print(f"Translation Loss: {10 *translation_loss} \n")




    
if __name__ == '__main__':

    # rot_pred = torch.rand((2,90), device='cuda:0')
    # rot_label = torch.zeros((2,90), device='cuda:0')
    # rot_label[:, 45] = 1

    # obj_c = torch.rand((2,3), device='cuda:0')
    # t_pred = torch.rand((2,3), device='cuda:0')
    # t_label = torch.rand((2,3), device='cuda:0')

    # batch = {
    #     'rot_pred': rot_pred,
    #     'rot_label': rot_label,
    #     'obj1_centre': obj_c,
    #     't_pred': t_pred,
    #     't_label': t_label
    # }

    # loss = Loss_4Dof('class', 100, 2)
    # loss(batch)


    n_classes = 18
    kldiv = KL_Divergence(100, 2, n_classes)

    # means = 0
    means = torch.tensor([-10, 0, 20])

    print(kldiv.get_dist(means))
    # print(kldiv.get_dist(means)[0,45])
    print(torch.sum(kldiv.get_dist(means), dim=1))
    # print(len(kldiv.get_dist(means)))

    # print(kldiv.get_dist(0).exp())

    # print(kldiv.get_dist(0).exp()[45])
    
    input_ = torch.rand((4,n_classes))
    labels = torch.rand((4,n_classes))
    print(kldiv({'pred': input_, 'label': labels}))