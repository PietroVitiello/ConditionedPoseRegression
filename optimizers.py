import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau


def build_optimizer(model, name, lr):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.1)
    else:
        raise ValueError(f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")


def build_scheduler(name, optimizer, max_epochs):
    """
    Returns:
        scheduler (dict):{
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
            'monitor': 'val_f1', (optional)
            'frequency': x, (optional)
        }
    """
    scheduler = {'interval': 'step'}

    if name == "plateau":
        scheduler.update(
            {'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.85, patience=100, min_lr=1e-6, verbose=True),
             'monitor': 'training_window_loss'})
    elif name == 'MultiStepLR':
        scheduler.update(
            {'scheduler': MultiStepLR(optimizer, [3, 6, 9, 12], gamma=0.5)})
    elif name == 'CosineAnnealing':
        batch_size = 4
        total_dataset_size = 145177 * 0.98
        t_max = int((total_dataset_size / batch_size) * max_epochs)
        scheduler.update(
            {'scheduler': CosineAnnealingLR(optimizer, t_max, eta_min=1e-6)})
    elif name == 'ExponentialLR':
        scheduler.update(
            {'scheduler': ExponentialLR(optimizer, 0.999992)})
    else:
        raise NotImplementedError()

    return scheduler