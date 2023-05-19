import torch
import torchvision.transforms as T

from typing import Tuple, List, Union

import os.path as path
from PIL.Image import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

Transform = nn.Module

def random_augmentation(
    live_images: torch.Tensor,
    cond_images: torch.Tensor,
    transform_list: T.Compose = None,
    prob:  Union[List[float], np.ndarray] = None
) -> Tuple[torch.Tensor, torch.Tensor]:

    n_images = live_images.shape[0]
    stacked_images = torch.cat((live_images, cond_images), dim=0)
    transform = RandomApply(transform_list, prob)
    transformed_images = transform(stacked_images)
    return transformed_images[:n_images], transformed_images[n_images:]

class RandomApply(nn.Module):
    def __init__(
        self, 
        transform_list: List[Transform] = None,
        p: Union[List[float], np.ndarray] = None
     ) -> None:

        super().__init__()
        if transform_list == None:
            transform_list = [
                # T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                T.ColorJitter(brightness=0.5, contrast=0.8, saturation=0.8, hue=0.4),
                # T.ColorJitter(brightness=1, contrast=0, saturation=0, hue=0),
                # T.GaussianBlur(kernel_size=5, sigma=(0.1, 0.5)),
                T.GaussianBlur(kernel_size=5, sigma=(1, 2)),
                T.RandomInvert(p=1)
            ]
            p = np.array([0.4, 0.2, 0.01])
            # p = np.array([0.5, 0.2, 0.3])
            # p = np.array([1, 0, 0])
            # p = np.array([0,1,0])
            # p = np.array([0,0,1])
            # p = np.array([0, 0, 0])

        if not isinstance(p, np.ndarray):
            p = np.array(p)
        self.p = p
        self.transforms = transform_list

    def forward(self, images: torch.Tensor):
        random_num = np.random.rand()
        cumulative_prob = 0
        for t, t_prob in zip(self.transforms, self.p):
            cumulative_prob += t_prob
            if random_num < cumulative_prob:
                return t(images)
        return images


# def get_augmentation_list() -> T.Compose:

#     color_jitter = T.Compose([
#         T.ToPILImage(mode='F'),
#         T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
#         T.,
#         T.ToTensor()
#     ])

#     random_invert = T.Compose([
#         T.ToPILImage(mode='RGB'),
#         T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
#         T.ToTensor()
#     ])

#     # gaussian_

#     transform_list = T.Compose([
#             color_jitter,
#             T.GaussianBlur(kernel_size=5, sigma=(0.1, 0.5))
#             T

#         ])

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from head_cam.dataset import HeadCamDataset
    from head_cam.data_processing.normalise_img import common_normalisation
    
    dataset = HeadCamDataset(
        dataset_name="test",
        return_rgb=True
    )
    data = DataLoader(dataset, batch_size=32)
    for samples in data:
        live = common_normalisation(samples['rgb_1'])
        cond = common_normalisation(samples['rgb_2'])
        T.ToPILImage(mode='RGB')(live[0]).show()
        live, cond = random_augmentation(live, cond)
        T.ToPILImage(mode='RGB')(live[0]).show()
        T.ToPILImage(mode='RGB')(cond[0]).show()


class DataAugmentation():

    def __init__(self) -> None:
        pass


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass