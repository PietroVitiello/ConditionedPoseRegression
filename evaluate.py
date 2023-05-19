import os
from pathlib import Path
import torch
import numpy as np
import open3d as o3d

from Dataset.blender_dataset_1dof import BlenderDataset_1dof
from CNN.dresv1_pretrain import PreTrained_DRes_Classification

def get_batch(data):
    batch = {
        "rgb0": torch.tensor(data["rgb0"]).unsqueeze(0),
        "rgb1": torch.tensor(data["rgb1"]).unsqueeze(0),
        "vmap0": torch.tensor(data["vmap0"]).unsqueeze(0),
        "vmap1": torch.tensor(data["vmap1"]).unsqueeze(0),
    }
    return batch

def get_angle_pred(pred):
    pred = pred.detach().numpy()[0]
    angle_id = np.argmax(pred)
    print(-44.5 + angle_id)
    return -44.5 + angle_id

def get_rotation_matrix(angle_pred):
    R = np.eye(3,3)
    c = np.cos(np.deg2rad(angle_pred))
    s = np.sin(np.deg2rad(angle_pred))
    R[:2,:2] = np.array([[c, -s], [s, c]])
    return R

def img_to_o3d_pcd(depth: np.ndarray, intrinsic_matrix: np.ndarray, rgb=None):
    if depth.dtype == np.int32:
        depth = depth.astype(np.uint16)
    assert depth.dtype == np.uint16, f'The depth image must be \'mm\' stored in the dtype np.uint16 and not {depth.dtype}'
    if rgb is not None:
        assert rgb.dtype == np.uint8, f'The RGB image must be stored in the dtype np.uint8 and not {depth.rgb}'

    intrinsic_matrix_o3d = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_matrix_o3d.intrinsic_matrix = intrinsic_matrix
    depth_o3d = o3d.geometry.Image(depth.copy())
    if rgb is not None:
        rgb_o3d = o3d.geometry.Image(rgb.copy())
        rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(color=rgb_o3d, depth=depth_o3d,
                                                                    depth_scale=1000,
                                                                    convert_rgb_to_intensity=False)
        pcd_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_o3d, intrinsic=intrinsic_matrix_o3d,
                                                                extrinsic=np.eye(4))
    else:
        pcd_o3d = o3d.geometry.PointCloud.create_from_depth_image(depth=depth_o3d, intrinsic=intrinsic_matrix_o3d,
                                                                extrinsic=np.eye(4))
    return pcd_o3d

def vmap_to_o3d_pcd(vmap: np.ndarray):
    # if depth.dtype == np.int32:
    #     depth = depth.astype(np.uint16)
    # assert depth.dtype == np.uint16, f'The depth image must be \'mm\' stored in the dtype np.uint16 and not {depth.dtype}'
    # if rgb is not None:
    #     assert rgb.dtype == np.uint8, f'The RGB image must be stored in the dtype np.uint8 and not {depth.rgb}'

    # intrinsic_matrix_o3d = o3d.camera.PinholeCameraIntrinsic()
    # intrinsic_matrix_o3d.intrinsic_matrix = intrinsic_matrix_o3d
    print(vmap.shape)
    # vmap = (vmap * 1000).astype(np.uint16)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vmap.reshape((-1, 3)))
    return pcd

def get_transformation(data, R_pred):
    T_pred = np.eye(4,4)
    T_pred[:3,:3] = R_pred
    T_WC = data["T_WC_opencv"]
    T_CW = data["T_CW_opencv"]
    T_delta_cam = T_CW @ T_pred @ T_WC
    return T_delta_cam    

def visualise_pred(pcd1, pcd2, data, R):
    pcd1.paint_uniform_color([1, 0.706, 0])
    pcd2.paint_uniform_color([0, 0.651, 0.929])

    cl, ind = pcd1.remove_statistical_outlier(nb_neighbors=10, std_ratio=4.0)
    pcd1 = pcd1.select_by_index(ind)
    cl, ind = pcd2.remove_statistical_outlier(nb_neighbors=10, std_ratio=4.0)
    pcd2 = pcd2.select_by_index(ind)

    T_delta = get_transformation(data, R)
    center_pcd1 = pcd1.get_center()
    pcd1.rotate(R=T_delta[:3,:3], center=center_pcd1)

    translation = pcd2.get_center() - center_pcd1
    pcd1.translate(translation)

    print(f"center 1: {pcd1.get_center()}")
    print(f"center 2: {pcd2.get_center()}")


    o3d.visualization.draw([pcd1, pcd2])

    # pcd1.translate(np.array([1,1,1]))
    # print(f"center 1: {pcd1.get_center()}")
    # print(f"center 2: {pcd2.get_center()}")
    # o3d.visualization.draw([pcd1, pcd2])

class PoseEstimator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = PreTrained_DRes_Classification(90)

    def forward(self, x):
        return self.model(x)

data = BlenderDataset_1dof(debug=True)
model = PoseEstimator()

ckpt_dir = "pt_dres_kl_1/version_4/checkpoints/epoch=6-val_ori_error=3.1384.ckpt"
ckpt_dir = os.path.join(Path(__file__).parent.parent, 'logs', ckpt_dir)
model.load_state_dict(torch.load(ckpt_dir)['state_dict'])
model.eval()

for dpoint in data:
    # print(dpoint.keys())
    # print(dpoint['vmap0'][0,40,:])

    batch = get_batch(dpoint)
    pred = model(batch)
    angle_pred = get_angle_pred(pred)
    rot_mtx = get_rotation_matrix(angle_pred)

    # print(pred)
    # print(rot_mtx)

    pcd1 = vmap_to_o3d_pcd(dpoint['vmap0'].transpose(1,2,0))
    pcd2 = vmap_to_o3d_pcd(dpoint['vmap1'].transpose(1,2,0))
    
    visualise_pred(pcd1, pcd2, dpoint, rot_mtx)

        