import numpy as np
import torch

def normalize_vector(v, return_mag=False):
    _device = v.device
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-9]).to(_device)))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if (return_mag == True):
        return v, v_mag[:, 0]
    else:
        return v

def row_wise_dot_product(tensor1: torch.Tensor,
                         tensor2: torch.Tensor):
    element_wise = tensor1 * tensor2
    row_wise = torch.sum(element_wise, dim = -1)
    return torch.unsqueeze(row_wise, dim=1)

# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out

def encode_rotation_matrix(rot):
    numpy_input = False
    len_1_input = False

    if isinstance(rot, np.ndarray):
        numpy_input = True
        rot = torch.tensor(rot)

    if len(rot.shape) == 2:
        len_1_input = True
        rot = rot.unsqueeze(dim=0)

    batch = rot.shape[0]
    encoding = rot[:, :, :2].permute(0, 2, 1).contiguous().view(batch, -1)

    if len_1_input:
        encoding = encoding.squeeze(0)

    if numpy_input:
        encoding = encoding.cpu().detach().numpy()

    return encoding

def decode_rotation_encoding(encoding):
    numpy_input = False
    len_1_input = False
    float32 = False

    if isinstance(encoding, torch.Tensor):
        if encoding.dtype == torch.float32:
            float32 = True
    elif isinstance(encoding, np.ndarray):
        if encoding.dtype == np.float32:
            float32 = True

    if isinstance(encoding, np.ndarray):
        numpy_input = True
        encoding = torch.tensor(encoding)

    if len(encoding.shape) == 1:
        len_1_input = True
        encoding = encoding.unsqueeze(dim=0)

    # change the encoding to double precision
    encoding = encoding.to(dtype=torch.float64)

    # Grab the x and y-axis of the rotation matrix
    x_raw = encoding[:, 0:3]  # batch*3
    y_raw = encoding[:, 3:6]  # batch*3

    # Ensure x_raw != (0, 0, 0). If this is the case, set it to (1., 0., 0.)
    # x_raw[(x_raw == 0).all(dim=1)] = torch.tensor([1., 0., 0.], dtype=x_raw.dtype, device=x_raw.device)
    diff = torch.tensor([1., 0., 0.], dtype=x_raw.dtype, device=x_raw.device) - x_raw[(x_raw == 0).all(dim=1)]
    x_raw[(x_raw == 0).all(dim=1)] = x_raw[(x_raw == 0).all(dim=1)] + diff

    # Ensure y_raw != (0, 0, 0). If this is the case, set it to (0., 1., 0.)
    # y_raw[(y_raw == 0).all(dim=1)] = torch.tensor([0., 1., 0.], dtype=y_raw.dtype, device=y_raw.device)
    diff = torch.tensor([0., 1., 0.], dtype=y_raw.dtype, device=y_raw.device) - y_raw[(y_raw == 0).all(dim=1)]
    y_raw[(y_raw == 0).all(dim=1)] = y_raw[(y_raw == 0).all(dim=1)] + diff

    # Check if any of x_raw are parallel to y_raw
    cross_prod = torch.cross(x_raw, y_raw)
    x_parallel_to_y = (cross_prod * cross_prod).sum(dim=1) == 0.

    rots = torch.zeros((len(encoding), 3, 3), dtype=torch.float64, device=encoding.device)

    # Now, there are two ways to go about constructing the rotation matrix from the raw axis.
    # Case 1: x_raw and y_raw are not parallel

    # x axis
    x = normalize_vector(x_raw[torch.logical_not(x_parallel_to_y)])  # batch*3
    # y axis
    y = y_raw[torch.logical_not(x_parallel_to_y)] - (
                row_wise_dot_product(x, y_raw[torch.logical_not(x_parallel_to_y)]) * x)
    y = normalize_vector(y)  # batch*3
    # z axis
    z = cross_product(x, y)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    rots[torch.logical_not(x_parallel_to_y)] = torch.cat((x, y, z), dim=2)  # batch*3*3

    # Case 2: x_raw and y_raw are parallel
    if x_parallel_to_y.any():
        x = normalize_vector(x_raw[x_parallel_to_y])  # batch*3
        z = torch.zeros_like(x)
        x_parallel_to_y_axis = (x == torch.tensor([0., 1., 0.])).all(dim=1)

        z[torch.logical_not(x_parallel_to_y_axis)] = cross_product(x[torch.logical_not(x_parallel_to_y_axis)],
                                                                   torch.tensor([[0., 1., 0.]]))  # batch*3
        if x_parallel_to_y_axis.any():
            z[x_parallel_to_y_axis] = cross_product(torch.tensor([[1., 0., 0.]]), x[x_parallel_to_y_axis])
        z = normalize_vector(z)  # batch*3
        y = cross_product(z, x)  # batch*3

        x = x.view(-1, 3, 1)
        y = y.view(-1, 3, 1)
        z = z.view(-1, 3, 1)

        rots[x_parallel_to_y] = torch.cat((x, y, z), dim=2)  # batch*3*3

    if len_1_input:
        rots = rots.squeeze(0)

    if numpy_input:
        rots = rots.cpu().detach().numpy()
        if float32:
            rots = rots.astype(np.float32)
    elif float32:
        rots = rots.to(torch.float32)

    return rots