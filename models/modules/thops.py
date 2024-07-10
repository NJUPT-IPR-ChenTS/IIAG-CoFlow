
import torch

#####有用
def sum(tensor, dim=None, keepdim=False):
    if dim is None:
        # sum up all dim
        return torch.sum(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.sum(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor

#####有用
def mean(tensor, dim=None, keepdim=False):
    if dim is None:
        # mean all dim
        return torch.mean(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor

######重要
def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]

###有用
def cat_feature(tensor_a, tensor_b, type="norm"):
    B = tensor_a.size(0)
    if type == "norm":
        return torch.cat((tensor_a, tensor_b), dim=1)
    elif type == "cross":  # 实现交叉拼接操作
        tensor = torch.stack((tensor_a, tensor_b), dim=2)
        tensor = tensor.view(B, -1, tensor.size(3), tensor.size(4))
        return tensor

###有用
def pixels(tensor):
    return int(tensor.size(2) * tensor.size(3))