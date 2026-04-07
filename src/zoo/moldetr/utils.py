import math
import torch 
import torch.nn as nn
import torch.nn.functional as F 
def inverse_sigmoid(x: torch.Tensor, eps: float=1e-5) -> torch.Tensor:
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))
def deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights):
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape
    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level].flatten(2).permute(
            0, 2, 1).reshape(bs * n_head, c, h, w)
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute(
            0, 2, 1, 3, 4).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points)
    output = (torch.stack(
        sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).reshape(bs, n_head * c, Len_q)
    return output.permute(0, 2, 1)
import math 
def bias_init_with_prob(prior_prob=0.01):
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init
def get_activation(act: str, inpace: bool=True):
    act = act.lower()
    if act == 'silu':
        m = nn.SiLU()
    elif act == 'relu':
        m = nn.ReLU()
    elif act == 'leaky_relu':
        m = nn.LeakyReLU()
    elif act == 'silu':
        m = nn.SiLU()
    elif act == 'gelu':
        m = nn.GELU()
    elif act is None:
        m = nn.Identity()
    elif isinstance(act, nn.Module):
        m = act
    else:
        raise RuntimeError('')  
    if hasattr(m, 'inplace'):
        m.inplace = inpace
    return m 