from typing import Tuple
import warnings
from math import ceil
import interp_methods
from fractions import Fraction
from functools import lru_cache;
from methodtools import lru_cache as class_cache;

class NoneClass:
    pass
try:
    import torch
    from torch import nn
    nnModuleWrapped = nn.Module
except ImportError:
    warnings.warn('No PyTorch found, will work only with Numpy')
    torch = None
    nnModuleWrapped = NoneClass
try:
    import numpy
except ImportError:
    warnings.warn('No Numpy found, will work only with PyTorch')
    numpy = None
if numpy is None and torch is None:
    raise ImportError("Must have either Numpy or PyTorch but both not found")

@lru_cache(maxsize=40)
def resize(input, scale_factors=None, out_shape=None,
           interp_method=interp_methods.cubic, support_sz=None,
           antialiasing=True, by_convs=False, scale_tolerance=None,
           max_numerator=10, pad_mode='constant'):
    # get properties of the input tensor
    in_shape, n_dims = input.shape, input.ndim

    # fw stands for framework that can be either numpy or torch,
    # determined by the input type
    fw = numpy if type(input) is numpy.ndarray else torch
    eps = fw.finfo(fw.float32).eps
    device = input.device if fw is torch else None

    # set missing scale factors or output shapem one according to another,
    # scream if both missing. this is also where all the defults policies
    # take place. also handling the by_convs attribute carefully.
    scale_factors, out_shape, by_convs = set_scale_and_out_sz(in_shape,
                                                              out_shape,
                                                              scale_factors,
                                                              by_convs,
                                                              scale_tolerance,
                                                              max_numerator,
                                                              eps, fw)

    # sort indices of dimensions according to scale of each dimension.
    # since we are going dim by dim this is efficient
    sorted_filtered_dims_and_scales = [(dim, scale_factors[dim], by_convs[dim],
                                        in_shape[dim], out_shape[dim])
                                       for dim in sorted(range(n_dims),
                                       key=lambda ind: scale_factors[ind])
                                       if scale_factors[dim] != 1.]

    # unless support size is specified by the user, it is an attribute
    # of the interpolation method
    if support_sz is None:
        support_sz = interp_method.support_sz

    # output begins identical to input and changes with each iteration
    output = input

    # iterate over dims
    for (dim, scale_factor, dim_by_convs, in_sz, out_sz
         ) in sorted_filtered_dims_and_scales:
        # STEP 1- PROJECTED GRID: The non-integer locations of the projection
        # of output pixel locations to the input tensor
        projected_grid = get_projected_grid(in_sz, out_sz,
                                            scale_factor, fw, dim_by_convs,
                                            device)

        # STEP 1.5: ANTIALIASING- If antialiasing is taking place, we modify
        # the window size and the interpolation method (see inside function)
        cur_interp_method, cur_support_sz = apply_antialiasing_if_needed(
                                                                interp_method,
                                                                support_sz,
                                                                scale_factor,
                                                                antialiasing)

        # STEP 2- FIELDS OF VIEW: for each output pixels, map the input pixels
        # that influence it. Also calculate needed padding and update grid
        # accoedingly
        field_of_view = get_field_of_view(projected_grid, cur_support_sz, fw,
                                          eps, device)

        # STEP 2.5- CALCULATE PAD AND UPDATE: according to the field of view,
        # the input should be padded to handle the boundaries, coordinates
        # should be updated. actual padding only occurs when weights are
        # aplied (step 4). if using by_convs for this dim, then we need to
        # calc right and left boundaries for each filter instead.
        pad_sz, projected_grid, field_of_view = calc_pad_sz(in_sz, out_sz,
                                                            field_of_view,
                                                            projected_grid,
                                                            scale_factor,
                                                            dim_by_convs, fw,
                                                            device)

        # STEP 3- CALCULATE WEIGHTS: Match a set of weights to the pixels in
        # the field of view for each output pixel
        weights = get_weights(cur_interp_method, projected_grid, field_of_view)

        # STEP 4- APPLY WEIGHTS: Each output pixel is calculated by multiplying
        # its set of weights with the pixel values in its field of view.
        # We now multiply the fields of view with their matching weights.
        # We do this by tensor multiplication and broadcasting.
        # if by_convs is true for this dim, then we do this action by
        # convolutions. this is equivalent but faster.
        if not dim_by_convs:
            output = apply_weights(output, field_of_view, weights, dim, n_dims,
                                   pad_sz, pad_mode, fw)
        else:
            output = apply_convs(output, scale_factor, in_sz, out_sz, weights,
                                 dim, pad_sz, pad_mode, fw)
    return output

@lru_cache(maxsize=40)
def get_projected_grid(in_sz, out_sz, scale_factor, fw, by_convs, device=None):
    grid_sz = out_sz if not by_convs else scale_factor.numerator
    out_coordinates = fw_arange(grid_sz, fw, device)
    return (out_coordinates / float(scale_factor) +(in_sz - 1) / 2 - (out_sz - 1) / (2 * float(scale_factor)))

@lru_cache(maxsize=40)
def get_field_of_view(projected_grid, cur_support_sz, fw, eps, device):
    left_boundaries = fw_ceil(projected_grid - cur_support_sz / 2 - eps, fw)
    ordinal_numbers = fw_arange(ceil(cur_support_sz - eps), fw, device)
    return left_boundaries[:, None] + ordinal_numbers

@lru_cache(maxsize=40)
def calc_pad_sz(in_sz, out_sz, field_of_view, projected_grid, scale_factor,dim_by_convs, fw, device):
    if not dim_by_convs:
        pad_sz = [-field_of_view[0, 0].item(),field_of_view[-1, -1].item() - in_sz + 1]
        field_of_view += pad_sz[0]
        projected_grid += pad_sz[0]
    else:
        num_convs, stride = scale_factor.numerator, scale_factor.denominator
        left_pads = -field_of_view[:, 0]
        right_pads = (((out_sz - fw_arange(num_convs, fw, device) - 1)// num_convs)* stride+ field_of_view[:, -1]- in_sz + 1)
        pad_sz = list(zip(left_pads, right_pads))
    return pad_sz, projected_grid, field_of_view

@lru_cache(maxsize=40)
def get_weights(interp_method, projected_grid, field_of_view):
    weights = interp_method(projected_grid[:, None] - field_of_view)
    sum_weights = weights.sum(1, keepdims=True)
    sum_weights[sum_weights == 0] = 1
    return weights / sum_weights

@lru_cache(maxsize=40)
def apply_weights(input, field_of_view, weights, dim, n_dims, pad_sz, pad_mode,fw):
    tmp_input = fw_swapaxes(input, dim, 0, fw)
    tmp_input = fw_pad(tmp_input, fw, pad_sz, pad_mode)
    neighbors = tmp_input[field_of_view]
    tmp_weights = fw.reshape(weights, (*weights.shape, * [1] * (n_dims - 1)))
    tmp_output = (neighbors * tmp_weights).sum(1)
    return fw_swapaxes(tmp_output, 0, dim, fw)

@lru_cache(maxsize=40)
def apply_convs(input, scale_factor, in_sz, out_sz, weights, dim, pad_sz,pad_mode, fw):
    input = fw_swapaxes(input, dim, -1, fw)
    stride, num_convs = scale_factor.denominator, scale_factor.numerator
    tmp_out_shape = list(input.shape)
    tmp_out_shape[-1] = out_sz
    tmp_output = fw_empty(tuple(tmp_out_shape), fw, input.device)
    for conv_ind, (pad_sz, filt) in enumerate(zip(pad_sz, weights)):
        pad_dim = input.ndim - 1
        tmp_input = fw_pad(input, fw, pad_sz, pad_mode, dim=pad_dim)
        tmp_output[..., conv_ind::num_convs] = fw_conv(tmp_input, filt, stride)
    return fw_swapaxes(tmp_output, -1, dim, fw)

@lru_cache(maxsize=40)
def set_scale_and_out_sz(in_shape, out_shape, scale_factors, by_convs,scale_tolerance, max_numerator, eps, fw):
    if scale_factors is None and out_shape is None:
        raise ValueError("either scale_factors or out_shape should be provided")
    if out_shape is not None:
        out_shape = (list(out_shape) + list(in_shape[len(out_shape):])
                     if fw is numpy
                     else list(in_shape[:-len(out_shape)]) + list(out_shape))
        if scale_factors is None:
            scale_factors = [out_sz / in_sz for out_sz, in_sz in zip(out_shape, in_shape)]
    if scale_factors is not None:
        scale_factors = (scale_factors
                         if isinstance(scale_factors, (list, tuple))
                         else [scale_factors, scale_factors])
        scale_factors = (list(scale_factors) + [1] *
                         (len(in_shape) - len(scale_factors)) if fw is numpy
                         else [1] * (len(in_shape) - len(scale_factors)) +
                         list(scale_factors))
        if out_shape is None:
            out_shape = [ceil(scale_factor * in_sz)
                         for scale_factor, in_sz in
                         zip(scale_factors, in_shape)]
        if not isinstance(by_convs, (list, tuple)):
            by_convs = [by_convs] * len(out_shape)
        for ind, (sf, dim_by_convs) in enumerate(zip(scale_factors, by_convs)):
            if dim_by_convs:
                frac = Fraction(1/sf).limit_denominator(max_numerator)
                frac = Fraction(numerator=frac.denominator, denominator=frac.numerator)
            if scale_tolerance is None:
                scale_tolerance = eps
            if dim_by_convs and abs(frac - sf) < scale_tolerance:
                scale_factors[ind] = frac
            else:
                scale_factors[ind] = float(sf)
                by_convs[ind] = False
        return scale_factors, out_shape, by_convs

@lru_cache(maxsize=40)
def apply_antialiasing_if_needed(interp_method, support_sz, scale_factor,antialiasing):
    scale_factor = float(scale_factor)
    if scale_factor >= 1.0 or not antialiasing:
        return interp_method, support_sz
    cur_interp_method = (lambda arg: scale_factor *interp_method(scale_factor * arg))
    cur_support_sz = support_sz / scale_factor
    return cur_interp_method, cur_support_sz

@lru_cache(maxsize=40)
def fw_ceil(x, fw):
    if fw is numpy:
        return fw.int_(fw.ceil(x))
    else:
        return x.ceil().long()
    
@lru_cache(maxsize=40)
def fw_floor(x, fw):
    if fw is numpy:
        return fw.int_(fw.floor(x))
    else:
        return x.floor().long()
    
@lru_cache(maxsize=40)
def fw_cat(x, fw):
    if fw is numpy:
        return fw.concatenate(x)
    else:
        return fw.cat(x)
    
@lru_cache(maxsize=40)
def fw_swapaxes(x, ax_1, ax_2, fw):
    if fw is numpy:
        return fw.swapaxes(x, ax_1, ax_2)
    else:
        return x.transpose(ax_1, ax_2)
    
@lru_cache(maxsize=40)
def fw_pad(x, fw, pad_sz, pad_mode, dim=0):
    if pad_sz == (0, 0):
        return x
    if fw is numpy:
        pad_vec = [(0, 0)] * x.ndim
        pad_vec[dim] = pad_sz
        return fw.pad(x, pad_width=pad_vec, mode=pad_mode)
    else:
        if x.ndim < 3:
            x = x[None, None, ...]
        pad_vec = [0] * ((x.ndim - 2) * 2)
        pad_vec[0:2] = pad_sz
        return fw.nn.functional.pad(x.transpose(dim, -1), pad=pad_vec,mode=pad_mode).transpose(dim, -1)
    
@lru_cache(maxsize=40)
def fw_conv(input, filter, stride):
    reshaped_input = input.reshape(1, 1, -1, input.shape[-1])
    reshaped_output = torch.nn.functional.conv2d(reshaped_input,filter.view(1, 1, 1, -1),stride=(1, stride))
    return reshaped_output.reshape(*input.shape[:-1], -1)

@lru_cache(maxsize=40)
def fw_arange(upper_bound, fw, device):
    if fw is numpy:
        return fw.arange(upper_bound)
    else:
        return fw.arange(upper_bound, device=device)

@lru_cache(maxsize=40)
def fw_empty(shape, fw, device):
    if fw is numpy:
        return fw.empty(shape)
    else:
        return fw.empty(size=(*shape,), device=device)
