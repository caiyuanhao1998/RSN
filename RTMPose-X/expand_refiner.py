"""Embed a trained high-resolution RSN branch in a wider branch.

Old channels keep their triangular-step positions. Added lateral features have
zero outgoing weights at initialization. Added step channels start at zero but
have nonzero output weights, allowing their incoming weights to learn.
No previous checkpoint is modified and no AP is attributed to a new shape
until that shape is independently evaluated.
"""
import copy

import torch
from torch import nn

from models import HighResolutionRefiner, MultiScaleRefiner, RSN_PARAMETER_PREFIXES, RTMPoseX


def copy_conv(destination, original, outputs, inputs=None, zero_extra=True):
    if zero_extra:
        destination.weight.zero_()
    if inputs is None:  # depthwise convolution
        destination.weight[outputs] = original.weight
    else:
        destination.weight[outputs[:, None], inputs[None, :]] = original.weight
    if original.bias is not None:
        destination.bias.zero_()
        destination.bias[outputs] = original.bias


def copy_norm(destination, original, indices):
    for name in ('weight', 'bias', 'running_mean', 'running_var'):
        getattr(destination, name)[indices] = getattr(original, name)
    destination.num_batches_tracked.copy_(original.num_batches_tracked)


def copy_steps(source, target):
    old_width = source.project.conv.out_channels//source.num_steps
    width = target.project.conv.out_channels//target.num_steps
    if width < old_width:
        raise ValueError('Expansion must preserve all residual step channels')

    device = source.output.weight.device
    channels = torch.tensor([row*width+col for row in range(source.num_steps)
                             for col in range(old_width)], device=device)
    copy_conv(target.project.conv, source.project.conv, channels,
              torch.arange(source.project.conv.in_channels, device=device))
    copy_norm(target.project.bn, source.project.bn, channels)
    steps = torch.arange(old_width, device=device)
    for key in source.steps:
        copy_conv(target.steps[key].conv, source.steps[key].conv, steps, steps)
        copy_norm(target.steps[key].bn, source.steps[key].bn, steps)
    if source.use_prm:
        for index in (1, 3):
            copy_conv(target.prm_channel[index], source.prm_channel[index], channels, channels)
        copy_conv(target.prm_spatial[0], source.prm_spatial[0], channels)
    copy_conv(target.output, source.output, torch.arange(source.output.out_channels, device=device),
              channels, zero_extra=False)
    return channels


def copy_spatial_attention(source, target, expanded_channels=None):
    if source is None:
        return
    if target is None or target.width != source.width:
        raise ValueError('Expansion must preserve the trained spatial attention width')
    if len(target.context_blocks) < len(source.context_blocks):
        raise ValueError('Expansion cannot remove trained coarse attention layers')
    weights = source.state_dict()
    if expanded_channels is not None:
        for name in ('query.weight', 'key_value.weight'):
            widened = torch.zeros_like(target.get_parameter(name))
            widened.index_copy_(1, expanded_channels, weights[name])
            weights[name] = widened
    new_blocks = tuple(f'context_blocks.{index}.'
                       for index in range(len(source.context_blocks), len(target.context_blocks)))
    incompatible = target.load_state_dict(weights, strict=False)
    assert not incompatible.unexpected_keys
    assert all(name.startswith(new_blocks) for name in incompatible.missing_keys)


def copy_spatial_refiner(source, target):
    width = source.detail.conv.out_channels
    if target is None or target.detail.conv.out_channels < width:
        raise ValueError('Expansion must preserve the trained spatial joint head')
    if len(target.rsn_blocks) < len(source.rsn_blocks):
        raise ValueError('Expansion cannot remove trained spatial RSN blocks')

    expanded_channels = None
    if target.detail.conv.out_channels == width:
        extra_modules = ('attention.', 'rsn_blocks.')
        weights = {name: tensor for name, tensor in source.state_dict().items() if not name.startswith(extra_modules)}
        incompatible = target.load_state_dict(weights, strict=False)
        assert not incompatible.unexpected_keys
        assert all(name.startswith(extra_modules) for name in incompatible.missing_keys)
    else:
        device = source.detail.conv.weight.device
        detail_channels = torch.arange(width, device=device)
        detail_inputs = torch.arange(source.detail.conv.in_channels, device=device)
        copy_conv(target.detail.conv, source.detail.conv, detail_channels, detail_inputs, zero_extra=False)
        copy_norm(target.detail.bn, source.detail.bn, detail_channels)
        nn.init.kaiming_uniform_(target.output.weight, a=5**.5)
        expanded_channels = copy_steps(source, target)
        target.axis_gain.copy_(source.axis_gain)

    copy_spatial_attention(source.attention, target.attention, expanded_channels)
    for old, new in zip(source.rsn_blocks, target.rsn_blocks):
        if new.output.in_channels > old.output.in_channels:
            nn.init.kaiming_uniform_(new.output.weight, a=5**.5)
        channels = copy_steps(old, new)
        copy_spatial_attention(old.attention, new.attention, channels)


@torch.no_grad()
def widen_refiner(source, width, lateral_channels, context_width=0, local_width=0, spatial_width=0,
                  pixel_width=0, spatial_attention_width=0, spatial_attention_layers=1,
                  spatial_rsn_blocks=0, spatial_rsn_width=0, spatial_rsn_attention_layers=1):
    old_lateral = source.lateral.conv.out_channels
    if lateral_channels < old_lateral:
        raise ValueError('Expansion must preserve all lateral channels')
    if isinstance(source, MultiScaleRefiner) and not context_width:
        raise ValueError('Expansion cannot remove the trained context branch')
    if context_width and source.spatial_stride != 8:
        raise ValueError('Widening with context requires matching stride-8 grids')
    target = (MultiScaleRefiner(width, source.num_steps, lateral_channels, context_width,
                               local_width, spatial_width, pixel_width, spatial_attention_width,
                               spatial_attention_layers, spatial_rsn_blocks, spatial_rsn_width,
                               spatial_rsn_attention_layers) if context_width
              else HighResolutionRefiner(width, source.num_steps, lateral_channels, source.spatial_stride))
    target.to(device=source.output.weight.device, dtype=source.output.weight.dtype)
    device = source.output.weight.device
    old_lateral_indices = torch.arange(old_lateral, device=device)
    copy_conv(target.lateral.conv, source.lateral.conv, old_lateral_indices,
              torch.arange(source.lateral.conv.in_channels, device=device), zero_extra=False)
    copy_norm(target.lateral.bn, source.lateral.bn, old_lateral_indices)
    channels = copy_steps(source, target)
    target.token_norm.load_state_dict(source.token_norm.state_dict())
    target.token_projection.load_state_dict(source.token_projection.state_dict())
    if isinstance(source, MultiScaleRefiner):
        nn.init.kaiming_uniform_(target.context_steps.output.weight, a=5**.5)
        copy_steps(source.context_steps, target.context_steps)
        for old, new in ((source.axis_x, target.axis_x), (source.axis_y, target.axis_y)):
            copy_conv(new, old, torch.arange(old.out_channels, device=device), channels)
        target.quality = copy.deepcopy(source.quality)
        if source.local is not None:
            if target.local is None or target.local.offset[0].out_features != source.local.offset[0].out_features:
                raise ValueError('Expansion must preserve the trained local coordinate head')
            target.local.load_state_dict(source.local.state_dict())
        if source.spatial is not None:
            copy_spatial_refiner(source.spatial, target.spatial)
        if source.pixel is not None:
            if target.pixel is None or target.pixel.offset[0].out_features != source.pixel.offset[0].out_features:
                raise ValueError('Expansion must preserve the trained pixel refinement head')
            target.pixel.load_state_dict(source.pixel.state_dict())
    target.train(source.training)
    return target


def expand_model(source, width, lateral_channels, context_width=0, local_width=0, spatial_width=None,
                 pixel_width=None, spatial_attention_width=None, spatial_attention_layers=None,
                 spatial_rsn_blocks=None, spatial_rsn_width=None, spatial_rsn_attention_layers=None):
    spatial_width = source.spatial_width if spatial_width is None else spatial_width
    pixel_width = source.pixel_width if pixel_width is None else pixel_width
    attention_width = source.spatial_attention_width if spatial_attention_width is None else spatial_attention_width
    attention_layers = source.spatial_attention_layers if spatial_attention_layers is None else spatial_attention_layers
    spatial_rsn = {
        name: getattr(source, name) if value is None else value
        for name, value in zip(('spatial_rsn_blocks', 'spatial_rsn_width', 'spatial_rsn_attention_layers'),
                               (spatial_rsn_blocks, spatial_rsn_width, spatial_rsn_attention_layers))
    }
    model = RTMPoseX('rsn_highres', width=width, num_steps=source.head.refiner.num_steps,
                     lateral_channels=lateral_channels, spatial_stride=source.spatial_stride,
                     context_width=context_width, local_width=local_width, trunk_width=source.trunk_width,
                     spatial_width=spatial_width, pixel_width=pixel_width, spatial_attention_width=attention_width,
                     spatial_attention_layers=attention_layers, **spatial_rsn)
    core = {key:value for key,value in source.state_dict().items()
            if not key.startswith('head.refiner.')}
    missing = model.load_state_dict(core, strict=False)
    assert not missing.unexpected_keys
    assert all(key.startswith('head.refiner.') for key in missing.missing_keys)
    model.head.refiner = widen_refiner(source.head.refiner, width, lateral_channels,
                                      context_width, local_width, spatial_width, pixel_width, attention_width,
                                      attention_layers, **spatial_rsn)
    return model.eval()


def capacity_candidate(source, width, lateral_channels, spatial_stride=None, context_width=0, local_width=0,
                       trunk_width=None, fresh_refiner=False, pose_oks_confidence=False, spatial_width=None,
                       pixel_width=None, proposal_confidence=False, proposal_size_confidence=False,
                       spatial_attention_width=None, spatial_attention_layers=None, spatial_rsn_blocks=None,
                       spatial_rsn_width=None, spatial_rsn_attention_layers=None):
    spatial_stride = source.spatial_stride if spatial_stride is None else spatial_stride
    trunk_width = source.trunk_width if trunk_width is None else trunk_width
    spatial_width = source.spatial_width if spatial_width is None else spatial_width
    pixel_width = source.pixel_width if pixel_width is None else pixel_width
    attention_width = source.spatial_attention_width if spatial_attention_width is None else spatial_attention_width
    attention_layers = source.spatial_attention_layers if spatial_attention_layers is None else spatial_attention_layers
    spatial_rsn = {
        name: getattr(source, name) if value is None else value
        for name, value in zip(('spatial_rsn_blocks', 'spatial_rsn_width', 'spatial_rsn_attention_layers'),
                               (spatial_rsn_blocks, spatial_rsn_width, spatial_rsn_attention_layers))
    }
    if not fresh_refiner and spatial_stride == source.spatial_stride and trunk_width == source.trunk_width:
        model = expand_model(source, width, lateral_channels, context_width, local_width, spatial_width,
                             pixel_width, attention_width, attention_layers, **spatial_rsn)
    else:
        # A new grid or differently allocated model starts with zero RSN residuals.
        model = RTMPoseX('rsn_highres', width=width, num_steps=source.head.refiner.num_steps,
                         lateral_channels=lateral_channels, spatial_stride=spatial_stride,
                         context_width=context_width, local_width=local_width, trunk_width=trunk_width,
                         spatial_width=spatial_width, pixel_width=pixel_width, spatial_attention_width=attention_width,
                         spatial_attention_layers=attention_layers, **spatial_rsn)
        core = {key:value for key,value in source.state_dict().items()
                if not key.startswith(RSN_PARAMETER_PREFIXES)}
        missing = model.load_state_dict(core, strict=False)
        assert not missing.unexpected_keys
        assert all(key.startswith(RSN_PARAMETER_PREFIXES) for key in missing.missing_keys)
    if pose_oks_confidence or proposal_confidence or proposal_size_confidence:
        model.head.enable_pose_oks_confidence(proposal_confidence=proposal_confidence or proposal_size_confidence,
                                              proposal_size_confidence=proposal_size_confidence)
    return model.eval()
