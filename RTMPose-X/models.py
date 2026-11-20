# Copyright (c) OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Pure-PyTorch RTMPose-X port and RSN research extensions.
# Reference: third_party/mmpose at 759b39c13fea6ba094afc1fa932f51dc1b11cbf9.
import math
from contextlib import contextmanager, nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


RSN_PARAMETER_PREFIXES = ('head.refiner.', 'backbone.stage_refiners.')


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, kernel=1, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel, stride, kernel // 2,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout, eps=1e-5, momentum=0.1)
        self.activate = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class DepthwiseSeparable(nn.Module):
    def __init__(self, cin, cout, kernel=5):
        super().__init__()
        self.depthwise_conv = ConvBNAct(cin, cin, kernel, groups=cin)
        self.pointwise_conv = ConvBNAct(cin, cout)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class CSPNeXtBlock(nn.Module):
    def __init__(self, channels, identity=True):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, 3)
        self.conv2 = DepthwiseSeparable(channels, channels, 5)
        self.add_identity = identity

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return out + x if self.add_identity else out


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        return x * self.act(self.fc(self.global_avgpool(x)))


def checkpoint_module(module, *inputs):
    return checkpoint(module, *inputs, use_reentrant=False,
                      context_fn=lambda: (nullcontext(), preserve_bn_buffers(module)))


def checkpoint_in_chunks(module, x, chunk_size):
    """Rematerialize independent examples while preserving the optimizer batch.

    Frozen BN is required: splitting a batch must not change normalization.
    Downstream training BN still receives the complete concatenated batch.
    """
    if any(isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training
           for m in module.modules()):
        raise ValueError('Chunked rematerialization requires frozen BatchNorm')
    return torch.cat([checkpoint_module(module, part) for part in x.split(chunk_size)], dim=0)


class CSPLayer(nn.Module):
    def __init__(self, cin, cout, blocks, identity=True):
        super().__init__()
        mid = cout // 2
        self.checkpoint_blocks = False
        self.main_conv = ConvBNAct(cin, mid)
        self.short_conv = ConvBNAct(cin, mid)
        self.final_conv = ConvBNAct(mid * 2, cout)
        self.blocks = nn.Sequential(*[CSPNeXtBlock(mid, identity) for _ in range(blocks)])
        self.attention = ChannelAttention(mid * 2)

    def forward(self, x):
        efficient = self.checkpoint_blocks and self.training and torch.is_grad_enabled()
        run = checkpoint_module if efficient else lambda module, value: module(value)
        main = run(self.main_conv, x)
        for block in self.blocks:
            main = run(block, main)
        short = run(self.short_conv, x)
        return run(self.final_conv, run(self.attention, torch.cat((main, short), 1)))


class SPPBottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels // 2)
        self.poolings = nn.ModuleList([nn.MaxPool2d(k, stride=1, padding=k // 2)
                                      for k in (5, 9, 13)])
        self.conv2 = ConvBNAct(2 * channels, channels)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [pool(x) for pool in self.poolings], 1))


@contextmanager
def preserve_bn_buffers(module):
    """Use disposable BN buffers during recomputation, preserving saved tensors."""
    buffers = []
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats:
            original = (m.running_mean, m.running_var, m.num_batches_tracked)
            buffers.append((m, original))
            m.running_mean, m.running_var, m.num_batches_tracked = (v.clone() for v in original)
    try:
        yield
    finally:
        for m, original in buffers:
            m.running_mean, m.running_var, m.num_batches_tracked = original


class CSPNeXtX(nn.Module):
    def __init__(self, checkpoint_stages=(), checkpoint_granularity='stage', stem_chunk_size=0, stage1_chunk_size=0,
                 lateral_stage=3, return_context=False, return_fine=False, trunk_width=0):
        super().__init__()
        self.checkpoint_stages = tuple(checkpoint_stages)
        self.checkpoint_granularity = checkpoint_granularity
        self.stem_chunk_size = stem_chunk_size
        self.stage1_chunk_size = stage1_chunk_size
        self.lateral_stage = lateral_stage
        self.return_context = return_context
        self.return_fine = return_fine
        self.stage_refiners = nn.ModuleDict({
            str(stage): ResidualStepsRefiner(trunk_width, num_steps=4,
                                            input_channels=channels, output_channels=channels)
            for stage, channels in ((3, 640), (4, 1280))
        } if trunk_width else {})
        self.stem = nn.Sequential(ConvBNAct(3, 40, 3, 2),
                                  ConvBNAct(40, 40, 3), ConvBNAct(40, 80, 3))
        for i, (cin, cout, blocks, identity, spp) in enumerate([
                (80, 160, 4, True, False), (160, 320, 8, True, False),
                (320, 640, 8, True, False), (640, 1280, 4, False, True)], 1):
            layers = [ConvBNAct(cin, cout, 3, 2)]
            if spp:
                layers.append(SPPBottleneck(cout))
            layers.append(CSPLayer(cout, cout, blocks, identity))
            stage = nn.Sequential(*layers)
            if checkpoint_granularity in {'blocks', 'nested'} and i in self.checkpoint_stages:
                stage[-1].checkpoint_blocks = True
            setattr(self, f'stage{i}', stage)

    def forward(self, x):
        if self.training and self.stem_chunk_size and x.shape[0] > self.stem_chunk_size:
            x = checkpoint_in_chunks(self.stem, x, self.stem_chunk_size)
        elif self.training and 0 in self.checkpoint_stages:
            if self.checkpoint_granularity == 'nested':
                x = checkpoint_module(self.stem, x)
            else:
                for module in self.stem:
                    x = checkpoint_module(module, x)
        else:
            x = self.stem(x)
        lateral, context, fine = None, None, None
        for i in range(1, 5):
            module = getattr(self, f'stage{i}')
            if self.training and i == 1 and self.stage1_chunk_size and x.shape[0] > self.stage1_chunk_size:
                x = checkpoint_in_chunks(module, x, self.stage1_chunk_size)
            elif self.training and i in self.checkpoint_stages:
                if self.checkpoint_granularity == 'blocks':
                    for component in module:
                        x = component(x) if isinstance(component, CSPLayer) else checkpoint_module(component, x)
                else:
                    x = checkpoint_module(module, x)
            else:
                x = module(x)
            if str(i) in self.stage_refiners:
                x = self.stage_refiners[str(i)](x, None)
            if i == self.lateral_stage:
                lateral = x
            if self.return_context and i == 3:
                context = x
            if self.return_fine and i == 1:
                fine = x
        if self.return_context:
            lateral = (lateral, context, fine) if self.return_fine else (lateral, context)
        return lateral, x


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class Scale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.scale


class RTMCCBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.o = nn.Linear(512, 256, bias=False)
        self.uv = nn.Linear(256, 1152, bias=False)
        self.gamma = nn.Parameter(torch.rand(2, 128))
        self.beta = nn.Parameter(torch.rand(2, 128))
        self.ln = ScaleNorm(256)
        self.act_fn = nn.SiLU(inplace=True)
        self.res_scale = Scale(256)

    def forward(self, x):
        u, v, base = self.act_fn(self.uv(self.ln(x))).split((512, 512, 128), dim=2)
        q, k = (base.unsqueeze(2) * self.gamma[None, None] + self.beta).unbind(dim=2)
        kernel = F.relu(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(128)).square()
        return self.res_scale(x) + self.o(u * torch.bmm(kernel, v))


class ResidualStepsRefiner(nn.Module):
    """RSN triangular residual steps plus factorized PRM on low-width maps.

    The triangular cross-step connections follow RSN. Three steps use six
    spatial convolutions; four steps recover the original ten-convolution topology.
    Keeping them off the 1280-channel trunk limits the latency increase.
    A zero output projection exactly preserves the pretrained predictor at init.
    """
    def __init__(self, width=8, lateral=False, prm=True, num_steps=3,
                 input_channels=17, output_channels=17):
        super().__init__()
        self.use_lateral = lateral
        self.use_prm = prm
        assert 2 <= num_steps <= 4
        self.num_steps = num_steps
        self.checkpoint_refinement = False
        if lateral:
            self.lateral = nn.Sequential(ConvBNAct(640, 17), nn.AvgPool2d(2))
        self.project = ConvBNAct(input_channels + (17 if lateral else 0), width * num_steps)
        self.steps = nn.ModuleDict({f'{r}_{c}': ConvBNAct(width, width, 3)
                                    for r in range(num_steps) for c in range(r + 1)})
        if prm:
            channels = width * num_steps
            self.prm_channel = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                             nn.Conv2d(channels, channels, 1),
                                             nn.SiLU(), nn.Conv2d(channels, channels, 1),
                                             nn.Sigmoid())
            self.prm_spatial = nn.Sequential(nn.Conv2d(channels, channels, 9, padding=4,
                                                       groups=channels), nn.Sigmoid())
        self.output = nn.Conv2d(width * num_steps, output_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def refine_features(self, inputs):
        if self.checkpoint_refinement and torch.is_grad_enabled() and (self.training or inputs.requires_grad):
            return checkpoint(self._refine_features, inputs, use_reentrant=False,
                              context_fn=lambda: (nullcontext(), preserve_bn_buffers(self)))
        return self._refine_features(inputs)

    def _refine_features(self, inputs):
        splits = self.project(inputs).chunk(self.num_steps, dim=1)
        outputs = {}
        diagonals = []
        for row in range(self.num_steps):
            for col in range(row + 1):
                if col == 0:
                    x = splits[row]
                else:
                    x = outputs[(row, col - 1)]
                if row > col:
                    x = x + outputs[(row - 1, col)]
                outputs[(row, col)] = self.steps[f'{row}_{col}'](x)
            diagonals.append(outputs[(row, row)])
        x = torch.cat(diagonals, 1)
        if self.use_prm:
            x = x * (1 + self.prm_channel(x) * self.prm_spatial(x))
        return x

    def forward(self, maps, lateral):
        inputs = torch.cat((maps, self.lateral(lateral)), 1) if self.use_lateral else maps
        x = self.refine_features(inputs)
        return maps + self.output(x)


class HighResolutionRefiner(ResidualStepsRefiner):
    """RSN refinement at stride 8 or 16, added as a residual SimCC token branch.

    Retaining the lateral feature's spatial resolution lets this branch recover
    detail discarded by average pooling. Only the final token projection starts
    at zero: zeroing both projections would prevent either from learning.
    """
    def __init__(self, width=16, num_steps=3, lateral_channels=17, spatial_stride=16):
        super().__init__(width, lateral=True, num_steps=num_steps)
        if spatial_stride not in (8, 16):
            raise ValueError('RSN spatial stride must be 8 or 16')
        self.spatial_stride = spatial_stride
        self.lateral = ConvBNAct(320 if spatial_stride == 8 else 640, lateral_channels)
        if lateral_channels != 17:
            self.project = ConvBNAct(17+lateral_channels, width*num_steps)
        nn.init.kaiming_uniform_(self.output.weight, a=math.sqrt(5))
        nn.init.zeros_(self.output.bias)
        locations = (384//spatial_stride)*(288//spatial_stride)
        self.token_norm = ScaleNorm(locations)
        self.token_projection = nn.Linear(locations, 256, bias=False)
        nn.init.zeros_(self.token_projection.weight)

    def forward(self, maps, lateral, return_maps=False):
        coarse = F.interpolate(maps, size=lateral.shape[-2:], mode='bilinear', align_corners=False)
        inputs = torch.cat((coarse, self.lateral(lateral)), dim=1)
        fine_maps = self.output(self.refine_features(inputs))
        tokens = self.token_projection(self.token_norm(fine_maps.flatten(2)))
        return (tokens, fine_maps) if return_maps else tokens


class LocalJointRefiner(nn.Module):
    """Predict bounded offsets from patches around the current joints."""

    def __init__(self, width, feature_channels=160, feature_stride=4, radius=1,
                 sample_spacing=4., max_shift=6.):
        super().__init__()
        self.feature_stride = feature_stride
        self.max_shift = max_shift
        offsets = torch.arange(-radius, radius+1, dtype=torch.float32) * sample_spacing
        y, x = torch.meshgrid(offsets, offsets, indexing='ij')
        self.register_buffer('window', torch.stack((x, y), -1).reshape(-1, 2), persistent=False)
        self.offset = nn.Sequential(nn.Linear(feature_channels*len(self.window) + 256, width),
                                    nn.LayerNorm(width), nn.SiLU(),
                                    nn.Linear(width, 2))
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)

    @staticmethod
    def translate(logits, offset, bins=None):
        if bins is None:
            bins = torch.arange(logits.shape[-1], device=logits.device, dtype=torch.float32)
        sample = torch.add(bins, offset.float().unsqueeze(-1), alpha=-2).clamp_(0, logits.shape[-1]-1)
        lower = sample.long()
        upper = (lower + 1).clamp_max_(logits.shape[-1]-1)
        fraction = (sample - lower).to(logits.dtype)
        shifted = torch.lerp(logits.gather(-1, lower), logits.gather(-1, upper), fraction)
        # Preserve the SimCC confidence scale while adjusting the coordinates.
        return shifted + (logits.amax(-1, keepdim=True) - shifted.amax(-1, keepdim=True))

    @staticmethod
    def anchors(prediction):
        return torch.stack(tuple(axis.argmax(-1) for axis in prediction), -1).float() * .5

    def predict_offset(self, features, tokens, anchor):
        height, width = features.shape[-2:]
        x, y = ((anchor.unsqueeze(-2) + self.window) / self.feature_stride + .5).unbind(-1)
        grid = torch.stack((x * (2 / width) - 1, y * (2 / height) - 1), -1).to(features.dtype)
        patches = F.grid_sample(features, grid, align_corners=False, padding_mode='border')
        patches = patches.permute(0, 2, 1, 3).flatten(2)
        return self.max_shift * self.offset(torch.cat((patches, tokens.detach()), -1)).tanh()

    def forward(self, features, tokens, prediction):
        anchor = self.anchors(prediction)
        offset = self.predict_offset(features, tokens, anchor)
        shifted = tuple(self.translate(axis, offset[..., i]) for i, axis in enumerate(prediction))
        return shifted, anchor, offset


class PoseOKSConfidence(nn.Module):
    """Estimate confidence for an entire person crop from tokens and SimCC peaks."""

    def __init__(self, proposal_confidence=False, proposal_size_confidence=False):
        super().__init__()
        self.proposal_confidence = proposal_confidence
        self.token_norm = nn.LayerNorm(256)
        self.projection = nn.Sequential(nn.Linear(260, 64), nn.SiLU(), nn.Linear(64, 1))
        self.size_context = nn.Linear(2, 64, bias=False) if proposal_size_confidence else None
        if self.size_context is not None:
            nn.init.zeros_(self.size_context.weight)
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.constant_(self.projection[-1].bias, 3. if proposal_confidence else 0.)

    def forward(self, tokens, prediction, proposal_size=None):
        statistics = []
        for axis in prediction:
            axis = axis.detach()
            peak = axis.amax(dim=-1)
            statistics.extend((peak, peak - axis.mean(dim=-1)))
        evidence = torch.stack(statistics, dim=-1).to(tokens.dtype)
        inputs = torch.cat((self.token_norm(tokens.detach()), evidence), dim=-1)
        hidden = self.projection[0](inputs)
        if self.size_context is not None:
            if proposal_size is None:
                raise ValueError('The confidence head requires the proposal size')
            hidden = hidden + self.size_context(proposal_size.to(hidden.dtype)).unsqueeze(1)
        correction = self.projection[2](self.projection[1](hidden)).mean(dim=1)
        if self.proposal_confidence:
            return correction
        return .3 * correction.tanh()


def attend_spatial_tokens(query, key, value, heads):
    def split(tokens):
        return tokens.unflatten(-1, (heads, -1)).transpose(1, 2)

    attended = F.scaled_dot_product_attention(split(query), split(key), split(value))
    return attended.transpose(1, 2).flatten(2)


class RSNCoarseAttention(nn.Module):
    """Refine the pooled RSN context before attending from the finer grid."""

    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, 2 * width, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, keys, values):
        query, key, value = self.qkv(self.norm(keys + values)).chunk(3, dim=-1)
        key_delta, value_delta = self.output(attend_spatial_tokens(query, key, value, self.heads)).chunk(2, dim=-1)
        return keys + key_delta, values + value_delta


class RSNSpatialAttention(nn.Module):
    """Attend across a reduced spatial grid of RSN residual-step features."""

    heads = 4
    query_reduction = 2
    key_value_reduction = 8

    def __init__(self, channels, width, layers=1):
        super().__init__()
        if width <= 0 or width % self.heads:
            raise ValueError('Spatial attention width must be a positive multiple of the head count')
        if layers < 1:
            raise ValueError('Spatial attention requires at least one layer')
        self.width = width
        self.query = nn.Linear(channels, width, bias=False)
        self.key_value = nn.Linear(channels, 2 * width, bias=False)
        self.query_norm = nn.LayerNorm(width)
        self.key_norm = nn.LayerNorm(width)
        self.context_blocks = nn.ModuleList(RSNCoarseAttention(width, self.heads) for _ in range(layers - 1))
        self.output = nn.Linear(width, 17, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, features):
        query_grid = F.avg_pool2d(features, self.query_reduction)
        key_value_grid = F.avg_pool2d(query_grid, self.key_value_reduction // self.query_reduction)
        queries = self.query(query_grid.flatten(2).transpose(1, 2))
        keys, values = self.key_value(key_value_grid.flatten(2).transpose(1, 2)).chunk(2, dim=-1)
        for block in self.context_blocks:
            keys, values = block(keys, values)
        attended = attend_spatial_tokens(self.query_norm(queries), self.key_norm(keys), values, self.heads)
        heatmaps = self.output(attended).transpose(1, 2)
        heatmaps = heatmaps.unflatten(-1, query_grid.shape[-2:])
        return F.interpolate(heatmaps, size=features.shape[-2:], mode='bilinear', align_corners=False)


class SpatialRSNHeatmapBlock(ResidualStepsRefiner):
    """Add a learned spatial residual to the preceding joint heatmaps."""

    def __init__(self, width, attention_width, attention_layers):
        super().__init__(width, prm=False, num_steps=3, input_channels=160 + 17)
        self.attention = RSNSpatialAttention(self.num_steps * width, attention_width, attention_layers)

    def forward(self, detail, heatmaps):
        features = self._refine_features(torch.cat((heatmaps, detail), dim=1))
        return heatmaps + self.output(features) + self.attention(features)


class SpatialJointRefiner(ResidualStepsRefiner):
    """Use a stride-four joint heatmap to refine the two SimCC marginals."""

    logit_scale = 10.

    def __init__(self, width, attention_width=0, attention_layers=1, rsn_blocks=0, rsn_width=0,
                 rsn_attention_layers=1):
        super().__init__(width, prm=False, num_steps=2, input_channels=width + 17)
        self.checkpoint_chunk_size = 0
        self.detail = ConvBNAct(160, width)
        self.axis_gain = nn.Parameter(torch.zeros(2, 17, 1))
        self.attention = RSNSpatialAttention(2 * width, attention_width, attention_layers) if attention_width else None
        self.rsn_blocks = nn.ModuleList(SpatialRSNHeatmapBlock(rsn_width, attention_width, rsn_attention_layers)
                                        for _ in range(rsn_blocks))

    def forward(self, detail, coarse, prediction):
        if not self.checkpoint_chunk_size or not torch.is_grad_enabled():
            return self.refine_pose(detail, coarse, prediction)
        chunks = (tensor.split(self.checkpoint_chunk_size) for tensor in (detail, coarse, *prediction))
        parts = [
            checkpoint(self.refine_pose, detail_part, coarse_part, (x, y), False, use_reentrant=False)
            for detail_part, coarse_part, x, y in zip(*chunks)
        ]
        predictions, heatmaps = zip(*parts)
        return tuple(torch.cat(axis) for axis in zip(*predictions)), torch.cat(heatmaps)

    def refine_pose(self, detail, coarse, prediction, checkpoint_features=True):
        height, width = detail.shape[-2:]
        coarse = F.interpolate(F.pad(coarse, (0, 1, 0, 1), mode='replicate'),
                               (height + 1, width + 1), mode='bilinear', align_corners=True)
        coarse = coarse[..., :height, :width]
        refine_features = self.refine_features if checkpoint_features else self._refine_features
        features = refine_features(torch.cat((coarse, self.detail(detail)), dim=1))
        heatmaps = coarse + self.output(features)
        if self.attention is not None:
            heatmaps = heatmaps + self.attention(features)
        for block in self.rsn_blocks:
            if checkpoint_features and self.checkpoint_refinement and torch.is_grad_enabled():
                heatmaps = checkpoint_module(block, detail, heatmaps)
            else:
                heatmaps = block(detail, heatmaps)
        spatial_logits = heatmaps.float() * self.logit_scale
        # A shared per-joint shift cancels when each marginal is centered below.
        probability = (spatial_logits - spatial_logits.amax((-2, -1), keepdim=True)).exp()
        minimum = torch.finfo(probability.dtype).tiny
        marginals = (probability.sum(-2).clamp_min(minimum).log(),
                     probability.sum(-1).clamp_min(minimum).log())
        refined = []
        for values, logits, gain in zip(marginals, prediction, self.axis_gain.tanh()):
            # Include the last cell boundary so interpolation preserves the trained lattice.
            values = F.interpolate(F.pad(values, (0, 1), mode='replicate'),
                                   size=logits.shape[-1] + 1, mode='linear', align_corners=True)[..., :-1]
            centered = (values - values.amax(-1, keepdim=True)) / self.logit_scale
            adjusted = logits + (centered * gain).to(logits.dtype)
            refined.append(adjusted + (logits.amax(-1, keepdim=True) - adjusted.amax(-1, keepdim=True)))
        return tuple(refined), heatmaps


class MultiScaleRefiner(HighResolutionRefiner):
    """Refine deeper stride-16 context and predict local SimCC corrections at stride 8."""
    def __init__(self, width, num_steps, lateral_channels, context_width, local_width=0, spatial_width=0,
                 pixel_width=0, spatial_attention_width=0, spatial_attention_layers=1,
                 spatial_rsn_blocks=0, spatial_rsn_width=0, spatial_rsn_attention_layers=1):
        super().__init__(width, num_steps, lateral_channels, spatial_stride=8)
        self.context_steps = ResidualStepsRefiner(context_width, num_steps=num_steps,
                                                  input_channels=640, output_channels=lateral_channels)
        channels = width * num_steps
        self.bins_per_cell = 2 * self.spatial_stride
        self.axis_x = nn.Conv1d(channels, 17*self.bins_per_cell, 3, padding=1, bias=False)
        self.axis_y = nn.Conv1d(channels, 17*self.bins_per_cell, 3, padding=1, bias=False)
        nn.init.zeros_(self.axis_x.weight)
        nn.init.zeros_(self.axis_y.weight)
        self.quality = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.quality[-1].weight)
        nn.init.zeros_(self.quality[-1].bias)
        self.local = LocalJointRefiner(local_width) if local_width else None
        self.spatial = (SpatialJointRefiner(spatial_width, spatial_attention_width, spatial_attention_layers,
                                            spatial_rsn_blocks, spatial_rsn_width, spatial_rsn_attention_layers)
                        if spatial_width else None)
        self.pixel = (LocalJointRefiner(pixel_width, feature_channels=3, feature_stride=1, radius=3,
                                        sample_spacing=1.5, max_shift=4.) if pixel_width else None)

    def forward(self, maps, laterals):
        lateral, context = laterals
        details = self.lateral(lateral)
        self.context_steps.checkpoint_refinement = self.checkpoint_refinement
        context_features = self.context_steps.refine_features(context)
        context_map = self.context_steps.output(context_features)
        details = details + F.interpolate(context_map, size=details.shape[-2:], mode='bilinear', align_corners=False)
        coarse = F.interpolate(maps, size=details.shape[-2:], mode='bilinear', align_corners=False)
        features = self.refine_features(torch.cat((coarse, details), 1))
        fine_maps = self.output(features)
        tokens = self.token_projection(self.token_norm(fine_maps.flatten(2)))
        axes = (self.axis_x(features.mean(dim=2)), self.axis_y(features.mean(dim=3)))
        # Each grid cell predicts contiguous sub-cell bins on its own axis.
        corrections = tuple(axis.unflatten(1, (17, self.bins_per_cell)).transpose(2, 3).flatten(2)
                            for axis in axes)
        return tokens, fine_maps, corrections


class RTMCCHead(nn.Module):
    def __init__(self, variant='baseline', width=8, num_steps=3, lateral_channels=17,
                 spatial_stride=16, context_width=0, local_width=0, spatial_width=0, pixel_width=0,
                 spatial_attention_width=0, spatial_attention_layers=1, spatial_rsn_blocks=0,
                 spatial_rsn_width=0, spatial_rsn_attention_layers=1):
        super().__init__()
        self.final_layer = nn.Conv2d(1280, 17, 7, padding=3)
        self.mlp = nn.Sequential(ScaleNorm(108), nn.Linear(108, 256, bias=False))
        self.gau = RTMCCBlock()
        self.cls_x = nn.Linear(256, 576, bias=False)
        self.cls_y = nn.Linear(256, 768, bias=False)
        self.variant = variant
        self.context_width = context_width
        self.local_width = local_width
        self.spatial_width = spatial_width
        self.pixel_width = pixel_width
        if local_width:
            self.register_buffer('simcc_x_bins', torch.arange(576), persistent=False)
            self.register_buffer('simcc_y_bins', torch.arange(768), persistent=False)
        self.auxiliary_heatmaps = False
        self.raw_confidence_output = False
        if context_width:
            self.refiner = MultiScaleRefiner(width, num_steps, lateral_channels, context_width,
                                             local_width, spatial_width, pixel_width,
                                             spatial_attention_width, spatial_attention_layers, spatial_rsn_blocks,
                                             spatial_rsn_width, spatial_rsn_attention_layers)
        elif variant == 'rsn_highres':
            self.refiner = HighResolutionRefiner(width, num_steps=num_steps,
                                                lateral_channels=lateral_channels, spatial_stride=spatial_stride)
        elif variant != 'baseline':
            self.refiner = ResidualStepsRefiner(width, lateral=variant == 'rsn_fusion', num_steps=num_steps)

    def enable_pose_oks_confidence(self, proposal_confidence=False, proposal_size_confidence=False):
        if not self.context_width:
            raise ValueError('Pose quality supervision requires the multiscale RSN head')
        quality = self.refiner.quality
        same_mode = isinstance(quality, PoseOKSConfidence) and quality.proposal_confidence == proposal_confidence
        if same_mode and (quality.size_context is not None) == proposal_size_confidence:
            return
        capacity = PoseOKSConfidence(proposal_confidence, proposal_size_confidence).to(self.cls_x.weight)
        if same_mode:
            capacity.load_state_dict({name: value for name, value in quality.state_dict().items()
                                      if not name.startswith('size_context.')}, strict=False)
        self.refiner.quality = capacity

    def forward(self, feats, images=None, proposal_size=None):
        lateral, x = feats
        maps = self.final_layer(x)
        if self.variant in {'rsn', 'rsn_fusion'}:
            maps = self.refiner(maps, lateral)
        tokens = self.mlp(maps.flatten(2))
        auxiliary = None
        corrections = None
        fine = None
        if self.local_width or self.spatial_width:
            lateral, fine = lateral[:2], lateral[2]
        if self.context_width:
            residual, fine_maps, corrections = self.refiner(maps, lateral)
            tokens = tokens + residual
            if self.training and self.auxiliary_heatmaps:
                auxiliary = fine_maps
        elif self.variant == 'rsn_highres':
            if self.training and self.auxiliary_heatmaps:
                residual, auxiliary = self.refiner(maps, lateral, return_maps=True)
                tokens = tokens + residual
            else:
                tokens = tokens + self.refiner(maps, lateral)
        tokens = self.gau(tokens)
        prediction = (self.cls_x(tokens), self.cls_y(tokens))
        score_delta = None
        local_targets = None
        pixel_targets = None
        if corrections is not None:
            prediction = tuple(base + correction for base, correction in zip(prediction, corrections))
        if self.local_width:
            local = self.refiner.local
            anchor = local.anchors(prediction)
            offset = local.predict_offset(fine, tokens, anchor)
            combined = offset
            if self.pixel_width:
                pixel_anchor = (anchor + offset.float()).detach()
                pixel_offset = self.refiner.pixel.predict_offset(images, tokens, pixel_anchor)
                combined = offset + pixel_offset
                if self.training:
                    pixel_targets = (pixel_anchor, pixel_offset)
            bins = self.simcc_x_bins, self.simcc_y_bins
            prediction = tuple(local.translate(axis, combined[..., i], bins[i]) for i, axis in enumerate(prediction))
            if self.training:
                local_targets = (anchor, offset)
        if self.spatial_width:
            spatial = self.refiner.spatial
            spatial.checkpoint_refinement = self.refiner.checkpoint_refinement
            prediction, spatial_heatmaps = spatial(fine, fine_maps, prediction)
            if self.training and self.auxiliary_heatmaps:
                auxiliary = spatial_heatmaps
        if corrections is not None and (self.training or not self.raw_confidence_output):
            quality = self.refiner.quality
            if isinstance(quality, PoseOKSConfidence):
                score_delta = quality(tokens, prediction, proposal_size)
                if quality.proposal_confidence and not self.training:
                    # Preserve the ranking of confident proposals.
                    score_delta = (F.logsigmoid(score_delta.float()) + math.log(2.)).clamp_max_(0.)
            else:
                score_delta = quality(tokens.detach()).squeeze(-1).tanh()
        if self.training and (auxiliary is not None or score_delta is not None):
            result = {'prediction': prediction}
            if auxiliary is not None:
                result['auxiliary_heatmaps'] = auxiliary
            if score_delta is not None:
                result['score_delta'] = score_delta
            if local_targets is not None:
                result['local_refinement'] = local_targets
            if pixel_targets is not None:
                result['pixel_refinement'] = pixel_targets
            return result
        if score_delta is not None:
            return (*prediction, score_delta)
        return prediction


class RTMPoseX(nn.Module):
    def __init__(self, variant='baseline', width=8, checkpoint_stages=(), num_steps=3,
                 checkpoint_granularity='stage', stem_chunk_size=0, stage1_chunk_size=0,
                 lateral_channels=17, spatial_stride=16, context_width=0, local_width=0,
                 trunk_width=0, spatial_width=0, pixel_width=0, spatial_attention_width=0,
                 spatial_attention_layers=1, spatial_rsn_blocks=0, spatial_rsn_width=0,
                 spatial_rsn_attention_layers=1):
        super().__init__()
        if variant not in {'baseline', 'rsn', 'rsn_fusion', 'rsn_highres'}:
            raise ValueError(variant)
        if spatial_stride not in (8, 16) or (spatial_stride == 8 and variant != 'rsn_highres'):
            raise ValueError('Stride 8 is supported only by the high-resolution RSN branch')
        if context_width < 0 or (context_width and spatial_stride != 8):
            raise ValueError('Multiscale context requires the stride-8 high-resolution RSN branch')
        if local_width < 0 or (local_width and not context_width):
            raise ValueError('Local joint refinement requires multiscale context')
        if trunk_width < 0 or (trunk_width and variant != 'rsn_highres'):
            raise ValueError('Backbone RSN blocks require the high-resolution variant')
        if spatial_width < 0 or (spatial_width and not context_width):
            raise ValueError('Spatial joint refinement requires multiscale context')
        if spatial_attention_width < 0 or (spatial_attention_width and not spatial_width):
            raise ValueError('Spatial attention requires the spatial joint refiner')
        if spatial_attention_layers < 1 or (spatial_attention_layers > 1 and not spatial_attention_width):
            raise ValueError('Additional attention layers require an enabled spatial attention branch')
        if (spatial_rsn_blocks < 0 or spatial_rsn_width < 0 or spatial_rsn_attention_layers < 1 or
                bool(spatial_rsn_blocks) != bool(spatial_rsn_width) or
                (spatial_rsn_blocks and not spatial_attention_width)):
            raise ValueError('Spatial RSN blocks require a positive width and spatial attention')
        if pixel_width < 0 or (pixel_width and not local_width):
            raise ValueError('Pixel refinement builds on the stride-four local offsets')
        self.spatial_stride = spatial_stride
        self.context_width = context_width
        self.local_width = local_width
        self.trunk_width = trunk_width
        self.spatial_width = spatial_width
        self.spatial_attention_width = spatial_attention_width
        self.spatial_attention_layers = spatial_attention_layers
        self.spatial_rsn_blocks = spatial_rsn_blocks
        self.spatial_rsn_width = spatial_rsn_width
        self.spatial_rsn_attention_layers = spatial_rsn_attention_layers
        self.pixel_width = pixel_width
        self.backbone = CSPNeXtX(checkpoint_stages, checkpoint_granularity, stem_chunk_size, stage1_chunk_size,
                                lateral_stage=2 if spatial_stride == 8 else 3, return_context=bool(context_width),
                                return_fine=bool(local_width or spatial_width), trunk_width=trunk_width)
        self.head = RTMCCHead(variant, width, num_steps, lateral_channels, spatial_stride, context_width,
                              local_width, spatial_width, pixel_width, spatial_attention_width, spatial_attention_layers,
                              spatial_rsn_blocks, spatial_rsn_width, spatial_rsn_attention_layers)
        self.variant = variant

    @property
    def uses_proposal_size(self):
        quality = getattr(getattr(self.head, 'refiner', None), 'quality', None)
        return isinstance(quality, PoseOKSConfidence) and quality.size_context is not None

    def forward(self, images, proposal_size=None):
        return self.head(self.backbone(images), images if self.pixel_width else None, proposal_size)

    def load_official(self, path):
        loaded = torch.load(path, map_location='cpu', weights_only=True)
        state = loaded.get('state_dict', loaded.get('model', loaded))
        incompatible = self.load_state_dict(state, strict=False)
        assert not incompatible.unexpected_keys, incompatible.unexpected_keys
        assert all(k.startswith(RSN_PARAMETER_PREFIXES) for k in incompatible.missing_keys), incompatible.missing_keys
        return incompatible
