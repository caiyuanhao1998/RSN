"""
@author: Yuanhao Cai
@date:  2020.03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from lib.utils.loss import JointsL2Loss


class conv_bn_relu(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, 
            has_bn=True, has_relu=True, efficient=False):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=padding)
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.efficient = efficient
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        def _func_factory(conv, bn, relu, has_bn, has_relu):
            def func(x):
                x = conv(x)
                if has_bn:
                    x = bn(x)
                if has_relu:
                    x = relu(x)
                return x
            return func 

        func = _func_factory(
                self.conv, self.bn, self.relu, self.has_bn, self.has_relu)

        if self.efficient:
            x = checkpoint(func, x)
        else:
            x = func(x)

        return x


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None,
            efficient=False):
        super(Bottleneck, self).__init__()
        self.conv_bn_relu1 = conv_bn_relu(in_planes, planes, kernel_size=1,
                stride=1, padding=0, has_bn=True, has_relu=True,
                efficient=efficient) 
        self.conv_bn_relu2 = conv_bn_relu(planes, planes, kernel_size=3,
                stride=stride, padding=1, has_bn=True, has_relu=True,
                efficient=efficient) 
        self.conv_bn_relu3 = conv_bn_relu(planes, planes * self.expansion,
                kernel_size=1, stride=1, padding=0, has_bn=True,
                has_relu=False, efficient=efficient) 
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv_bn_relu1(x)
        out = self.conv_bn_relu2(out)
        out = self.conv_bn_relu3(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x 
        out = self.relu(out)

        return out


class ResNet_top(nn.Module):

    def __init__(self):
        super(ResNet_top, self).__init__()
        self.conv = conv_bn_relu(3, 64, kernel_size=7, stride=2, padding=3,
                has_bn=True, has_relu=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)

        return x


class ResNet_downsample_module(nn.Module):

    def __init__(self, block, layers, has_skip=False, efficient=False,
            zero_init_residual=False):
        super(ResNet_downsample_module, self).__init__()
        self.has_skip = has_skip 
        self.in_planes = 64
        self.layer1 = self._make_layer(block, 64, layers[0],
                efficient=efficient)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                efficient=efficient)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                efficient=efficient)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                efficient=efficient)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, efficient=False):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = conv_bn_relu(self.in_planes, planes * block.expansion,
                    kernel_size=1, stride=stride, padding=0, has_bn=True,
                    has_relu=False, efficient=efficient)

        layers = list() 
        layers.append(block(self.in_planes, planes, stride, downsample,
            efficient=efficient))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, efficient=efficient))

        return nn.Sequential(*layers)

    def forward(self, x, skip1, skip2):
        x1 = self.layer1(x)
        if self.has_skip:
            x1 = x1 + skip1[0] + skip2[0]
        x2 = self.layer2(x1)
        if self.has_skip:
            x2 = x2 + skip1[1] + skip2[1]
        x3 = self.layer3(x2)
        if self.has_skip:
            x3 = x3 + skip1[2] + skip2[2]
        x4 = self.layer4(x3)
        if self.has_skip:
            x4 = x4 + skip1[3] + skip2[3]

        return x4, x3, x2, x1


class Upsample_unit(nn.Module): 

    def __init__(self, ind, in_planes, up_size, output_chl_num, output_shape,
            chl_num=256, gen_skip=False, gen_cross_conv=False, efficient=False):
        super(Upsample_unit, self).__init__()
        self.output_shape = output_shape

        self.u_skip = conv_bn_relu(in_planes, chl_num, kernel_size=1, stride=1,
                padding=0, has_bn=True, has_relu=False, efficient=efficient)
        self.relu = nn.ReLU(inplace=True)

        self.ind = ind
        if self.ind > 0:
            self.up_size = up_size
            self.up_conv = conv_bn_relu(chl_num, chl_num, kernel_size=1,
                    stride=1, padding=0, has_bn=True, has_relu=False,
                    efficient=efficient)

        self.gen_skip = gen_skip
        if self.gen_skip:
            self.skip1 = conv_bn_relu(in_planes, in_planes, kernel_size=1,
                    stride=1, padding=0, has_bn=True, has_relu=True,
                    efficient=efficient)
            self.skip2 = conv_bn_relu(chl_num, in_planes, kernel_size=1,
                    stride=1, padding=0, has_bn=True, has_relu=True,
                    efficient=efficient)

        self.gen_cross_conv = gen_cross_conv
        if self.ind == 3 and self.gen_cross_conv:
            self.cross_conv = conv_bn_relu(chl_num, 64, kernel_size=1,
                    stride=1, padding=0, has_bn=True, has_relu=True,
                    efficient=efficient)

        self.res_conv1 = conv_bn_relu(chl_num, chl_num, kernel_size=1,
                stride=1, padding=0, has_bn=True, has_relu=True,
                efficient=efficient)
        self.res_conv2 = conv_bn_relu(chl_num, output_chl_num, kernel_size=3,
                stride=1, padding=1, has_bn=True, has_relu=False,
                efficient=efficient)

    def forward(self, x, up_x):
        out = self.u_skip(x)

        if self.ind > 0:
            up_x = F.interpolate(up_x, size=self.up_size, mode='bilinear',
                    align_corners=True)
            up_x = self.up_conv(up_x)
            out += up_x 
        out = self.relu(out)

        res = self.res_conv1(out)
        res = self.res_conv2(res)
        res = F.interpolate(res, size=self.output_shape, mode='bilinear',
                align_corners=True)

        skip1 = None
        skip2 = None
        if self.gen_skip:
            skip1 = self.skip1(x)
            skip2 = self.skip2(out)

        cross_conv = None
        if self.ind == 3 and self.gen_cross_conv:
            cross_conv = self.cross_conv(out)

        return out, res, skip1, skip2, cross_conv


class Upsample_module(nn.Module):

    def __init__(self, output_chl_num, output_shape, chl_num=256,
            gen_skip=False, gen_cross_conv=False, efficient=False):
        super(Upsample_module, self).__init__()
        self.in_planes = [512, 256, 128, 64] 
        h, w = output_shape
        self.up_sizes = [
                (h // 8, w // 8), (h // 4, w // 4), (h // 2, w // 2), (h, w)]
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv

        self.up1 = Upsample_unit(0, self.in_planes[0], self.up_sizes[0],
                output_chl_num=output_chl_num, output_shape=output_shape,
                chl_num=chl_num, gen_skip=self.gen_skip,
                gen_cross_conv=self.gen_cross_conv, efficient=efficient)
        self.up2 = Upsample_unit(1, self.in_planes[1], self.up_sizes[1],
                output_chl_num=output_chl_num, output_shape=output_shape,
                chl_num=chl_num, gen_skip=self.gen_skip,
                gen_cross_conv=self.gen_cross_conv, efficient=efficient)
        self.up3 = Upsample_unit(2, self.in_planes[2], self.up_sizes[2],
                output_chl_num=output_chl_num, output_shape=output_shape,
                chl_num=chl_num, gen_skip=self.gen_skip,
                gen_cross_conv=self.gen_cross_conv, efficient=efficient)
        self.up4 = Upsample_unit(3, self.in_planes[3], self.up_sizes[3],
                output_chl_num=output_chl_num, output_shape=output_shape,
                chl_num=chl_num, gen_skip=self.gen_skip,
                gen_cross_conv=self.gen_cross_conv, efficient=efficient)

    def forward(self, x4, x3, x2, x1):
        out1, res1, skip1_1, skip2_1, _ = self.up1(x4, None)
        out2, res2, skip1_2, skip2_2, _ = self.up2(x3, out1)
        out3, res3, skip1_3, skip2_3, _ = self.up3(x2, out2)
        out4, res4, skip1_4, skip2_4, cross_conv = self.up4(x1, out3)

        # 'res' starts from small size
        res = [res1, res2, res3, res4]
        skip1 = [skip1_4, skip1_3, skip1_2, skip1_1]
        skip2 = [skip2_4, skip2_3, skip2_2, skip2_1]

        return res, skip1, skip2, cross_conv


class Single_stage_module(nn.Module):

    def __init__(self, output_chl_num, output_shape, has_skip=False,
            gen_skip=False, gen_cross_conv=False, chl_num=256, efficient=False,
            zero_init_residual=False,):
        super(Single_stage_module, self).__init__()
        self.has_skip = has_skip
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv
        self.chl_num = chl_num
        self.zero_init_residual = zero_init_residual 
        self.layers = [2, 2, 2, 2]
        self.downsample = ResNet_downsample_module(Bottleneck, self.layers,
                self.has_skip, efficient, self.zero_init_residual)
        self.upsample = Upsample_module(output_chl_num, output_shape,
                self.chl_num, self.gen_skip, self.gen_cross_conv, efficient)

    def forward(self, x, skip1, skip2):
        x4, x3, x2, x1 = self.downsample(x, skip1, skip2)
        res, skip1, skip2, cross_conv = self.upsample(x4, x3, x2, x1)
        
        return res, skip1, skip2, cross_conv


class MSPN(nn.Module):
    
    def __init__(self, cfg, run_efficient=False, **kwargs):
        super(MSPN, self).__init__()
        self.top = ResNet_top()
        self.stage_num = cfg.MODEL.STAGE_NUM
        self.output_chl_num = cfg.DATASET.KEYPOINT.NUM
        self.output_shape = cfg.OUTPUT_SHAPE
        self.upsample_chl_num = cfg.MODEL.UPSAMPLE_CHANNEL_NUM
        self.ohkm = cfg.LOSS.OHKM
        self.topk = cfg.LOSS.TOPK
        self.ctf = cfg.LOSS.COARSE_TO_FINE
        self.mspn_modules = list() 
        for i in range(self.stage_num):
            if i == 0:
                has_skip = False
            else:
                has_skip = True
            if i != self.stage_num - 1:
                gen_skip = True
                gen_cross_conv = True
            else:
                gen_skip = False 
                gen_cross_conv = False 
            self.mspn_modules.append(
                    Single_stage_module(
                        self.output_chl_num, self.output_shape,
                        has_skip=has_skip, gen_skip=gen_skip,
                        gen_cross_conv=gen_cross_conv,
                        chl_num=self.upsample_chl_num,
                        efficient=run_efficient,
                        **kwargs
                        )
                    )
            setattr(self, 'stage%d' % i, self.mspn_modules[i])

    def _calculate_loss(self, outputs, valids, labels):
        # outputs: stg1 -> stg2 -> ... , res1: bottom -> up
        # valids: (n, 17, 1), labels: (n, 5, 17, h, h)
        loss1 = JointsL2Loss()
        if self.ohkm:
            loss2 = JointsL2Loss(has_ohkm=self.ohkm, topk=self.topk)
        
        loss = 0
        for i in range(self.stage_num):
            for j in range(4):
                ind = j
                if i == self.stage_num - 1 and self.ctf:
                    ind += 1 
                tmp_labels = labels[:, ind, :, :, :]

                if j == 3 and self.ohkm:
                    tmp_loss = loss2(outputs[i][j], valids, tmp_labels)
                else:
                    tmp_loss = loss1(outputs[i][j], valids, tmp_labels)

                if j < 3:
                    tmp_loss = tmp_loss / 4

                loss += tmp_loss

        return dict(total_loss=loss)
        
    def forward(self, imgs, valids=None, labels=None):
        x = self.top(imgs)
        skip1 = None
        skip2 = None
        outputs = list()
        for i in range(self.stage_num):
            res, skip1, skip2, x = eval('self.stage' + str(i))(x, skip1, skip2)
            outputs.append(res)

        if valids is None and labels is None:
            return outputs[-1][-1]
        else:
            return self._calculate_loss(outputs, valids, labels)



if __name__ == '__main__':
    from config import cfg
    mspn = MSPN(cfg, run_efficient=cfg.RUN_EFFICIENT)
    imgs = torch.randn(2, 3, 256, 192)
    valids = torch.randn(2, 17, 1)
    labels = torch.randn(2, 5, 17, 64, 48)
    out = mspn(imgs)
    print(out.shape)