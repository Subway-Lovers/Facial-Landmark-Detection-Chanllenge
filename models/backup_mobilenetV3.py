# """
# Creates a MobileNetV3 Model as defined in:
# Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
# Searching for MobileNetV3
# arXiv preprint arXiv:1905.02244.
# """

# import torch.nn as nn
# import math


# __all__ = ['mobilenetv3_large', 'mobilenetv3_small']


# def _make_divisible(v, divisor, min_value=None):
#     """
#     This function is taken from the original tf repo.
#     It ensures that all layers have a channel number that is divisible by 8
#     It can be seen here:
#     https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
#     :param v:
#     :param divisor:
#     :param min_value:
#     :return:
#     """
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v


# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)

#     def forward(self, x):
#         return self.relu(x + 3) / 6


# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)

#     def forward(self, x):
#         return x * self.sigmoid(x)


# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=4):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#                 nn.Linear(channel, _make_divisible(channel // reduction, 8)),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(_make_divisible(channel // reduction, 8), channel),
#                 h_sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y


# def conv_3x3_bn(inp, oup, stride):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#         nn.BatchNorm2d(oup),
#         h_swish()
#     )


# def conv_1x1_bn(inp, oup):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#         nn.BatchNorm2d(oup),
#         h_swish()
#     )


# class InvertedResidual(nn.Module):
#     def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
#         super(InvertedResidual, self).__init__()
#         assert stride in [1, 2]

#         self.identity = stride == 1 and inp == oup

#         if inp == hidden_dim:
#             self.conv = nn.Sequential(
#                 # dw
#                 nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 h_swish() if use_hs else nn.ReLU(inplace=True),
#                 # Squeeze-and-Excite
#                 SELayer(hidden_dim) if use_se else nn.Identity(),
#                 # pw-linear
#                 nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(oup),
#             )
#         else:
#             self.conv = nn.Sequential(
#                 # pw
#                 nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 h_swish() if use_hs else nn.ReLU(inplace=True),
#                 # dw
#                 nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 # Squeeze-and-Excite
#                 SELayer(hidden_dim) if use_se else nn.Identity(),
#                 h_swish() if use_hs else nn.ReLU(inplace=True),
#                 # pw-linear
#                 nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(oup),
#             )

#     def forward(self, x):
#         if self.identity:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)


# class MobileNetV3(nn.Module):
#     def __init__(self, cfgs, mode, num_classes=1000, width_mult=1.):
#         super(MobileNetV3, self).__init__()
#         # setting of inverted residual blocks
#         self.cfgs = cfgs
#         assert mode in ['large', 'small']

#         # building first layer
#         input_channel = _make_divisible(16 * width_mult, 8)
#         layers = [conv_3x3_bn(3, input_channel, 2)]
#         # building inverted residual blocks
#         block = InvertedResidual
#         for k, t, c, use_se, use_hs, s in self.cfgs:
#             output_channel = _make_divisible(c * width_mult, 8)
#             exp_size = _make_divisible(input_channel * t, 8)
#             layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
#             input_channel = output_channel
#         self.features = nn.Sequential(*layers)
#         # building last several layers
#         self.conv = conv_1x1_bn(input_channel, exp_size)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         output_channel = {'large': 1280, 'small': 1024}
#         output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
#         self.classifier = nn.Sequential(
#             nn.Linear(exp_size, output_channel),
#             h_swish(),
#             nn.Dropout(0.2),
#             nn.Linear(output_channel, num_classes),
#         )

#         self._initialize_weights()

#     def forward(self, x):
#         x = self.features(x)
#         x = self.conv(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()


# def mobilenetv3_large(**kwargs):
#     """
#     Constructs a MobileNetV3-Large model
#     """
#     cfgs = [
#         # k, t, c, SE, HS, s 
#         [3,   1,  16, 0, 0, 1],
#         [3,   4,  24, 0, 0, 2],
#         [3,   3,  24, 0, 0, 1],
#         [5,   3,  40, 1, 0, 2],
#         [5,   3,  40, 1, 0, 1],
#         [5,   3,  40, 1, 0, 1],
#         [3,   6,  80, 0, 1, 2],
#         [3, 2.5,  80, 0, 1, 1],
#         [3, 2.3,  80, 0, 1, 1],
#         [3, 2.3,  80, 0, 1, 1],
#         [3,   6, 112, 1, 1, 1],
#         [3,   6, 112, 1, 1, 1],
#         [5,   6, 160, 1, 1, 2],
#         [5,   6, 160, 1, 1, 1],
#         [5,   6, 160, 1, 1, 1]
#     ]
#     return MobileNetV3(cfgs, mode='large', **kwargs)


# def mobilenetv3_small(outnum, **kwargs):
#     """
#     Constructs a MobileNetV3-Small model
#     """
#     cfgs = [
#         # k, t, c, SE, HS, s 
#         [3,    1,  16, 1, 0, 2],
#         [3,  4.5,  24, 0, 0, 2],
#         [3, 3.67,  24, 0, 0, 1],
#         [5,    4,  40, 1, 1, 2],
#         [5,    6,  40, 1, 1, 1],
#         [5,    6,  40, 1, 1, 1],
#         [5,    3,  48, 1, 1, 1],
#         [5,    3,  48, 1, 1, 1],
#         [5,    6,  96, 1, 1, 2],
#         [5,    6,  96, 1, 1, 1],
#         [5,    6,  96, 1, 1, 1],
#     ]

#     return MobileNetV3(cfgs, mode='small', **kwargs, num_classes = outnum)


import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MobileNetV3', 'mobilenetv3']


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, n_class=1000, input_size=224, dropout=0.8, mode='small', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 160
        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', 2],
                [5, 120, 40,  True,  'RE', 1],
                [5, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                [5, 672, 160, True,  'HS', 2],
                [5, 960, 160, True,  'HS', 1],
                [5, 960, 160, True,  'HS', 1],
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', 2],
                # [5, 576, 96,  True,  'HS', 1],
                # [5, 576, 96,  True,  'HS', 1],
                [5, 128, 96,  True,  'HS', 1],
                [5, 128, 96,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError

        # building first layer
        assert input_size % 32 == 0
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]
        self.classifier = []

        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building last several layers
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        elif mode == 'small':
            # last_conv = make_divisible(576 * width_mult)
            last_conv = make_divisible(128 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        else:
            raise NotImplementedError

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),    # refer to paper section 6
            nn.Linear(last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def mobilenetv3(outnum, pretrained=False, **kwargs):
    model = MobileNetV3(n_class = outnum, **kwargs)
    if pretrained:
        state_dict = torch.load('mobilenetv3_small_67.4.pth.tar')
        model.load_state_dict(state_dict, strict=True)
        # raise NotImplementedError
    return model



