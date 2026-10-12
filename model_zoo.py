"""
Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
2. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
3. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun 
Deep Residual Learning for Image Recognition.
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
"""

from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision

#from ..transforms._presets import ImageClassification
#from ..utils import _log_api_usage_once
#from ._api import register_model, Weights, WeightsEnum
#from ._meta import _IMAGENET_CATEGORIES
#from ._utils import _ovewrite_named_param, handle_legacy_interface


__all__ = [
    "ResNet",
    "ResNet18_Weights",
    "ResNet34_Weights",
    "ResNet50_Weights",
    "ResNet101_Weights",
    "ResNet152_Weights",
    "ResNeXt50_32X4D_Weights",
    "ResNeXt101_32X8D_Weights",
    "ResNeXt101_64X4D_Weights",
    "Wide_ResNet50_2_Weights",
    "Wide_ResNet101_2_Weights",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
    "resnet10",
    "Net",
    "ProjectionHead",
    "ModelWithProjection",
    "create_model",
]


# ======================= 投影头（参考 MOON 的 ModelFedCon） =======================
class ProjectionHead(nn.Module):
    """
    轻量级两层 MLP 投影头，将各模型倒数第二层的特征
    统一映射到公共对比空间（out_dim 维）。

    参考 MOON: self.l1 = nn.Linear(num_ftrs, num_ftrs)
               self.l2 = nn.Linear(num_ftrs, out_dim)
    """
    def __init__(self, in_dim, out_dim=256):
        super(ProjectionHead, self).__init__()
        self.l1 = nn.Linear(in_dim, in_dim)
        self.l2 = nn.Linear(in_dim, out_dim)

    def forward(self, h):
        """
        Args:
            h: 深层特征向量，shape (B, in_dim)
        Returns:
            z: 投影后的特征向量，shape (B, out_dim)，已 L2 归一化
        """
        x = F.relu(self.l1(h))
        z = self.l2(x)
        # L2 归一化，使得余弦相似度可以直接用内积计算
        z = F.normalize(z, dim=1)
        return z


# ======================= 带投影头的模型包装器 =======================
class ModelWithProjection(nn.Module):
    """
    将任意 backbone 包装成带投影头的模型。
    参考 MOON 的 ModelFedCon 结构:
      backbone → h (深层特征)
      projection head → z (对比空间特征)
      classifier → y (logits)

    forward 返回三元组: (logits, features, proj_features)
    - logits:        分类输出 (B, num_classes)
    - features:      backbone 倒数第二层特征 (B, feat_dim)（用于其他用途）
    - proj_features: 投影后的对比空间特征 (B, out_dim)（用于对比损失）
    """
    def __init__(self, backbone, feat_dim, out_dim=256):
        """
        Args:
            backbone: 原始模型，其 forward(x) 返回 (logits, features)
            feat_dim: backbone 倒数第二层特征的维度
            out_dim:  投影空间的维度（默认256，与 MOON 一致）
        """
        super(ModelWithProjection, self).__init__()
        self.backbone = backbone
        self.projection_head = ProjectionHead(feat_dim, out_dim)

    def forward(self, x):
        logits, features = self.backbone(x)
        # features 可能是 4D (B, C, H, W)，需要展平
        if features.dim() > 2:
            h = F.adaptive_avg_pool2d(features, (1, 1))
            h = h.view(h.size(0), -1)
        else:
            h = features
        proj = self.projection_head(h)
        return logits, features, proj

    def parameters(self, recurse=True):
        """确保投影头的参数也参与优化"""
        return super().parameters(recurse=recurse)


# ======================= 原有模型定义（保持不变） =======================

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super().__init__()
        #_log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        features = x  # <--- 正确位置：在 Flatten 之后，FC 层之前提取

        x = self.fc(x)

        return x, features

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, progress, **kwargs):
    return ResNet(block, layers, **kwargs)


def resnet18(*, progress=True, **kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2], progress, **kwargs)


def resnet152(*, progress=True, **kwargs):
    return _resnet(Bottleneck, [3, 8, 36, 3], progress, **kwargs)


def resnet10(*, progress=True, **kwargs):
    return _resnet(BasicBlock, [1, 1, 1, 1], progress, **kwargs)

class Net(nn.Module):
    def __init__(self, num_classes=10):  # <--- 新增参数，默认保持10
        super(Net, self).__init__()
        self.feature1=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))#64*16*16

        self.feature2=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),#64*2*2
            nn.Linear(64,48),
            nn.ReLU(inplace=True),
        )

        self.fc=nn.Sequential(
            nn.Linear(48, num_classes) # <--- 修改这里，把 10 替换成 num_classes
        )

    def forward(self, x):
        x = self.feature1(x)
        x = self.feature2(x)

        feature = x

        x = self.fc(x)
        return x, feature


# ======================= 各模型的特征维度映射表 =======================
# 这些维度是各 backbone 在 forward 中返回的 features 经过
# AdaptiveAvgPool2d + flatten 后的维度
_FEAT_DIM_MAP = {
    "cnn": 48,       # Net: feature1 输出通道数=64, 经 avgpool(1,1) 后变 (B,64)
    "resnet10": 512,   # ResNet10: maxpool 后通道数=64
    "resnet18": 512,   # ResNet18: maxpool 后通道数=64
    "resnet152": 2048,  # ResNet152: maxpool 后通道数=64
}


def create_model(model_name, out_dim=256, num_classes=10):
    backbone = None
    if model_name == "resnet18":
        backbone = resnet18(num_classes=num_classes)
    elif model_name == 'resnet10':
        backbone = resnet10(num_classes=num_classes)
    elif model_name == "resnet152":
        backbone = resnet152(num_classes=num_classes)
    elif model_name == "cnn":
        # 修改这里：将 num_classes 传给 Net
        backbone = Net(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    feat_dim = _FEAT_DIM_MAP.get(model_name, 64)
    model = ModelWithProjection(backbone, feat_dim, out_dim)
    return model