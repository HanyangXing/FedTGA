import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.utils import save_image
import torchvision.transforms as transforms
import numpy as np


import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, padding=1), 
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 4, 3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1), #8x4x4
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1), #8x8x8
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ================= 新增：全局缓存参数，防止节点初始化时疯狂读硬盘 =================
_cached_autoencoder_model = None


def create_autoencoder():
    global _cached_autoencoder_model

    if _cached_autoencoder_model is None:
        # 只在第一次真正实例化模型并移动到 GPU
        model = Autoencoder().cuda()

        # 添加 weights_only=True 消除日志中的 FutureWarning
        params = torch.load("params.pkl", weights_only=True)
        model.load_state_dict(params)
        model.eval()

        # 冻结参数：全过程 Autoencoder 只做特征提取和还原，不需要梯度
        # 这可以进一步节省大量的 GPU 显存
        for param in model.parameters():
            param.requires_grad = False

        _cached_autoencoder_model = model

    # 之后的所有节点直接返回已在显存中的同一个模型实例
    return _cached_autoencoder_model