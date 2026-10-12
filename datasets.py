"""
Reference:
https://github.com/FedML-AI/FedML
"""
import logging

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

# ================= 新增：全局缓存，解决数百次读取硬盘导致极慢的问题 =================
_cifar_raw_cache = {}
# ====================================================================================


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        global _cifar_raw_cache
        cache_key = (self.root, self.train)

        # 只在第一次真正从硬盘读取 CIFAR10 数据
        if cache_key not in _cifar_raw_cache:
            cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)
            _cifar_raw_cache[cache_key] = (cifar_dataobj.data, np.array(cifar_dataobj.targets))

        data, target = _cifar_raw_cache[cache_key]

        # 针对每个 client 的切片，直接从内存拷贝，微秒级完成
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR100_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        global _cifar_raw_cache
        cache_key = (self.root, 'cifar100', self.train) # 缓存键加上 dataset 名字防止冲突

        if cache_key not in _cifar_raw_cache:
            cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)
            _cifar_raw_cache[cache_key] = (cifar_dataobj.data, np.array(cifar_dataobj.targets))

        data, target = _cifar_raw_cache[cache_key]

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)