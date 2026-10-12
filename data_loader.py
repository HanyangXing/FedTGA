"""
Reference:
https://github.com/FedML-AI/FedML
"""
import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


from datasets import CIFAR10_truncated, CIFAR100_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ===== 新增：全局缓存，确保只从硬盘加载一次 =====
_cifar10_cache = {}

def _get_cifar10_cached(datadir):
    """
    只从硬盘读取一次 CIFAR-10 全量数据，缓存在内存中。
    后续 225 个节点直接从内存按索引切片，零硬盘 IO。
    """
    if datadir not in _cifar10_cache:
        from torchvision.datasets import CIFAR10 as _CIFAR10_raw
        raw = _CIFAR10_raw(datadir, train=True, download=True)
        # raw.data: numpy array (50000, 32, 32, 3), uint8
        # raw.targets: list of int, len=50000
        _cifar10_cache[datadir] = (raw.data, np.array(raw.targets))
        print(f"[Cache] CIFAR-10 train loaded once from disk: {raw.data.shape}")
    return _cifar10_cache[datadir]


# generate the non-IID distribution for all methods
def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'):
    distribution = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0]:
                tmp = x.split(':')
                if '{' == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
    return distribution


def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'):
    net_dataidx_map = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0] and ']' != x[0]:
                tmp = x.split(':')
                if '[' == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(',')
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def load_cifar10_data(datadir):
    train_transform, test_transform = _data_transforms_cifar10()

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=train_transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def _data_transforms_cifar100():
    CIFAR_MEAN = [0.50707516, 0.48654887, 0.44091784]
    CIFAR_STD = [0.26733429, 0.25643846, 0.27615047]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def partition_data_dataset(X_train, y_train, n_nets, alpha):
    min_size = 0
    # 动态获取当前数据集的类别总数 (CIFAR-10 是 10，CIFAR-100 是 100)
    K = len(np.unique(y_train))
    N = y_train.shape[0]
    logging.info("N = " + str(N))
    net_dataidx_map = {}

    while min_size < 10:
        idx_batch = [[] for _ in range(n_nets)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.seed(k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))

            np.random.shuffle(idx_k)

            proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])


    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    return net_dataidx_map

def partition_data(dataset, datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    if partition == "homo":
        total_num = n_train
        test_total_num= n_test
        idxs = np.random.permutation(total_num)
        idxs_test= np.random.permutation(test_total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        batch_idxs_test = np.array_split(idxs_test, n_nets)
        net_dataidx_map_train = {i: batch_idxs[i] for i in range(n_nets)}
        net_dataidx_map_test={i: batch_idxs_test[i] for i in range(n_nets)}

    elif partition == "hetero":
        net_dataidx_map_train=partition_data_dataset(X_train,y_train,n_nets,alpha)
        net_dataidx_map_test=partition_data_dataset(X_test,y_test,n_nets,alpha)
    else:
        raise Exception("partition arg error")

    return X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs_train=None,dataidxs_test=None):
    return get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs_train,dataidxs_test)


def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs_train=None,dataidxs_test=None):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=False, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=False, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def load_partition_data_distributed_cifar10(process_id, dataset, data_dir, partition_method, partition_alpha,
                                            client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map_train, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        dataidxs = net_dataidx_map[process_id - 1]
        local_data_num = len(dataidxs)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 dataidxs)
        logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            process_id, len(train_data_local), len(test_data_local)))
        train_data_global = None
        test_data_global = None
    return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num


def load_partition_data_cifar10(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num_train = len(np.unique(y_train))
    class_num_test = len(np.unique(y_test))
    #logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map_train[r]) for r in range(client_number)])
    test_data_num = sum([len(net_dataidx_map_test[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    #test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict_train = dict()
    data_local_num_dict_test = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs_train = net_dataidx_map_train[client_idx]
        dataidxs_test =  net_dataidx_map_test[client_idx]


        local_data_num_train = len(dataidxs_train)
        local_data_num_test = len(dataidxs_test)


        data_local_num_dict_train[client_idx] = local_data_num_train
        data_local_num_dict_test[client_idx] = local_data_num_test

        logging.info("client_idx = %d, train_local_sample_number = %d" % (client_idx, local_data_num_train))
        logging.info("client_idx = %d, test_local_sample_number = %d" % (client_idx, local_data_num_test))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 dataidxs_train,dataidxs_test)


        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local



    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict_train, data_local_num_dict_test,train_data_local_dict, test_data_local_dict, class_num_train,class_num_test,net_dataidx_map_train


def get_dataloader_for_init(dataset, datadir, batch_size, dataidxs):
    # 根据命令行传入的 dataset 动态决定从哪个缓存读取
    if dataset == 'cifar100':
        all_data, all_targets = _get_cifar100_cached(datadir)
    else:
        all_data, all_targets = _get_cifar10_cached(datadir)

    # 按索引切片
    if dataidxs is not None:
        local_data = all_data[dataidxs]       # (N_local, 32, 32, 3)
        local_targets = all_targets[dataidxs]  # (N_local,)
    else:
        local_data = all_data
        local_targets = all_targets

    # 1. 转为 Tensor 并调换维度
    data_tensor = torch.from_numpy(local_data).permute(0, 3, 1, 2).float() / 255.0

    # 2. 动态使用正确的均值和方差进行标准化
    if dataset == 'cifar100':
        mean = torch.tensor([0.50707516, 0.48654887, 0.44091784]).view(1, 3, 1, 1)
        std = torch.tensor([0.26733429, 0.25643846, 0.27615047]).view(1, 3, 1, 1)
    else:
        mean = torch.tensor([0.49139968, 0.48215827, 0.44653124]).view(1, 3, 1, 1)
        std = torch.tensor([0.24703233, 0.24348505, 0.26158768]).view(1, 3, 1, 1)

    data_tensor = (data_tensor - mean) / std

    # 3. 处理标签
    targets_tensor = torch.from_numpy(local_targets.copy()).long()

    # 4. 直接切分 Batch
    batches = []
    n = len(local_data)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batches.append((data_tensor[start:end], targets_tensor[start:end]))

    return batches



# ====================================================================
# ▼▼▼ 新增：CIFAR-100 专用的数据加载和划分逻辑 ▼▼▼
# ====================================================================

_cifar100_cache = {}

def _get_cifar100_cached(datadir):
    if datadir not in _cifar100_cache:
        from torchvision.datasets import CIFAR100 as _CIFAR100_raw
        raw = _CIFAR100_raw(datadir, train=True, download=True)
        _cifar100_cache[datadir] = (raw.data, np.array(raw.targets))
        print(f"[Cache] CIFAR-100 train loaded once from disk: {raw.data.shape}")
    return _cifar100_cache[datadir]

def _data_transforms_cifar100():
    CIFAR_MEAN = [0.50707516, 0.48654887, 0.44091784]
    CIFAR_STD = [0.26733429, 0.25643846, 0.27615047]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform

def load_cifar100_data(datadir):
    train_transform, test_transform = _data_transforms_cifar100()
    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=train_transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=test_transform)
    return (cifar100_train_ds.data, cifar100_train_ds.target, cifar100_test_ds.data, cifar100_test_ds.target)

def partition_data_cifar100(dataset, datadir, partition, n_nets, alpha):
    logging.info("*********partition data cifar100***************")
    X_train, y_train, X_test, y_test = load_cifar100_data(datadir)

    if partition == "homo":
        idxs = np.random.permutation(X_train.shape[0])
        idxs_test = np.random.permutation(X_test.shape[0])
        batch_idxs = np.array_split(idxs, n_nets)
        batch_idxs_test = np.array_split(idxs_test, n_nets)
        net_dataidx_map_train = {i: batch_idxs[i] for i in range(n_nets)}
        net_dataidx_map_test = {i: batch_idxs_test[i] for i in range(n_nets)}
    elif partition == "hetero":
        net_dataidx_map_train = partition_data_dataset(X_train, y_train, n_nets, alpha)
        net_dataidx_map_test = partition_data_dataset(X_test, y_test, n_nets, alpha)
    else:
        raise Exception("partition arg error")
    return X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test

def get_dataloader_CIFAR100(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    transform_train, transform_test = _data_transforms_cifar100()
    train_ds = CIFAR100_truncated(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = CIFAR100_truncated(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)
    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=False, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)
    return train_dl, test_dl

def load_partition_data_cifar100(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test = partition_data_cifar100(
        dataset, data_dir, partition_method, client_number, partition_alpha)

    class_num_train, class_num_test = len(np.unique(y_train)), len(np.unique(y_test))
    train_data_num = sum([len(net_dataidx_map_train[r]) for r in range(client_number)])
    test_data_num = sum([len(net_dataidx_map_test[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader_CIFAR100(data_dir, batch_size, batch_size)

    data_local_num_dict_train, data_local_num_dict_test = dict(), dict()
    train_data_local_dict, test_data_local_dict = dict(), dict()

    for client_idx in range(client_number):
        dataidxs_train, dataidxs_test = net_dataidx_map_train[client_idx], net_dataidx_map_test[client_idx]
        data_local_num_dict_train[client_idx] = len(dataidxs_train)
        data_local_num_dict_test[client_idx] = len(dataidxs_test)

        train_data_local, test_data_local = get_dataloader_CIFAR100(
            data_dir, batch_size, batch_size, dataidxs_train, dataidxs_test)

        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    return train_data_num, test_data_num, train_data_global, test_data_global, \
        data_local_num_dict_train, data_local_num_dict_test, train_data_local_dict, \
        test_data_local_dict, class_num_train, class_num_test, net_dataidx_map_train