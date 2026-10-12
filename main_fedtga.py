



import logging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
import numpy as np
import torch
import argparse
from fedagg import run_fedagg
from model_zoo import create_model
from data_loader import load_partition_data_cifar10, load_partition_data_cifar100
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed):
    """
    统一设置所有的随机种子，确保实验可重复
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"==> Global random seed set to: {seed}")


def add_args(parser):
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--comm_round', type=int, default=1000, metavar='N',
                        help='maximum communication rounds (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--client_number', type=int, default=225, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--edge_number', type=int, default=15,
                        help='number of edges')
    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')
    parser.add_argument('--partition_alpha', type=float, default=3.0, metavar='PA',
                        help='partition alpha (default: 3.0)')
    parser.add_argument('--method', type=str, default='fedagg',
                        help='method_name')
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')
    parser.add_argument('--class_num', type=int, default=10,
                        help='class_num')
    parser.add_argument("--T_agg", nargs="*", type=float, default=3.0, help="T_agg")

    # 在 add_args 函数中增加以下参数（放在 batch_size 参数之后）
    parser.add_argument('--init_batch_size', type=int, default=128, metavar='N',
                        help='Init阶段encoder编码用的batch_size (default: 128)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    # 采样与温度相关参数
    parser.add_argument('--tau_max', type=float, default=2.0,
                        help='温度调度器最大温度（训练初期），增加多样性')
    parser.add_argument('--tau_min', type=float, default=0.3,
                        help='温度调度器最小温度（训练后期），提高精准度')
    parser.add_argument('--samples_per_class', type=int, default=64,
                        help='每个类别均衡采样的桥接样本数量')

    # ==================== ★ 新增：采样与温度消融实验控制参数 ★ ====================
    parser.add_argument('--sampling_ablation', type=str, default='full',
                        choices=['full', 'no_alloc', 'no_temp', 'none'],
                        help='采样机制消融实验：'
                             'full(自适应分配+温度退火全开); '
                             'no_alloc(仅去掉自适应分配，使用均匀采样); '
                             'no_temp(仅去掉温度调整，使用固定温度); '
                             'none(两者全关)')
    parser.add_argument('--fixed_tau', type=float, default=1.0,
                        help='当关闭温度调整(no_temp 或 none)时，采样所使用的固定温度值(默认1.0，代表标准高斯分布)')
    # ==============================================================================


    # ★ 新增：MOON 风格对比学习相关参数
    parser.add_argument('--mu_con', type=float, default=1.0,
                        help='对比损失权重系数 μ（参考 MOON 论文中的 mu 参数）')
    parser.add_argument('--con_temperature', type=float, default=0.5,
                        help='对比损失温度参数 τ_con（参考 MOON 默认 0.5）')
    parser.add_argument('--out_dim', type=int, default=256,
                        help='投影头输出维度（对比空间维度，参考 MOON 默认 256）')


    # ★ 新增：Oort 风格自适应采样参数
    parser.add_argument('--util_clip_bound', type=float, default=0.95,
                        help='效用裁剪百分位（参考 Oort 的 clip_bound），防止极端类独占预算')
    parser.add_argument('--min_samples_per_class', type=int, default=8,
                        help='每类最少假样本数（保证简单类也有最低覆盖）')
    parser.add_argument('--max_sample_ratio', type=float, default=4.0,
                        help='最大采样倍率（相对均匀分配的上限，防止困难类过度占用）')
    parser.add_argument('--utility_warmup_rounds', type=int, default=10,
                        help='预热轮数：前 N 轮使用均匀分配（积累损失信息）')
    parser.add_argument('--con_warmup', type=int, default=20,
                        help='对比特征对齐机制的预热轮数阈值 Twarm')

    # ========== 新增：消融实验控制参数 ==========
    parser.add_argument('--ablation_mode', type=str, default='standard_moon',
                        choices=['none', 'standard_moon', 'weighted_push', 'hist_reg', 'reg_zprev', 'reg_ema', 'adaptive_gate', 'stop_grad'],
                        help='消融实验模式：standard_moon(默认), weighted_push(加权排斥), hist_reg(历史正则), adaptive_gate(自适应门控)')
    parser.add_argument('--w_neg', type=float, default=0.5,
                        help='weighted_push / adaptive_gate 模式下的负样本排斥力衰减系数')
    parser.add_argument('--w_reg', type=float, default=1.0,
                        help='hist_reg / reg_ema 模式下的历史特征正则化保留系数')
    parser.add_argument('--mu_reg', type=float, default=1.0,
                        help='历史表示作正则时的正则化权重系数')
    parser.add_argument('--ema_momentum', type=float, default=0.99,
                        help='EMA 教师模型的动量/衰减率 (推荐 0.99 或 0.999)')
    # ============================================


    parser.add_argument('--log_name', type=str, default='default_exp', help='自定义日志文件名')
    #parser.add_argument('--tsne_dir', type=str, default='tsne_plots',help='自定义 t-SNE 可视化图片的保存目录')

    args = parser.parse_args()
    args.client_number_per_round = args.client_number
    args.personal_learning_rate = args.lr
    args.client_num_in_total = args.client_number
    return args


def load_data(args, dataset_name):
    if dataset_name == "cifar10":
        data_loader = load_partition_data_cifar10
    elif dataset_name == "cifar100": # <-- 新增
        # 请确保你在顶部导入了 load_partition_data_cifar100
        data_loader = load_partition_data_cifar100
    else:
        raise Exception("dataset not implemented error")

    train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num_train, class_num_test, net_dataidx_map_train = data_loader(
        args.dataset, args.data_dir, args.partition_method,
        args.partition_alpha, args.client_number, args.batch_size)

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict,
               test_data_local_dict, class_num_train, class_num_test, net_dataidx_map_train]
    return dataset


def create_edge_model(args, n_classes, index):
    return create_model("resnet10", out_dim=args.out_dim, num_classes=n_classes)  # ★ 传入 out_dim


def create_client_model(args, n_classes, index):
    return create_model("cnn", out_dim=args.out_dim, num_classes=n_classes) # ★ 传入 out_dim


def create_client_models(args, n_classes):
    client_models = []
    for _ in range(args.client_number):
        client_models.append(create_client_model(args, n_classes, _))
    return client_models


def create_edge_models(args, n_classes):
    edge_models = []
    for _ in range(args.edge_number):
        edge_models.append(create_edge_model(args, n_classes, _))
    return edge_models


def create_cloud_model(args, n_classes):
    return create_model("resnet18", out_dim=args.out_dim, num_classes=n_classes) # ★ 传入 out_dim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)

    # ---------------- 核心修改部分 ----------------
    # 删除了原有的硬编码和含有随机性的种子设置，
    # 替换为调用外部传入的 args.seed (请确保在 add_args 中加入了 --seed)
    if hasattr(args, 'seed'):
        set_random_seed(args.seed)
    else:
        # 如果未传入 seed 参数，设置一个默认的固定种子
        set_random_seed(42)
        # ----------------------------------------------

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict,
     test_data_local_dict, class_num_train, class_num_test,
     net_dataidx_map_train] = dataset

    if args.method == 'fedagg':
        client_models = create_client_models(args, class_num_train)
        edge_models = create_edge_models(args, class_num_train)
        cloud_model = create_cloud_model(args, class_num_train)
        run_fedagg(client_models, edge_models, cloud_model,
                   train_data_local_num_dict, test_data_local_num_dict,
                   train_data_local_dict, test_data_local_dict,
                   test_data_global, args,
                   net_dataidx_map_train=net_dataidx_map_train)
    else:
        raise Exception("method not implemented error")


