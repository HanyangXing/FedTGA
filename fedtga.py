import copy
import numpy as np
import os
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.utils import save_image
import torchvision.transforms as transforms
import numpy as np
from autoencoder_pretrained import create_autoencoder
from utils import KL_Loss, CE_Loss
from torch import nn
import torch.nn.functional as F
import utils
import math
from collections import defaultdict

# ===== 新增导入用于 t-SNE ======
import matplotlib.pyplot as plt
# ==============================

# ===== 新增导入用于 t-SNE 和 Silhouette ======
#from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
# ==============================

# ======================= 通信开销追踪器 =======================
class CommunicationTracker:
    def __init__(self):
        # 第0轮：统计参数的传输量 (Bytes)
        self.round_0_client_edge_bytes = 0
        self.round_0_edge_cloud_bytes = 0
        # 后续轮次：每轮的模型交互量 (Bytes)
        self.current_round_client_edge_bytes = 0
        self.current_round_edge_cloud_bytes = 0

    def reset_round(self):
        self.current_round_client_edge_bytes = 0
        self.current_round_edge_cloud_bytes = 0

comm_tracker = CommunicationTracker()


def test_on_cloud(cloud_model, test_data_global, comm_round, args=None):
    acc_all = []
    if True:
        loss_avg = utils.RunningAverage()
        accTop1_avg = utils.RunningAverage()
        accTop5_avg = utils.RunningAverage()
        for batch_idx, (images, labels) in enumerate(test_data_global):
            images, labels = images.cuda(), labels.cuda()
            labels = torch.tensor(labels, dtype=torch.long)
            # ★ 修改：模型现在返回三元组，只取 logits
            log_probs, _, _ = cloud_model(images)
            metrics = utils.accuracy(log_probs, labels, topk=(1, 5))
            accTop1_avg.update(metrics[0].item())
            accTop5_avg.update(metrics[1].item())
        test_metrics = {
            'test_accTop1': accTop1_avg.value(),
            'test_accTop5': accTop5_avg.value(),
        }

        # ======================= 修改开始 =======================
        # 1. 严格按照要求的格式生成字符串
        output_str = f"Test/AccTop1 in comm_round {comm_round} {test_metrics['test_accTop1']}"

        # 2. 依然在终端打印
        print(output_str)

        # 3. 追加保存到 txt/log 文件中
        if args is not None:
            # =========================================================
            # ▼▼▼ 方便修改的地方：在这里自定义你的日志文件前缀名 ▼▼▼
            # 默认这里使用 数据集名字 + alpha 参数来区分不同实验
            custom_name = f"experiment_{args.dataset}_alpha{args.partition_alpha}"

            # 最终的完整文件名 (加上了 .log 后缀)
            log_filename = f"{args.log_name}.log"
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            with open(log_filename, "a") as f:
                f.write(output_str + "\n")




# ======================= Macro-F1 评估模块 =======================
def calculate_macro_f1(model, test_loader, num_classes):
    """
    计算给定模型在测试集上的 Macro-F1 score
    """
    model.eval()
    TP = [0.0] * num_classes
    FP = [0.0] * num_classes
    FN = [0.0] * num_classes

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            labels = labels.long()

            # 模型返回三元组，取第一个元素 logits
            log_probs, _, _ = model(images)
            _, preds = torch.max(log_probs, 1)

            # 统计各个类别的 TP, FP, FN
            for c in range(num_classes):
                TP[c] += ((preds == c) & (labels == c)).sum().item()
                FP[c] += ((preds == c) & (labels != c)).sum().item()
                FN[c] += ((preds != c) & (labels == c)).sum().item()

    macro_f1 = 0.0
    for c in range(num_classes):
        # 计算精确率和召回率（防止分母为0）
        precision = TP[c] / (TP[c] + FP[c]) if (TP[c] + FP[c]) > 0 else 0.0
        recall = TP[c] / (TP[c] + FN[c]) if (TP[c] + FN[c]) > 0 else 0.0

        # 计算单类 F1 分数
        if precision + recall > 0:
            f1_c = 2 * (precision * recall) / (precision + recall)
        else:
            f1_c = 0.0

        macro_f1 += f1_c

    # 所有类别的 F1 求无权算术平均
    macro_f1 /= num_classes
    return macro_f1


def get_all_leaf_models(node):
    """
    递归获取所有的叶子节点（客户端）模型
    """
    leaves = []
    if node.is_leaf():
        leaves.append((node.index, node.model))
    else:
        for child in node.children:
            leaves.extend(get_all_leaf_models(child))
    return leaves

def get_all_leaf_nodes(node):
    """
    递归获取所有的叶子节点对象（包含统计信息等）
    """
    leaves = []
    if node.is_leaf():
        leaves.append(node)
    else:
        for child in node.children:
            leaves.extend(get_all_leaf_nodes(child))
    return leaves


# def test_and_log_macro_f1_cloud(cloud_model, test_data_global, comm_round, args):
#     """
#     测试并记录云端模型 (Cloud Model) 的 Macro-F1
#     """
#     num_classes = args.class_num
#     f1 = calculate_macro_f1(cloud_model, test_data_global, num_classes)
#
#     output_str = f"\n{'='*60}\n"
#     output_str += f"[Macro-F1 Score] Comm Round {comm_round}\n"
#     output_str += f"  Target: Cloud Model\n"
#     output_str += f"  Cloud Macro-F1: {f1:.4f}\n"
#     output_str += f"{'='*60}\n"
#
#     print(output_str)
#
#     if args is not None:
#         log_filename = f"{args.log_name}.log"
#         with open(log_filename, "a") as f:
#             f.write(output_str)

def test_and_log_macro_f1_fewest_class_client(target_node, test_data_global, comm_round, args):
    """
    测试并记录包含种类最少的客户端模型 (Client Model) 的 Macro-F1
    """
    num_classes = args.class_num
    # 传入目标客户端的模型进行计算
    f1 = calculate_macro_f1(target_node.model, test_data_global, num_classes)

    output_str = f"\n{'='*60}\n"
    output_str += f"[Macro-F1 Score] Comm Round {comm_round}\n"
    output_str += f"  Target: Client {target_node.index} (Fewest classes: {len(target_node.class_stats)})\n"
    output_str += f"  Client Macro-F1: {f1:.4f}\n"
    output_str += f"{'='*60}\n"

    print(output_str)

    if args is not None:
        log_filename = f"{args.log_name}.log"
        with open(log_filename, "a") as f:
            f.write(output_str)

def calculate_accuracy(model, test_loader):
    """
    计算给定模型在测试集上的 Top-1 准确率
    """
    model.eval()
    accTop1_avg = utils.RunningAverage()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            labels = torch.tensor(labels, dtype=torch.long)

            # 模型返回三元组，取第一个元素 logits
            log_probs, _, _ = model(images)
            metrics = utils.accuracy(log_probs, labels, topk=(1,))
            accTop1_avg.update(metrics[0].item())

    return accTop1_avg.value()

def test_and_log_accuracy_variance_subset(target_nodes, test_data_global, comm_round, args):
    """
    修改后：只测试指定的客户端子集（种类最少的后30个），计算它们在全局测试集上的准确率方差
    """
    client_accs =[]

    # 获取选定客户端的准确率
    for node in target_nodes:
        acc = calculate_accuracy(node.model, test_data_global)
        client_accs.append(acc)

    if not client_accs:
        return

    num_clients = len(client_accs)

    # 1. 计算均值 μ_Acc
    mean_acc = sum(client_accs) / num_clients

    # 2. 计算方差 σ^2
    variance_acc = sum((acc - mean_acc) ** 2 for acc in client_accs) / num_clients

    # 3. 组织日志字符串
    output_str = f"\n{'*'*60}\n"
    output_str += f"[Client Accuracy Variance] Comm Round {comm_round}\n"
    output_str += f"  Target: Bottom {num_clients} Clients with fewest classes\n"
    output_str += f"  Mean Client Accuracy (μ): {mean_acc:.4f}%\n"
    output_str += f"  Accuracy Variance (σ²): {variance_acc:.4f}\n"
    output_str += f"{'*'*60}\n"

    # 打印到控制台
    print(output_str)

    # 追加写入日志文件
    if args is not None:
        log_filename = f"{args.log_name}.log"
        with open(log_filename, "a") as f:
            f.write(output_str)


def test_and_log_all_client_average_accuracy(all_leaf_nodes, test_data_global, comm_round, args):
    """
    测试所有客户端模型在全局测试集上的准确率，并计算平均准确率。

    Args:
        all_leaf_nodes: 所有叶子节点（客户端）列表
        test_data_global: 全局测试数据集
        comm_round: 当前通信轮次
        args: 参数对象
    """
    client_accs = []
    client_details = []

    # 获取每个客户端的准确率
    for node in all_leaf_nodes:
        acc = calculate_accuracy(node.model, test_data_global)
        client_accs.append(acc)
        client_details.append({
            'client_idx': node.index,
            'accuracy': acc,
            'num_classes': len(node.class_stats)
        })

    if not client_accs:
        return

    num_clients = len(client_accs)

    # 1. 计算平均准确率
    avg_acc = sum(client_accs) / num_clients

    # 2. 计算最高和最低准确率
    max_acc = max(client_accs)
    min_acc = min(client_accs)
    max_client = client_details[client_accs.index(max_acc)]
    min_client = client_details[client_accs.index(min_acc)]

    # 3. 计算方差
    variance_acc = sum((acc - avg_acc) ** 2 for acc in client_accs) / num_clients

    # 4. 计算标准差
    std_acc = variance_acc ** 0.5

    # 5. 组织日志字符串
    output_str = f"\n{'#'*60}\n"
    output_str += f"[All Clients Average Accuracy] Comm Round {comm_round}\n"
    output_str += f"  Total Clients: {num_clients}\n"
    output_str += f"  Average Accuracy (μ): {avg_acc:.4f}%\n"
    output_str += f"  Std Deviation (σ): {std_acc:.4f}%\n"
    output_str += f"  Variance (σ²): {variance_acc:.4f}\n"
    output_str += f"  Max Accuracy: {max_acc:.4f}% (Client {max_client['client_idx']}, {max_client['num_classes']} classes)\n"
    output_str += f"  Min Accuracy: {min_acc:.4f}% (Client {min_client['client_idx']}, {min_client['num_classes']} classes)\n"
    output_str += f"  Accuracy Range: {max_acc - min_acc:.4f}%\n"
    output_str += f"{'#'*60}\n"

    # 打印到控制台
    print(output_str)

    # 追加写入日志文件
    if args is not None:
        log_filename = f"{args.log_name}.log"
        with open(log_filename, "a") as f:
            f.write(output_str)

    return avg_acc  # 返回平均准确率，方便后续使用

# ======================= t-SNE 可视化模块 =======================
# def visualize_tsne_cloud(cloud_model, test_loader, comm_round, args):
#     """
#     针对云端模型，在全局测试集上提取 256 维投影特征并使用 t-SNE 进行可视化。
#     """
#     cloud_model.eval()
#     all_features = []
#     all_labels = []
#
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images = images.cuda()
#             labels = labels.numpy()
#             _, _, proj_features = cloud_model(images)
#             all_features.append(proj_features.cpu().numpy())
#             all_labels.append(labels)
#
#     all_features = np.concatenate(all_features, axis=0)
#     all_labels = np.concatenate(all_labels, axis=0)
#
#     # 随机采样以加快可视化过程
#     max_samples = 5000
#     if len(all_labels) > max_samples:
#         indices = np.random.choice(len(all_labels), max_samples, replace=False)
#         all_features = all_features[indices]
#         all_labels = all_labels[indices]
#
#     print(f"  [t-SNE] Calculating t-SNE for Cloud Model at Round {comm_round}...")
#
#     tsne = TSNE(n_components=2, random_state=42)
#     features_2d = tsne.fit_transform(all_features)
#
#     plt.figure(figsize=(10, 8))
#     scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=all_labels, cmap='tab10', s=15, alpha=0.8)
#     plt.legend(handles=scatter.legend_elements()[0], labels=set(all_labels), title="Classes")
#     plt.title(f"t-SNE Visualization (256D Features) - Cloud Model | Round {comm_round}")
#
#     # === 修改：将 tsne_save_dir 改为 tsne_dir ===
#     save_dir = getattr(args, "tsne_dir", "tsne_plots")
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, f"tsne_cloud_round_{comm_round}.png")
#
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.close()
#     print(f"  [t-SNE] Saved plot to {save_path}")


# ======================= 轮廓系数计算模块 (Cloud) =======================
# def calculate_and_log_silhouette_score_cloud(cloud_model, test_loader, comm_round, args):
#     """
#     针对云端模型，在全局测试集上提取特征，计算并在日志中记录轮廓系数。
#     """
#     cloud_model.eval()
#     all_features = []
#     all_labels = []
#
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images = images.cuda()
#             labels = labels.numpy()
#             _, _, proj_features = cloud_model(images)
#             all_features.append(proj_features.cpu().numpy())
#             all_labels.append(labels)
#
#     all_features = np.concatenate(all_features, axis=0)
#     all_labels = np.concatenate(all_labels, axis=0)
#
#     print(f"  [Silhouette] Calculating Silhouette Score for Cloud Model at Round {comm_round}...")
#     score = silhouette_score(all_features, all_labels, metric='euclidean')
#
#     output_str = f"  [Silhouette Score] Comm Round {comm_round} | Cloud Model | Global Score (S): {score:.4f}\n"
#     print(output_str.strip())
#
#     if args is not None:
#         log_filename = f"{args.log_name}.log"
#         with open(log_filename, "a") as f:
#             f.write(output_str)

def calculate_and_log_silhouette_score_fewest_class_client(target_node, test_loader, comm_round, args):
    """
    针对包含种类最少的客户端模型，在全局测试集上提取特征，计算并在日志中记录轮廓系数。
    """
    target_node.model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.numpy()
            # 客户端模型前向传播，取第三个返回值 proj_features (投影空间特征)
            _, _, proj_features = target_node.model(images)
            all_features.append(proj_features.cpu().numpy())
            all_labels.append(labels)

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"  [Silhouette] Calculating Silhouette Score for Client {target_node.index} (Fewest classes) at Round {comm_round}...")

    # 计算轮廓系数
    score = silhouette_score(all_features, all_labels, metric='euclidean')

    # 组织输出日志
    output_str = f"  [Silhouette Score] Comm Round {comm_round} | Client {target_node.index} (Fewest classes: {len(target_node.class_stats)}) | Global Score (S): {score:.4f}\n"
    print(output_str.strip())

    # 写入日志文件
    if args is not None:
        log_filename = f"{args.log_name}.log"
        with open(log_filename, "a") as f:
            f.write(output_str)

# ======================= Latent Representation Distance =======================

def calculate_and_log_representation_distance_cloud(
        cloud_model,
        test_loader,
        comm_round,
        args):

    """
    Calculate intra-class compactness and inter-class separability
    using projected contrastive features.
    """

    cloud_model.eval()

    all_features = []
    all_labels = []


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.cuda()
            labels = labels.cuda()

            # logits, backbone feature, projected feature
            _, _, proj_features = cloud_model(images)

            all_features.append(
                proj_features.cpu()
                .numpy()
            )

            all_labels.append(
                labels.cpu()
                .numpy()
            )


    features = np.concatenate(
        all_features,
        axis=0
    )

    labels = np.concatenate(
        all_labels,
        axis=0
    )


    # ===============================
    # calculate class centers
    # ===============================

    class_centers = []

    intra_distances = []


    unique_classes = np.unique(labels)


    for c in unique_classes:

        class_features = features[
            labels == c
            ]


        if len(class_features) == 0:
            continue


        center = np.mean(
            class_features,
            axis=0
        )

        class_centers.append(center)


        # intra-class distance

        distances = np.linalg.norm(
            class_features - center,
            axis=1
        )

        intra_distances.append(
            np.mean(distances)
        )


    intra_distance = np.mean(
        intra_distances
    )


    # ===============================
    # inter-class distance
    # ===============================

    class_centers = np.stack(
        class_centers,
        axis=0
    )


    inter_distances = []


    num_classes = len(class_centers)


    for i in range(num_classes):

        for j in range(i+1,num_classes):

            d = np.linalg.norm(
                class_centers[i]
                -
                class_centers[j]
            )

            inter_distances.append(d)


    inter_distance = np.mean(
        inter_distances
    )


    output_str = (
        f"\n[Representation Distance] "
        f"Comm Round {comm_round} | Cloud Model\n"
        f"  Intra-class Distance: "
        f"{intra_distance:.6f}\n"
        f"  Inter-class Distance: "
        f"{inter_distance:.6f}\n"
    )


    print(output_str)


    if args is not None:

        log_filename = f"{args.log_name}.log"

        with open(log_filename,"a") as f:
            f.write(output_str)


# ======================= 温度调度器 =======================
class TemperatureScheduler:
    """
    余弦退火温度调度器。
    训练初期 τ 较大 → 采样多样性高，后期 τ 衰减 → 采样更精准。

    公式: τ(t) = τ_min + 0.5 * (τ_max - τ_min) * (1 + cos(π * t / T))

    其中 t 为当前通信轮次，T 为总轮次。
    """

    def __init__(self, tau_max=2.0, tau_min=0.3, total_rounds=1000):
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.total_rounds = total_rounds

    def get_tau(self, current_round):
        """根据当前轮次返回温度值"""
        progress = current_round / max(self.total_rounds, 1)
        progress = min(progress, 1.0)  # 防止越界
        tau = self.tau_min + 0.5 * (self.tau_max - self.tau_min) * (1 + math.cos(math.pi * progress))
        return tau


# ======================= 类别自适应采样分配器（借鉴 Oort 统计效用） =======================
class ClassUtilityAllocator:
    """
    借鉴 Oort（SymbioticLab/Oort）的统计效用思想，将效用从「客户端级别」
    迁移到「类别级别」，实现按类别动态分配假样本数量。

    Oort 原始统计效用公式（training/param_server.py L262）:
        clientUtility = math.sqrt(iteration_loss) * size_of_sample_bin

    Oort 论文中统计效用的严格定义:
        U(i) = |B_i| * sqrt( (1/|B_i|) * Σ Loss(k)^2 )

    类别级别的统计效用公式:
        class_utility(c) = sqrt( mean( Loss_c^2 ) ) * scarcity_factor_c

    其中:
        - mean(Loss_c^2): 该类别在上一轮蒸馏中损失的均方值（注意是先平方再平均）
        - scarcity_factor_c: 数据稀缺度因子 = median_count / count_c

    参考 Oort 的归一化和裁剪策略（oort.py L289-L292 的 get_norm + clip_bound）：
        - 对效用值做 min-max 归一化到 [0, 1]
        - 裁剪极端值防止某类独占所有预算
    """

    def __init__(self, num_classes=10, clip_bound=0.95,
                 min_samples=8, max_ratio=4.0):
        """
        Args:
            num_classes: 类别总数
            clip_bound: Oort 风格的裁剪百分位（排除极端异常值）
            min_samples: 每类最少采样数
            max_ratio: 最大采样倍率（相对于均匀分配的倍数上限）
        """
        self.num_classes = num_classes
        self.clip_bound = clip_bound
        self.min_samples = min_samples
        self.max_ratio = max_ratio

        # 每类的历史均方损失（指数移动平均）
        # ★ 存储的是 mean(Loss^2)，而非 mean(Loss)
        self.class_avg_squared_loss = {}   # {class_id: float}
        self.ema_decay = 0.9               # EMA 衰减因子

    def update_class_loss(self, class_squared_losses):
        """
        每轮蒸馏后更新各类别的均方损失。

        Args:
            class_squared_losses: dict {class_id: mean(Loss^2) this round}
                                  ★ 注意：传入的已经是 Loss 的平方的均值
        """
        for c, sq_loss_val in class_squared_losses.items():
            if c in self.class_avg_squared_loss:
                self.class_avg_squared_loss[c] = (
                        self.ema_decay * self.class_avg_squared_loss[c]
                        + (1 - self.ema_decay) * sq_loss_val
                )
            else:
                self.class_avg_squared_loss[c] = sq_loss_val

    def _get_norm(self, values, clip_bound=0.95, thres=1e-4):
        """
        Oort 风格的归一化（参考 oort.py 的 get_norm 方法）。
        返回 (max_val, min_val, range_val, avg_val, clip_value)
        """
        if len(values) == 0:
            return 0, 0, thres, 0, 0

        sorted_vals = sorted(values)
        clip_idx = min(int(len(sorted_vals) * clip_bound), len(sorted_vals) - 1)
        clip_value = sorted_vals[clip_idx]

        max_val = sorted_vals[-1]
        min_val = sorted_vals[0]
        range_val = max(max_val - min_val, thres)
        avg_val = sum(values) / len(values)

        return max_val, min_val, range_val, avg_val, clip_value

    def compute_allocation(self, class_stats, total_budget):
        """
        核心方法：根据类别统计效用计算每类应分配的假样本数量。

        数学推导（严格对齐 Oort 论文）:
        1. 从存储中取出 mean(Loss_c^2)
        2. 计算 loss_factor = sqrt( mean(Loss_c^2) )
           → 这就是 Oort 论文中的 RMS（均方根）乘数
        3. scarcity_factor = median_count / count_c
        4. class_utility(c) = loss_factor * scarcity_factor

        Args:
            class_stats: dict {class_id: {'mean', 'cov', 'count', 'feat_shape'}}
            total_budget: 总假样本预算 = num_classes * samples_per_class

        Returns:
            allocation: dict {class_id: num_samples}
        """
        if len(class_stats) == 0:
            return {}

        classes = list(class_stats.keys())
        num_active_classes = len(classes)
        uniform_share = total_budget / max(num_active_classes, 1)

        # ===== 第一步：计算每类的原始统计效用 =====
        counts = {c: class_stats[c]['count'] for c in classes}
        median_count = float(np.median(list(counts.values())))

        raw_utilities = {}
        for c in classes:
            # ★ 严格对齐 Oort: loss_factor = sqrt( mean(Loss^2) )
            #   由于 self.class_avg_squared_loss[c] 存储的就是 mean(Loss^2)，
            #   所以 sqrt 后恰好还原 Oort 论文的 RMS 乘数
            if c in self.class_avg_squared_loss and self.class_avg_squared_loss[c] > 0:
                loss_factor = math.sqrt(self.class_avg_squared_loss[c])
            else:
                loss_factor = 1.0

            scarcity_factor = median_count / max(counts[c], 1.0)

            raw_utilities[c] = loss_factor * scarcity_factor

        # ===== 第二步：Oort 风格裁剪 =====
        utility_values = list(raw_utilities.values())
        _, min_util, range_util, _, clip_value = self._get_norm(
            utility_values, self.clip_bound)

        clipped_utilities = {}
        for c in classes:
            clipped_utilities[c] = min(raw_utilities[c], clip_value)

        # ===== 第三步：归一化 + 按比例分配 =====
        total_util = sum(clipped_utilities.values())
        if total_util < 1e-8:
            return {c: int(uniform_share) for c in classes}

        allocation = {}
        allocated_sum = 0
        max_samples = int(uniform_share * self.max_ratio)

        for c in classes:
            ratio = clipped_utilities[c] / total_util
            raw_alloc = ratio * total_budget
            alloc = int(max(self.min_samples, min(raw_alloc, max_samples)))
            allocation[c] = alloc
            allocated_sum += alloc

        # ===== 第四步：预算修正 =====
        if allocated_sum > total_budget:
            scale = total_budget / allocated_sum
            for c in classes:
                allocation[c] = max(self.min_samples, int(allocation[c] * scale))
        elif allocated_sum < total_budget:
            remaining = total_budget - allocated_sum
            sorted_by_util = sorted(classes, key=lambda c: clipped_utilities[c], reverse=True)
            for c in sorted_by_util:
                if remaining <= 0:
                    break
                add = min(remaining, max_samples - allocation[c])
                if add > 0:
                    allocation[c] += add
                    remaining -= add

        return allocation

# ======================= 升级：统一多角色与多权重的历史表示对照损失 =======================
# ======================= 升级：统一多角色、多权重与停止梯度的历史表征对照损失 =======================
class FlexibleHistoricalLoss(nn.Module):
    """
    针对 z_prev (上一轮特征) 的多权重/多角色/停止梯度对照损失函数
    支持五种模式 (通过 mode 参数调节):
    1. 'standard_moon': 默认 MOON，z_prev 作为硬负样本 (w_neg=1.0)
    2. 'weighted_push': 加权排斥，通过 w_neg 削弱 z_prev 的负样本排斥力
    3. 'hist_reg':      历史正则，把 z_prev 当作弱正样本拉近，防止灾难性遗忘 (w_reg 控制保留强度)
    4. 'adaptive_gate': 自适应门控，根据 Teacher 和 Prev 的冲突程度，动态决定排斥力
    5. 'stop_grad':     【新增：停止梯度实验】彻底抛弃负样本排斥，对 target 施加显式 Stop-Gradient 避免坍塌
    """
    def __init__(self, temperature=0.5, mode='weighted_push', w_neg=0.5, w_reg=0.5):
        super(FlexibleHistoricalLoss, self).__init__()
        self.temperature = temperature
        self.mode = mode
        self.w_neg = w_neg  # 负样本排斥权重 [0.0 ~ 1.0]
        self.w_reg = w_reg  # 历史正则/Stop-Grad 拉近权重 [0.0 ~ 1.0]
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.mse_loss = nn.MSELoss()

    def forward(self, z_student, z_teacher, z_prev):
        # 计算基础余弦相似度
        pos_sim = self.cos_sim(z_student, z_teacher) / self.temperature
        neg_sim = self.cos_sim(z_student, z_prev) / self.temperature

        if self.mode == 'standard_moon':
            # 1. 原始 MOON 模式 (硬负样本)
            logits = torch.stack([pos_sim, neg_sim], dim=1)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            return F.cross_entropy(logits, labels)

        elif self.mode == 'weighted_push':
            # 2. 加权排斥模式：削弱 z_prev 的负样本分母权重
            exp_pos = torch.exp(pos_sim)
            exp_neg = torch.exp(neg_sim)
            loss = -torch.log(exp_pos / (exp_pos + self.w_neg * exp_neg + 1e-8))
            return loss.mean()

        elif self.mode in ['hist_reg', 'reg_zprev', 'reg_ema']:
            # 3. 历史正则模式 (把 z_prev 视为需要保留的经验，而非偏差)
            loss_pos = (1.0 - (pos_sim * self.temperature)).mean()
            loss_reg = self.mse_loss(z_student, z_prev)
            return loss_pos + self.w_reg * loss_reg

        elif self.mode == 'adaptive_gate':
            # 4. 自适应门控模式：判断 Teacher 是否与 Prev 存在冲突
            with torch.no_grad():
                teacher_prev_sim = self.cos_sim(z_teacher, z_prev)
                conflict_weight = torch.clamp(1.0 - teacher_prev_sim, min=0.0, max=1.0)

            exp_pos = torch.exp(pos_sim)
            exp_neg = torch.exp(neg_sim)
            loss = -torch.log(exp_pos / (exp_pos + (self.w_neg * conflict_weight) * exp_neg + 1e-8))
            return loss.mean()

        elif self.mode == 'stop_grad':
            # =====================================================================
            # 5. 【新增：停止梯度对照实验】(SimSiam / BYOL 思想)
            # 理论逻辑：
            # (1) 彻底不再将 z_prev 作为负样本(Negative Sample)进行排斥；
            # (2) 将 z_teacher 和 z_prev 均作为正向特征指导锚点；
            # (3) 显式使用 .detach() 截断目标端的梯度回传 (Stop-Gradient)，
            #     构建非对称优化，证明仅靠 Stop-Grad 即可防止表征坍塌，无需强行排斥历史表征。
            # =====================================================================
            z_teacher_sg = z_teacher.detach()
            z_prev_sg = z_prev.detach()

            # 最大化学生与教师特征的余弦相似度 (等价于最小化余弦距离 1 - cos)
            loss_pos = -(self.cos_sim(z_student, z_teacher_sg)).mean()

            # 最大化学生与历史特征的余弦相似度 (通过 w_reg 控制对历史经验的吸收比例)
            loss_hist = -(self.cos_sim(z_student, z_prev_sg)).mean()

            # 返回综合损失：既不排斥历史，又不会发生模型退化
            return loss_pos + self.w_reg * loss_hist

        else:
            raise ValueError(f"Unknown historical loss mode: {self.mode}")

#仅包含正样本的对比学习
# class ContrastiveLoss(nn.Module):
#     """
#     【消融实验：仅正样本】
#     移除与 z_prev 的负样本排斥，仅保留与 z_teacher 的正样本拉近。
#     采用最大化余弦相似度（最小化余弦距离）作为损失。
#     """
#     def __init__(self, temperature=0.5):
#         super(ContrastiveLoss, self).__init__()
#         # 仅保留正样本时，其实可以不需要 temperature，但为了接口兼容保留
#         self.temperature = temperature
#         self.cos_sim = nn.CosineSimilarity(dim=-1)
#
#     def forward(self, z_student, z_teacher): # 移除了 z_prev
#         """
#         Args:
#             z_student: (B, out_dim) 当前学生模型的投影特征
#             z_teacher: (B, out_dim) 教师/邻居模型的投影特征
#         """
#         # 计算正样本的余弦相似度
#         pos_sim = self.cos_sim(z_student, z_teacher)
#
#         # 将相似度映射为损失（相似度为1时，loss为0）
#         loss = (1.0 - pos_sim).mean()
#         return loss


# ======================= 按类别统计特征分布（保持不变） =======================
def compute_class_statistics(noises, labels):
    """
    升级版：将端侧编码后的特征按类别聚类，计算每个类的均值向量 μ_c 和完整协方差矩阵 Σ_c。
    特征在通道和空间维度上展平为二维矩阵 (N_c, D)，其中 D = C * H * W。
    """
    class_stats = {}
    if len(noises) == 0:
        return class_stats

    all_noises = torch.cat(noises, dim=0)  # Shape: (N, C, H, W)
    all_labels = torch.cat(labels, dim=0)  # Shape: (N,)

    # 记录原始特征的空间结构，供采样时逆向 reshape 使用
    feat_shape = all_noises.shape[1:]  # (C, H, W)

    # 将特征展平为二维: (N, D)，其中 D = C * H * W
    N = all_noises.shape[0]
    D = all_noises[0].numel()  # C * H * W
    all_flat = all_noises.view(N, D)  # (N, D)

    unique_classes = torch.unique(all_labels)

    for c_tensor in unique_classes:
        c = c_tensor.item()
        mask = (all_labels == c_tensor)
        c_flat = all_flat[mask]  # Shape: (N_c, D)
        N_c = c_flat.shape[0]

        # 均值向量: (D,)
        mu = c_flat.mean(dim=0)

        # 完整协方差矩阵: (D, D)
        if N_c > 1:
            # 中心化
            centered = c_flat - mu.unsqueeze(0)  # (N_c, D)
            # 协方差 = (centered^T @ centered) / (N_c - 1)
            cov = (centered.t() @ centered) / (N_c - 1)  # (D, D)
        else:
            # 样本数不足，使用单位对角阵的缩放版
            cov = torch.eye(D, device=c_flat.device, dtype=c_flat.dtype) * 0.01

        # 对角线加微小常数，确保正定性（防止 Cholesky 分解失败）
        cov = cov + torch.eye(D, device=cov.device, dtype=cov.dtype) * 1e-6

        class_stats[c] = {
            'mean': mu.detach(),          # (D,) 展平后的均值向量
            'cov': cov.detach(),           # (D, D) 完整协方差矩阵
            'count': N_c,
            'feat_shape': feat_shape,      # (C, H, W) 用于采样后逆向 reshape
        }

    return class_stats


# ======================= 带温度的截断采样 =======================
# ======================= 带温度的截断采样（升级版：支持类别自适应数量）=======================
def sample_from_stats(class_stats, samples_per_class, tau, truncation=2.0, allocation=None):
    """
    升级版：支持按类别分配不同数量的假样本。

    Args:
        class_stats: 类别统计参数
        samples_per_class: 默认每类采样数（当 allocation 为 None 时使用）
        tau: 温度参数
        truncation: 截断阈值
        allocation: 可选; dict {class_id: num_samples}，按类别的自适应采样数量。
                    如果提供，则 samples_per_class 参数被忽略。
    """
    sampled_noises = []
    sampled_labels = []

    for c, stats in class_stats.items():
        # ★ 核心修改：优先使用 allocation 中的自适应数量，否则回退到均匀分配
        if allocation is not None and c in allocation:
            current_samples = allocation[c]
        else:
            current_samples = samples_per_class

        # 如果分配数量为 0，直接跳过该类别
        if current_samples <= 0:
            continue

        mu = stats['mean']
        cov = stats['cov']
        feat_shape = stats['feat_shape']
        D = mu.shape[0]

        # 1. 生成标准高斯白噪声矩阵
        # ★ 将原来的 samples_per_class 替换为 current_samples
        epsilon = torch.randn(current_samples, D, device=mu.device, dtype=mu.dtype)
        epsilon = torch.clamp(epsilon, -truncation, truncation)

        # 2. Cholesky 分解
        try:
            L = torch.linalg.cholesky(cov)
        except torch._C._LinAlgError:
            cov_fixed = cov + torch.eye(D, device=cov.device, dtype=cov.dtype) * 1e-4
            L = torch.linalg.cholesky(cov_fixed)


        z_sampled = mu.unsqueeze(0) + tau * (epsilon @ L.t())
        # 3. 多元高斯变换 (★ 创新点一消融实验修改：抛弃协方差矩阵L，仅使用均值和各向同性白噪声)
        # z_sampled = mu.unsqueeze(0) + tau * epsilon * 0.1

        # 4. 逆向 reshape
        # ★ 将原来的 samples_per_class 替换为 current_samples
        z_sampled = z_sampled.view(current_samples, *feat_shape)

        # 5. ReLU 截断
        z_sampled = torch.relu(z_sampled)

        sampled_noises.append(z_sampled)

        # ★ 将原来的 samples_per_class 替换为 current_samples
        label_tensor = torch.full((current_samples,), c, dtype=torch.long, device=mu.device)
        sampled_labels.append(label_tensor)

    return sampled_noises, sampled_labels


# ======================= 主流程入口 =======================
def run_fedagg(client_models, edge_models, cloud_model, train_data_local_num_dict,
               test_data_local_num_dict, train_data_local_dict, test_data_local_dict,
               test_data_global, args, net_dataidx_map_train=None):
    global comm_tracker # 声明使用全局的通信追踪器

    V1 = [Node(args, cloud_model)]
    V2 = create_child_for_upper_level(args, V1, args.edge_number, edge_models)
    assert args.client_number % args.edge_number == 0
    V3 = create_child_for_upper_level(args, V2, args.client_number // args.edge_number, client_models)
    for idx, node in enumerate(V3):
        node.dataset = [_ for _ in train_data_local_dict[idx]]
        if net_dataidx_map_train is not None:
            node.dataidxs = net_dataidx_map_train[idx]

    Init(V1[0])

    # ===== 新增：记录并输出第0轮的通信开销 =====
    print("=" * 60)
    print(f"[Communication Overhead] Round 0 (Init Phase):")
    print(f"  Client -> Edge (Stats Transfer): {comm_tracker.round_0_client_edge_bytes / 1024 / 1024:.4f} MB")
    print(f"  Edge -> Cloud (Stats Transfer):  {comm_tracker.round_0_edge_cloud_bytes / 1024 / 1024:.4f} MB")
    print("=" * 60)

    # 追加到日志文件
    if args is not None:
        log_filename = f"{args.log_name}.log"
        with open(log_filename, "a") as f:
            f.write(f"[Communication Overhead] Round 0 Client->Edge: {comm_tracker.round_0_client_edge_bytes / 1024 / 1024:.4f} MB\n")
            f.write(f"[Communication Overhead] Round 0 Edge->Cloud: {comm_tracker.round_0_edge_cloud_bytes / 1024 / 1024:.4f} MB\n")
    # ============================================

    tau_scheduler = TemperatureScheduler(
        tau_max=getattr(args, 'tau_max', 2.0),
        tau_min=getattr(args, 'tau_min', 0.3),
        total_rounds=args.comm_round
    )
    samples_per_class = getattr(args, 'samples_per_class', 64)

    # ★ 新增：创建类别自适应采样分配器（借鉴 Oort 统计效用）
    allocator = ClassUtilityAllocator(
        num_classes=args.class_num,
        clip_bound=getattr(args, 'util_clip_bound', 0.95),
        min_samples=getattr(args, 'min_samples_per_class', 8),
        max_ratio=getattr(args, 'max_sample_ratio', 4.0),
    )

    # ★ 预热轮数：前 N 轮使用均匀分配（因为还没有损失信息）
    warmup_rounds = getattr(args, 'utility_warmup_rounds', 10)

    for comm_round in range(args.comm_round):
        # ==================== ★ 修改：根据消融实验参数控制温度 Tau ★ ====================
        sampling_mode = getattr(args, 'sampling_ablation', 'full')

        if sampling_mode in ['no_temp', 'none']:
            # 仅去掉温度调整（或全关）：使用固定的标准温度（默认 1.0）
            tau = getattr(args, 'fixed_tau', 1.0)
            if comm_round % 50 == 0:
                print(f"[Round {comm_round}] (Ablation: No Temp Schedule) Fixed tau = {tau:.4f}")
        else:
            # 正常开启余弦退火温度调度
            tau = tau_scheduler.get_tau(comm_round)
            if comm_round % 50 == 0:
                print(f"[Round {comm_round}] Temperature tau = {tau:.4f}")
        # ==============================================================================

        # ==================== ★ 修改：根据消融实验参数控制分配器 Allocator ★ ====================
        if sampling_mode in ['no_alloc', 'none']:
            # 仅去掉自适应分配（或全关）：强制设为 None，退回到每类均匀采样 (samples_per_class)
            active_allocator = None
            if comm_round == 0:
                print("[Ablation Notice] Adaptive Allocation is DISABLED. Using uniform sampling.")
        else:
            # 正常开启：预热期后使用类别自适应分配器
            active_allocator = allocator if comm_round >= warmup_rounds else None
        # ==============================================================================

        # ★ 每轮蒸馏前重新采样（根据 active_allocator 自动决定是自适应分配还是均匀分配）
        resample_all_nodes(V1[0], samples_per_class, tau, allocator=active_allocator)

        # ★ 蒸馏训练，收集类别均方损失
        round_class_losses = train_FedAgg(V1[0], args, comm_round)

        # ===== 新增：记录并输出本轮（双向）的通信开销 =====
        print("-" * 60)
        print(f"[Communication Overhead] Round {comm_round}:")
        print(f"  Client <-> Edge (Bidirectional): {comm_tracker.current_round_client_edge_bytes / 1024 / 1024:.4f} MB")
        print(f"  Edge <-> Cloud (Bidirectional):  {comm_tracker.current_round_edge_cloud_bytes / 1024 / 1024:.4f} MB")
        print("-" * 60)

        # 追加到日志文件
        if args is not None:
            log_filename = f"{args.log_name}.log"
            with open(log_filename, "a") as f:
                f.write(f"[Communication Overhead] Round {comm_round} Client<->Edge: {comm_tracker.current_round_client_edge_bytes / 1024 / 1024:.4f} MB\n")
                f.write(f"[Communication Overhead] Round {comm_round} Edge<->Cloud: {comm_tracker.current_round_edge_cloud_bytes / 1024 / 1024:.4f} MB\n")
        # ============================================

        # ★ 用本轮的类别均方损失更新分配器
        #    对应 Oort 的 update_client_util(feedbacks['reward'])
        if round_class_losses and sampling_mode in ['full', 'no_temp']:
            allocator.update_class_loss(round_class_losses)
            if comm_round % 50 == 0:
                total_budget = samples_per_class * len(V1[0].class_stats)
                alloc = allocator.compute_allocation(V1[0].class_stats, total_budget)
                print(f"  [Adaptive Allocation] {dict(sorted(alloc.items()))}")

        update_prev_models(V1[0])

        test_on_cloud(V1[0].model, test_data_global, comm_round, args)

        # ================= Representation Distance =================

        calculate_and_log_representation_distance_cloud(
            V1[0].model,
            test_data_global,
            comm_round + 1,
            args
        )

        # ===========================================================

        # ==================== ★ 修改：每个轮次报告平均准确率 ★ ====================
        # 1. 获取所有叶子节点（客户端）并过滤掉没有数据的异常节点
        leaf_nodes = get_all_leaf_nodes(V1[0])
        if leaf_nodes:
            valid_leaf_nodes = [n for n in leaf_nodes if len(n.class_stats) > 0]

            if valid_leaf_nodes:
                # 按照节点拥有的类别数量进行升序排序（类别越少越靠前）
                valid_leaf_nodes.sort(key=lambda n: len(n.class_stats))

                # 提取包含种类最少的后 30 个客户端
                target_nodes_bottom30 = valid_leaf_nodes[:30]

                # 【执行报告 1】：计算并输出“所有客户端”的平均准确率
                test_and_log_all_client_average_accuracy(valid_leaf_nodes, test_data_global, comm_round, args)

                # 【执行报告 2】：计算并输出“种类最少30个客户端”的平均准确率（及方差）
                test_and_log_accuracy_variance_subset(target_nodes_bottom30, test_data_global, comm_round, args)

                # ==================== 定期报告耗时较长的其他指标 ====================
                if comm_round % 20 == 0:
                    target_node_fewest = valid_leaf_nodes[0]
                    print(f"[Evaluation] Round {comm_round} - Selected Client {target_node_fewest.index} as fewest-class client ({len(target_node_fewest.class_stats)} classes).")

                    # Macro-F1 与 轮廓系数评估
                    test_and_log_macro_f1_fewest_class_client(target_node_fewest, test_data_global, comm_round, args)
                    calculate_and_log_silhouette_score_fewest_class_client(target_node_fewest, test_data_global, comm_round, args)
        # ==============================================================================



def update_prev_models(node):
    """
    根据命令行参数选择更新上一轮快照 (prev_model) 或指数移动平均模型 (ema_model)。
    """
    ablation_mode = getattr(node.args, 'ablation_mode', 'none')

    if ablation_mode == 'reg_ema':
        ema_momentum = getattr(node.args, 'ema_momentum', 0.99)
        # ★ EMA 平滑更新：θ_ema = m * θ_ema + (1 - m) * θ_loc
        with torch.no_grad():
            for param_ema, param_model in zip(node.ema_model.parameters(), node.model.parameters()):
                param_ema.data = param_ema.data * ema_momentum + param_model.data * (1.0 - ema_momentum)
    else:
        # ★ 替换 deepcopy 为 load_state_dict，大幅提升速度并防止显存泄漏
        node.prev_model.load_state_dict(node.model.state_dict())

    # 递归更新所有子节点
    for child in node.children:
        update_prev_models(child)


global_index = 0


class Node:
    def __init__(self, args, model, father=None):
        global global_index
        self.model = model.cuda()
        self.dataset = None
        self.father = father
        self.children = []
        self.index = global_index
        self.autoencoder = create_autoencoder().cuda()
        self.noises = []
        self.labels = []
        # ========== 新增：存储类别统计参数 {class_id: {'mean', 'var', 'count'}} ==========
        self.class_stats = {}
        self.args = args
        # ===== 新增：存储该节点对应的数据索引，用于 Init 阶段构建专用 DataLoader =====
        self.dataidxs = None

        # ★ 新增：上一轮冻结的模型副本（用于对比损失中的负样本）
        # 参考 MOON: 维护 prev_model / old_model_pool
        self.prev_model = copy.deepcopy(model).cuda()
        self.prev_model.eval()
        for param in self.prev_model.parameters():
            param.requires_grad = False

            # ★ 新增：EMA 教师模型副本（用于 reg_ema）
        self.ema_model = copy.deepcopy(model).cuda()
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False

        global_index = global_index + 1

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.father is None


def create_child_for_upper_level(args, upper_level, children_number, models):
    result = []
    for ele in upper_level:
        sub_nodes = []
        for _, model in zip(range(children_number), models):
            sub_nodes.append(Node(args, model, ele))
        ele.children = sub_nodes
        result.extend(sub_nodes)
    return result


def train_FedAgg(node, args, comm_round):
    """
    修改后的训练函数：收集所有蒸馏对的类别均方损失并返回聚合结果。
    """
    all_class_losses = defaultdict(list)

    def _train_recursive(node):
        if node.is_root():
            for child in node.children:
                _train_recursive(child)
        elif node.is_leaf():
            # 传入 comm_round
            losses = BSBODP(node, node.father, args, comm_round)
            if losses:
                for c, v in losses.items():
                    all_class_losses[c].append(v)
        else:
            for child in node.children:
                _train_recursive(child)
            # 传入 comm_round
            losses = BSBODP(node, node.father, args, comm_round)
            if losses:
                for c, v in losses.items():
                    all_class_losses[c].append(v)

    _train_recursive(node)

    # 聚合全网的类别均方损失
    aggregated = {}
    for c, vals in all_class_losses.items():
        aggregated[c] = sum(vals) / len(vals)

    return aggregated


# ======================= 修改后的 Init =======================
def Init(node):
    """
    修改后的初始化：
    - 叶子节点：用大 batch_size + drop_last=False 的 DataLoader 加速编码
    - 确保所有节点（包括小样本节点）都能上传统计参数
    """
    if node.is_root():
        for child in node.children:
            Init(child)
    elif node.is_leaf():
        # ========== 修改点：使用专用的 Init DataLoader ==========
        from data_loader import get_dataloader_for_init

        init_batch_size = getattr(node.args, 'init_batch_size', 128)

        # 使用 dataidxs 构建专用 DataLoader（大batch + drop_last=False）
        if node.dataidxs is not None:
            init_loader = get_dataloader_for_init(
                node.args.dataset,
                node.args.data_dir,
                init_batch_size,
                node.dataidxs
            )
        else:
            # 兼容：如果没有设置 dataidxs，退回到使用 node.dataset
            init_loader = node.dataset

        local_noises = []
        local_labels = []

        with torch.no_grad():
            for batch_data in init_loader:
                img, label = batch_data
                img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True)

                # ====== 修复 1：反归一化，将输入特征域对齐到 AE 的 [0, 1] ======
                if getattr(node.args, 'dataset', 'cifar10') == 'cifar100':
                    mean = torch.tensor([0.50707516, 0.48654887, 0.44091784], device=img.device).view(1, 3, 1, 1)
                    std = torch.tensor([0.26733429, 0.25643846, 0.27615047], device=img.device).view(1, 3, 1, 1)
                else:
                    mean = torch.tensor([0.49139968, 0.48215827, 0.44653124], device=img.device).view(1, 3, 1, 1)
                    std = torch.tensor([0.24703233, 0.24348505, 0.26158768], device=img.device).view(1, 3, 1, 1)

                img_unnorm = img * std + mean
                img_unnorm = torch.clamp(img_unnorm, 0.0, 1.0) # 防御性截断
                # =========================================================

                # 这里原来是 encoder(img)，现在改为 encoder(img_unnorm)
                noise = node.autoencoder.encoder(img_unnorm)
                local_noises.append(noise)
                local_labels.append(label)

        # 即使该节点样本很少（甚至只有1个），也能正常计算统计参数
        if len(local_noises) > 0:
            # ==== 核心提速 2：直接保留大 Batch 列表，彻底消除 5万次 split 带来的显存碎片 ====
            node.noises = local_noises
            node.labels = local_labels
            # ========================================================
            node.class_stats = compute_class_statistics(local_noises, local_labels)
            merge_stats_to_parent(node)
            total_samples = sum(n.shape[0] for n in local_noises)
            print(f"  [Leaf Node {node.index}] Uploaded stats for {len(node.class_stats)} classes "
                  f"(encoded {total_samples} samples)")
        else:
            print(f"  [Leaf Node {node.index}] WARNING: No data available, skipping.")
    else:
        for child in node.children:
            Init(child)
        # 非叶非根节点：将自己聚合好的 class_stats 继续上传给父节点
        merge_stats_to_parent(node)


def merge_stats_to_parent(node):
    """
    将子节点的类别统计参数合并到父节点。
    使用严格的池化协方差（Pooled Covariance）公式：
        Σ_merged = [ (n1-1)*Σ1 + (n2-1)*Σ2 + n1*n2/(n1+n2) * (μ1-μ2)(μ1-μ2)^T ] / (n1+n2-1)

    其中 (μ1-μ2)(μ1-μ2)^T 是均值差的外积矩阵，用于恢复跨组方差。
    """
    global comm_tracker  # 声明使用全局的通信追踪器
    if node.father is None:
        return

    # ===== 新增：统计第0轮均值和协方差矩阵的传输数据量 =====
    bytes_sent = 0
    for c, stats in node.class_stats.items():
        D = stats['mean'].numel()
        # mean向量(D), cov矩阵(D*D), count标量(1)
        # 默认数据类型为 float32 (占据 4 Bytes)
        bytes_sent += (D + D * D + 1) * 4

    if node.is_leaf(): # Client 向 Edge 传递
        comm_tracker.round_0_client_edge_bytes += bytes_sent
    else:              # Edge 向 Cloud 传递
        comm_tracker.round_0_edge_cloud_bytes += bytes_sent
    # =======================================================

    father = node.father
    for c, stats in node.class_stats.items():
        if c not in father.class_stats:
            # 父节点还没有这个类别，直接复制
            father.class_stats[c] = {
                'mean': stats['mean'].clone(),
                'cov': stats['cov'].clone(),
                'count': stats['count'],
                'feat_shape': stats['feat_shape'],
            }
        else:
            # 父节点已有这个类别，做池化协方差合并
            f_stats = father.class_stats[c]
            n1, n2 = f_stats['count'], stats['count']
            total = n1 + n2

            # 1. 加权平均均值向量
            mu_merged = (n1 * f_stats['mean'] + n2 * stats['mean']) / total

            # 2. 池化协方差合并
            if total > 1:
                # 均值差向量: (D,)
                mu_diff = f_stats['mean'] - stats['mean']  # (D,)
                # 均值差的外积矩阵: (D, D)  ← 这是恢复跨组方差的关键项！
                outer_product = mu_diff.unsqueeze(1) @ mu_diff.unsqueeze(0)  # (D,1) x (1,D) -> (D,D)

                cov_merged = (
                                     max(n1 - 1, 0) * f_stats['cov'] +
                                     max(n2 - 1, 0) * stats['cov'] +
                                     (n1 * n2 / total) * outer_product
                             ) / max(total - 1, 1)
            else:
                cov_merged = stats['cov'].clone()

            # 3. 对角线加微小常数，防范浮点误差导致的非正定问题
            D = cov_merged.shape[0]
            cov_merged = cov_merged + torch.eye(D, device=cov_merged.device, dtype=cov_merged.dtype) * 1e-6

            father.class_stats[c] = {
                'mean': mu_merged.detach(),
                'cov': cov_merged.detach(),
                'count': total,
                'feat_shape': stats['feat_shape'],
            }


# ======================= 每轮重新采样 =======================
def resample_all_nodes(node, samples_per_class, tau, allocator=None):
    """
    每轮蒸馏前，所有非叶节点从统计分布中重新采样。
    ★ 升级：如果提供了 allocator，则按类别自适应分配采样数量。
    """
    if node.is_root():
        if len(node.class_stats) > 0:
            allocation = None
            if allocator is not None:
                total_budget = samples_per_class * len(node.class_stats)
                allocation = allocator.compute_allocation(node.class_stats, total_budget)
            # ★ 传入 allocation 关���字参数（对应修改 2 中新增的签名）
            node.noises, node.labels = sample_from_stats(
                node.class_stats, samples_per_class, tau, allocation=allocation)
        for child in node.children:
            resample_all_nodes(child, samples_per_class, tau, allocator=allocator)
    elif node.is_leaf():
        pass
    else:
        if len(node.class_stats) > 0:
            allocation = None
            if allocator is not None:
                total_budget = samples_per_class * len(node.class_stats)
                allocation = allocator.compute_allocation(node.class_stats, total_budget)
            node.noises, node.labels = sample_from_stats(
                node.class_stats, samples_per_class, tau, allocation=allocation)
        for child in node.children:
            resample_all_nodes(child, samples_per_class, tau, allocator=allocator)


def BSBODP(node1, node2, args, comm_round):
    global comm_tracker

    # ===== 修正后的统计端边、边云双向交互的数据量 =====
    # 统计本轮进行蒸馏交互的样本数量 (取两者的最小值，与 BSBODP_dir 的逻辑一致)
    len1 = sum(b.shape[0] for b in node1.noises) if len(node1.noises) > 0 else 0
    len2 = sum(b.shape[0] for b in node2.noises) if len(node2.noises) > 0 else 0
    num_samples = min(len1, len2)

    # 每次前向传播传递的数据：
    # 1. nei_logits: 大小为 class_num
    # 2. z_teacher:  大小为 out_dim (用于对比损失)
    out_dim = getattr(args, 'out_dim', 256)
    bytes_per_sample = (args.class_num + out_dim) * 4  # float32 占 4 Bytes

    # 节点1和节点2互相蒸馏，所以是双向传输，乘以 2
    tx_bytes = num_samples * bytes_per_sample * 2

    if node1.is_leaf() or node2.is_leaf(): # 端设备与边缘侧交互
        comm_tracker.current_round_client_edge_bytes += tx_bytes
    else: # 边缘侧与云设备交互
        comm_tracker.current_round_edge_cloud_bytes += tx_bytes
    # =================================================

    class_losses_1 = BSBODP_dir(node1, node2, args, comm_round)
    class_losses_2 = BSBODP_dir(node2, node1, args, comm_round)

    # 合并双向的类别均方损失信息
    merged_losses = {}
    all_classes = set()
    if class_losses_1:
        all_classes.update(class_losses_1.keys())
    if class_losses_2:
        all_classes.update(class_losses_2.keys())

    for c in all_classes:
        vals = []
        if class_losses_1 and c in class_losses_1:
            vals.append(class_losses_1[c])
        if class_losses_2 and c in class_losses_2:
            vals.append(class_losses_2[c])
        if vals:
            merged_losses[c] = sum(vals) / len(vals)

    return merged_losses

#包含正、负样本的对比学习
class Loss_Non_Leaf(nn.Module):
    def __init__(self, temperature=1, alpha=0.1, mu_con=1.0, con_temperature=0.5,
                 ablation_mode='standard_moon', w_neg=0.5, w_reg=1.0): # ★ 新增 w_neg 和 w_reg
        super(Loss_Non_Leaf, self).__init__()
        self.alpha = alpha
        self.mu_con = mu_con
        self.kl_loss_crit = KL_Loss(temperature)
        self.ce_loss_crit = nn.CrossEntropyLoss()

        # ★ 使用新的灵活对照损失类
        self.con_loss_crit = FlexibleHistoricalLoss(
            temperature=con_temperature,
            mode=ablation_mode,
            w_neg=w_neg,
            w_reg=w_reg
        )

    def forward(self, output_batch, teacher_outputs, label,
                z_student=None, z_teacher=None, z_prev=None, use_contrastive=True):
        loss_ce = self.ce_loss_crit(output_batch, label.long())
        loss_kl = self.kl_loss_crit(output_batch, teacher_outputs.detach())
        loss = loss_ce + self.alpha * loss_kl

        if use_contrastive and z_student is not None and z_teacher is not None and z_prev is not None:
            loss_con = self.con_loss_crit(z_student, z_teacher, z_prev)
            loss = loss + self.mu_con * loss_con
        return loss

#仅包含正样本的对比学习
# class Loss_Non_Leaf(nn.Module):
#     """
#     非叶节点损失 = CE(硬标签) + α·KL(软标签蒸馏) + μ·ContrastiveLoss
#     """
#     def __init__(self, temperature=1, alpha=0.1, mu_con=1.0, con_temperature=0.5):
#         super(Loss_Non_Leaf, self).__init__()
#         self.alpha = alpha
#         self.mu_con = mu_con
#         self.kl_loss_crit = KL_Loss(temperature)
#         # 🔴 修复 1：必须恢复 CE Loss，给 Edge 和 Cloud 提供硬标签锚点
#         self.ce_loss_crit = nn.CrossEntropyLoss()
#         self.con_loss_crit = ContrastiveLoss(con_temperature)
#
#     # 增加 use_contrastive 参数，默认为 True 兼容旧代码
#     def forward(self, output_batch, teacher_outputs, label,
#                 z_student=None, z_teacher=None, use_contrastive=True):
#         loss_ce = self.ce_loss_crit(output_batch, label.long())
#         loss_kl = self.kl_loss_crit(output_batch, teacher_outputs.detach())
#         loss = loss_ce + self.alpha * loss_kl
#
#         # 增加 use_contrastive 判断逻辑
#         if use_contrastive and z_student is not None and z_teacher is not None:
#             loss_con = self.con_loss_crit(z_student, z_teacher)
#             loss = loss + self.mu_con * loss_con
#
#         return loss

#包含完整的对比学习
# ======================= 修改后的 Loss_Leaf =======================
class Loss_Leaf(nn.Module):
    def __init__(self, temperature=1, alpha=1, alpha2=1, mu_con=1.0, con_temperature=0.5,
                 ablation_mode='standard_moon', w_neg=0.5, w_reg=1.0): # ★ 新增 w_neg 和 w_reg
        super(Loss_Leaf, self).__init__()
        self.alpha = alpha
        self.alpha2 = alpha2
        self.mu_con = mu_con
        self.kl_loss_crit = KL_Loss(temperature)
        self.ce_loss_crit = nn.CrossEntropyLoss()

        # ★ 使用新的灵活对照损失类
        self.con_loss_crit = FlexibleHistoricalLoss(
            temperature=con_temperature,
            mode=ablation_mode,
            w_neg=w_neg,
            w_reg=w_reg
        )

    def forward(self, output_fake, teacher_outputs_fake, output_true, label_true,
                z_student=None, z_teacher=None, z_prev=None):
        loss_kl = self.kl_loss_crit(output_fake, teacher_outputs_fake.detach())
        loss_ce = self.ce_loss_crit(output_true, label_true.long())
        loss_total = self.alpha * loss_kl + self.alpha2 * loss_ce

        if z_student is not None and z_teacher is not None and z_prev is not None:
            loss_con = self.con_loss_crit(z_student, z_teacher, z_prev)
            loss_total = loss_total + self.mu_con * loss_con

        return loss_total

#仅包含正样本的对比学习
# class Loss_Leaf(nn.Module):
#     """
#     用于叶子节点（Client）的混合损失。
#
#     包含两部分：
#     1. 假数据蒸馏损失（纯 KL）—— 对齐与父节点/邻居的软输出
#     2. 真数据分类损失（CE）—— 在本地真实数据上训练分类能力
#
#     ⚠️ 核心修改：假数据部分不再使用硬标签 CE，只用 KL。
#     """
#     def __init__(self, temperature=1, alpha=1, alpha2=1, mu_con=1.0, con_temperature=0.5):
#         super(Loss_Leaf, self).__init__()
#         self.alpha = alpha
#         self.alpha2 = alpha2
#         self.mu_con = mu_con
#         self.kl_loss_crit = KL_Loss(temperature)
#         self.ce_loss_crit = nn.CrossEntropyLoss()
#         self.con_loss_crit = ContrastiveLoss(con_temperature)
#
#     def forward(self, output_fake, teacher_outputs_fake, output_true, label_true,
#                 z_student=None, z_teacher=None): # 去掉 z_prev
#         loss_kl = self.kl_loss_crit(output_fake, teacher_outputs_fake.detach())
#         loss_ce = self.ce_loss_crit(output_true, label_true.long())
#         loss_total = self.alpha * loss_kl + self.alpha2 * loss_ce
#
#         if z_student is not None and z_teacher is not None:
#             loss_con = self.con_loss_crit(z_student, z_teacher) # 去掉 z_prev
#             loss_total = loss_total + self.mu_con * loss_con
#
#         return loss_total




#完整的代码
def BSBODP_dir(node_origin, node_neigh, args, comm_round):
    from collections import defaultdict
    batch_size = args.batch_size
    con_warmup = getattr(args, 'con_warmup', 20)
    use_contrastive = (comm_round >= con_warmup)

    # ========== ★ 修改：新增对不同历史特征权重的解析 ==========
    ablation_mode = getattr(args, 'ablation_mode', 'standard_moon')
    w_neg = getattr(args, 'w_neg', 0.5)      # 负样本排斥力系数
    w_reg = getattr(args, 'w_reg', 0.5)     # 历史特征正则保留系数 (原 mu_reg)
    # ======================================================

    len_neigh = sum(b.shape[0] for b in node_neigh.noises) if len(node_neigh.noises) > 0 else 0
    len_origin = sum(b.shape[0] for b in node_origin.noises) if len(node_origin.noises) > 0 else 0

    noises = node_neigh.noises if len_neigh < len_origin else node_origin.noises
    labels = node_neigh.labels if len_neigh < len_origin else node_origin.labels

    if len(noises) == 0:
        return {}

    noises_tensor = torch.cat(noises, dim=0)
    labels_tensor = torch.cat(labels, dim=0)

    T_agg = args.T_agg
    if isinstance(T_agg, list):
        T_agg = T_agg[0]

    mu_con = getattr(args, 'mu_con', 1.0)
    con_temperature = getattr(args, 'con_temperature', 0.5)

    # ===== ★ 修改：实例化非叶节点损失，传入权重参数 =====
    crit_non_leaf = Loss_Non_Leaf(
        T_agg, mu_con=mu_con, con_temperature=con_temperature,
        ablation_mode=ablation_mode, w_neg=w_neg, w_reg=w_reg
    )

    ce_loss_crit_per_sample = nn.CrossEntropyLoss(reduction='none')
    ce_loss_crit_mean = nn.CrossEntropyLoss()

    # ===== ★ 修改：实例化真数据使用的对比损失 =====
    contrastive_crit = FlexibleHistoricalLoss(
        temperature=con_temperature, mode=ablation_mode, w_neg=w_neg, w_reg=w_reg
    )
    # =======================================================


    optimizer = torch.optim.SGD(node_origin.model.parameters(), lr=node_origin.args.lr, momentum=0.9)
    dataset_iter = iter(node_origin.dataset) if node_origin.is_leaf() else None

    node_origin.model.train()
    node_neigh.model.eval()
    node_origin.prev_model.eval()

    # 收集每个类别的平方损失，用于更新 ClassUtilityAllocator
    class_squared_loss_accum = defaultdict(list)

    try:
        for start in range(0, len(noises_tensor), batch_size):
            end = min(start + batch_size, len(noises_tensor))
            b_noises = noises_tensor[start:end]
            b_labels = labels_tensor[start:end]

            if b_noises.shape[0] < 2:
                continue

            optimizer.zero_grad()
            fake_data = node_neigh.autoencoder.decoder(b_noises)

            # 动态判断 CIFAR-10 或 CIFAR-100 进行再归一化
            if getattr(args, 'dataset', 'cifar10') == 'cifar100':
                mean = torch.tensor([0.50707516, 0.48654887, 0.44091784], device=fake_data.device).view(1, 3, 1, 1)
                std = torch.tensor([0.26733429, 0.25643846, 0.27615047], device=fake_data.device).view(1, 3, 1, 1)
            else:
                mean = torch.tensor([0.49139968, 0.48215841, 0.44653091], device=fake_data.device).view(1, 3, 1, 1)
                std = torch.tensor([0.24703223, 0.24348513, 0.26158784], device=fake_data.device).view(1, 3, 1, 1)

            fake_data_norm = (fake_data - mean) / std

            with torch.no_grad():
                nei_logits, _, z_teacher = node_neigh.model(fake_data_norm)

            logits_fake, _, z_student = node_origin.model(fake_data_norm.detach())

            # ===== 核心恢复 1：提取假数据的上一轮负样本特征 =====
            with torch.no_grad():
                if ablation_mode == 'reg_ema':
                    _, _, z_hist = node_origin.ema_model(fake_data_norm.detach())
                else:
                    _, _, z_hist = node_origin.prev_model(fake_data_norm.detach())
            # 按样本计算 CE 损失并按类别聚合（存储 Loss^2）
            with torch.no_grad():
                per_sample_loss = ce_loss_crit_per_sample(logits_fake.detach(), b_labels.long())
                for i in range(b_labels.shape[0]):
                    c = b_labels[i].item()
                    squared_loss = per_sample_loss[i].item() ** 2
                    class_squared_loss_accum[c].append(squared_loss)

            loss = 0.0
            if node_origin.is_leaf():
                # 1. 假数据：端设备只计算蒸馏 KL Loss
                loss_distill = crit_non_leaf.alpha * crit_non_leaf.kl_loss_crit(logits_fake, nei_logits)

                if dataset_iter is not None:
                    try:
                        img, label_true = next(dataset_iter)
                        img, label_true = img.cuda(), label_true.cuda()

                        # 2. 提取真数据在三个模型下的特征和 logits
                        logits_true, _, z_student_true = node_origin.model(img)
                        with torch.no_grad():
                            _, _, z_teacher_true = node_neigh.model(img)
                            if ablation_mode == 'reg_ema':
                                _, _, z_hist_true = node_origin.ema_model(img)
                            else:
                                _, _, z_hist_true = node_origin.prev_model(img)

                        # 3. 计算真数据上的 CE Loss
                        loss_ce = ce_loss_crit_mean(logits_true, label_true.long())

                        # 4. 汇总：先加上假数据蒸馏 + 真数据分类
                        loss = loss_distill + loss_ce

                        # 如果过了预热期，再加上真数据对比损失（带负样本 z_prev_true）
                        if use_contrastive:
                            # 修改传入参数：z_prev_true -> z_hist_true
                            loss_con_true = contrastive_crit(z_student_true, z_teacher_true, z_hist_true)
                            loss = loss + mu_con * loss_con_true

                    except StopIteration:
                        dataset_iter = None
                        loss = loss_distill
                else:
                    loss = loss_distill
            else:
                # 假数据计算 (非叶子节点)
                loss = crit_non_leaf(logits_fake, nei_logits, b_labels.long(),
                                     z_student=z_student,
                                     z_teacher=z_teacher,
                                     z_prev=z_hist,  # ★ 修改：传入 z_hist
                                     use_contrastive=use_contrastive)

            loss.backward()
            optimizer.step()
    finally:
        node_neigh.model.train()

    # 返回每类的 mean(Loss^2)
    class_mean_squared_losses = {}
    for c, sq_losses in class_squared_loss_accum.items():
        class_mean_squared_losses[c] = sum(sq_losses) / len(sq_losses)

    return class_mean_squared_losses

#只包含正样本的对比学习
# def BSBODP_dir(node_origin, node_neigh, args, comm_round):
#     batch_size = args.batch_size
#     con_warmup = getattr(args, 'con_warmup', 20)
#     # 只有当前轮次大于等于预热阈值时，才开启对比学习
#     use_contrastive = (comm_round >= con_warmup)
#
#     len_neigh = sum(b.shape[0] for b in node_neigh.noises) if len(node_neigh.noises) > 0 else 0
#     len_origin = sum(b.shape[0] for b in node_origin.noises) if len(node_origin.noises) > 0 else 0
#
#     noises = node_neigh.noises if len_neigh < len_origin else node_origin.noises
#     labels = node_neigh.labels if len_neigh < len_origin else node_origin.labels
#
#     if len(noises) == 0:
#         return {}
#
#     noises_tensor = torch.cat(noises, dim=0)
#     labels_tensor = torch.cat(labels, dim=0)
#
#     T_agg = args.T_agg
#     if isinstance(T_agg, list):
#         T_agg = T_agg[0]
#
#     mu_con = getattr(args, 'mu_con', 1.0)
#     con_temperature = getattr(args, 'con_temperature', 0.5)
#
#     crit_non_leaf = Loss_Non_Leaf(T_agg, mu_con=mu_con, con_temperature=con_temperature)
#
#     # ★ 缺陷 2 修正：需要两个 CE Loss —— 一个 per-sample 用于统计，一个 mean 用于反向传播
#     ce_loss_crit_per_sample = nn.CrossEntropyLoss(reduction='none')  # 逐样本，不求均值
#     ce_loss_crit_mean = nn.CrossEntropyLoss()                        # 标准均值，用于叶子节点反向传播
#     contrastive_crit = ContrastiveLoss(con_temperature)
#
#     optimizer = torch.optim.SGD(node_origin.model.parameters(), lr=node_origin.args.lr, momentum=0.9)
#     dataset_iter = iter(node_origin.dataset) if node_origin.is_leaf() else None
#
#     node_origin.model.train()
#     node_neigh.model.eval()
#     node_origin.prev_model.eval()
#
#     # ★ 新增：收集每个类别的**平方损失**，用于更新 ClassUtilityAllocator
#     class_squared_loss_accum = defaultdict(list)
#
#     try:
#         for start in range(0, len(noises_tensor), batch_size):
#             end = min(start + batch_size, len(noises_tensor))
#             b_noises = noises_tensor[start:end]
#             b_labels = labels_tensor[start:end]
#
#             if b_noises.shape[0] < 2:
#                 continue
#
#             optimizer.zero_grad()
#             fake_data = node_neigh.autoencoder.decoder(b_noises)
#
#             # ====== 动态判断 CIFAR-10 或 CIFAR-100 进行再归一化 ======
#             if getattr(args, 'dataset', 'cifar10') == 'cifar100':
#                 mean = torch.tensor([0.50707516, 0.48654887, 0.44091784], device=fake_data.device).view(1, 3, 1, 1)
#                 std = torch.tensor([0.26733429, 0.25643846, 0.27615047], device=fake_data.device).view(1, 3, 1, 1)
#             else:
#                 mean = torch.tensor([0.49139968, 0.48215841, 0.44653091], device=fake_data.device).view(1, 3, 1, 1)
#                 std = torch.tensor([0.24703223, 0.24348513, 0.26158784], device=fake_data.device).view(1, 3, 1, 1)
#
#             fake_data_norm = (fake_data - mean) / std
#
#             with torch.no_grad():
#                 nei_logits, _, z_teacher = node_neigh.model(fake_data_norm)
#
#             logits_fake, _, z_student = node_origin.model(fake_data_norm.detach())
#
#
#             # ★ 缺陷 2 修正：按样本计算 CE 损失并按类别聚合（存储 Loss^2）
#             # 严格对齐 Oort 论文: U(i) = |B_i| * sqrt( (1/|B_i|) * Σ Loss(k)^2 )
#             with torch.no_grad():
#                 per_sample_loss = ce_loss_crit_per_sample(logits_fake.detach(), b_labels.long())
#                 for i in range(b_labels.shape[0]):
#                     c = b_labels[i].item()
#                     # ★ Oort 统计效用基于 Loss 的平方，这里存入 Loss^2
#                     squared_loss = per_sample_loss[i].item() ** 2
#                     class_squared_loss_accum[c].append(squared_loss)
#
#             loss = 0.0
#             if node_origin.is_leaf():
#                 # 1. 假数据：端设备只计算蒸馏 KL Loss
#                 loss_distill = crit_non_leaf.alpha * crit_non_leaf.kl_loss_crit(logits_fake, nei_logits)
#
#                 if dataset_iter is not None:
#                     try:
#                         img, label_true = next(dataset_iter)
#                         img, label_true = img.cuda(), label_true.cuda()
#
#                         # 2. 提取真数据在三个模型下的特征和 logits
#                         logits_true, _, z_student_true = node_origin.model(img)
#                         with torch.no_grad():
#                             _, _, z_teacher_true = node_neigh.model(img)
#
#
#                         # 3. 计算真数据上的 CE Loss 和 对比损失 (Contrastive Loss)
#                         loss_ce = ce_loss_crit_mean(logits_true, label_true.long())
#
#                         # 4. 汇总：先加上假数据蒸馏 + 真数据分类
#                         loss = loss_distill + loss_ce
#
#                         # 如果过了预热期，再加上真数据对比损失
#                         if use_contrastive:
#                             loss_con_true = contrastive_crit(z_student_true, z_teacher_true)
#                             loss = loss + mu_con * loss_con_true
#
#                     except StopIteration:
#                         dataset_iter = None
#                         # 如果本地真实数据在这个 epoch 用完了，就只计算假数据的蒸馏损失
#                         loss = loss_distill
#                 else:
#                     # 没有真实数据时，只计算假数据蒸馏损失
#                     loss = loss_distill
#             else:
#                 # 传入 use_contrastive 标志位，控制是否激活对比损失
#                 loss = crit_non_leaf(logits_fake, nei_logits, b_labels.long(),
#                                      z_student=z_student,
#                                      z_teacher=z_teacher,
#                                      use_contrastive=use_contrastive)
#
#             loss.backward()
#             optimizer.step()
#     finally:
#         node_neigh.model.train()
#
#     # ★ 返回每类的 mean(Loss^2)，后续 Allocator 中 sqrt 后即为 RMS
#     class_mean_squared_losses = {}
#     for c, sq_losses in class_squared_loss_accum.items():
#         class_mean_squared_losses[c] = sum(sq_losses) / len(sq_losses)
#
#     return class_mean_squared_losses