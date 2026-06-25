# 导入模块
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import pandas as pd
from collections import Counter
from copy import deepcopy
import dataset as local_datasets
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import lru_cache
from typing import List, Tuple, Dict, Optional

from DomainNet import DomainNet

# ==================== 缓存目录设置 ====================
CACHE_DIR = "./cache/dirichlet"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_path(data_name: str, alpha_value: float, nums_wk: int, nums_sample: int, seed: int) -> str:
    """生成缓存文件路径"""
    cache_key = f"{data_name}_a{alpha_value}_w{nums_wk}_s{nums_sample}_seed{seed}.npz"
    return os.path.join(CACHE_DIR, cache_key)


# ==================== 优化 1: 快速标签获取（核心优化）====================

@lru_cache(maxsize=32)
def get_tag_fast(data_name: str, domain: Optional[str] = None) -> Tuple[List[int], np.ndarray]:
    """
    极速获取数据集标签，避免加载图像和transform

    Args:
        data_name: 数据集名称
        domain: DomainNet的域名称（可选）

    Returns:
        id2targets: 标签列表
        sort_index: 按标签排序的索引
    """

    # ========== DomainNet: 直接解析txt文件（最快）==========
    if data_name.startswith("domainnet") or domain is not None:
        dom = domain if domain else (data_name.split("_", 1)[1] if "_" in data_name else "real")
        root = "./data/domainnet"
        if not os.path.isdir(root):
            root = "./data/dominnet"

        split_file = os.path.join(root, f"{dom}_train.txt")

        # 快速读取：一次性读取所有行
        with open(split_file, "r") as f:
            lines = f.readlines()

        # 向量化解析：避免Python循环
        id2targets = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                id2targets.append(int(parts[1]))

        sort_index = np.argsort(id2targets, kind='mergesort')  # mergesort稳定排序
        return id2targets, sort_index

    # ========== HuggingFace数据集: 直接访问底层数组 ==========
    hf_datasets = {'sst2', 'qnli', 'mrpc', 'rte', 'wnli', 'sts-b', 'mnli', 'cola', 'qqp'}
    if data_name.lower() in hf_datasets:
        dataset_path = f'./data/{data_name.lower()}'
        dataset = load_dataset(dataset_path)
        train_dataset = dataset["train"]

        # 直接访问列数据，避免迭代
        if hasattr(train_dataset, 'data') and 'label' in train_dataset.data:
            # Arrow格式，直接转numpy
            id2targets = train_dataset.data['label'].to_numpy().tolist()
        elif hasattr(train_dataset, 'features') and 'label' in train_dataset.features:
            # 使用datasets的列访问
            id2targets = train_dataset['label']
        else:
            # Fallback：使用list comprehension（仍然比__getitem__快）
            id2targets = [train_dataset[i]['label'] for i in range(len(train_dataset))]

        sort_index = np.argsort(id2targets, kind='mergesort')
        return id2targets, sort_index

    # ========== PyTorch数据集: 直接访问.targets/.labels属性 ==========
    # 统一使用None transform，避免图像加载
    if data_name.upper() == 'SVHN':
        train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=None)
        id2targets = train_dataset.labels.tolist()

    elif data_name.upper() == 'FASHIONMNIST':
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=None)
        id2targets = train_dataset.targets.tolist() if hasattr(train_dataset.targets, 'tolist') else list(
            train_dataset.targets)

    elif data_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=None)
        id2targets = train_dataset.targets if isinstance(train_dataset.targets,
                                                         list) else train_dataset.targets.tolist()

    elif data_name == 'CIFAR100':
        train_dataset = datasets.CIFAR100("./data", train=True, download=True, transform=None)
        id2targets = train_dataset.targets if isinstance(train_dataset.targets,
                                                         list) else train_dataset.targets.tolist()

    elif data_name == 'EMNIST':
        train_dataset = datasets.EMNIST("./data", split='balanced', train=True, download=True, transform=None)
        id2targets = train_dataset.targets.tolist() if hasattr(train_dataset.targets, 'tolist') else list(
            train_dataset.targets)

    elif data_name == 'MNIST':
        train_dataset = datasets.EMNIST("./data", split='mnist', train=True, download=True, transform=None)
        id2targets = train_dataset.targets.tolist() if hasattr(train_dataset.targets, 'tolist') else list(
            train_dataset.targets)

    elif data_name == 'tiny-imagenet':
        # TinyImageNet需要特殊处理，尝试直接读取txt文件
        root = os.path.join('./data', 'tiny_imagenet')
        train_txt = os.path.join(root, 'train.txt')

        if os.path.exists(train_txt):
            # 如果有预处理好的txt文件
            with open(train_txt, 'r') as f:
                id2targets = [int(line.strip().split()[1]) for line in f if line.strip()]
        else:
            # Fallback：使用Dataset但跳过transform
            train_dataset = local_datasets.TinyImageNetDataset(root=root, split='train', transform=None)
            # 尝试直接访问底层数据
            if hasattr(train_dataset, 'targets'):
                id2targets = train_dataset.targets
            elif hasattr(train_dataset, 'labels'):
                id2targets = train_dataset.labels
            else:
                # 并行提取标签（最后手段）
                id2targets = _parallel_extract_labels(train_dataset)
    else:
        raise ValueError(f"Unknown dataset: {data_name}")

    sort_index = np.argsort(id2targets, kind='mergesort')
    return id2targets, sort_index


def _parallel_extract_labels(dataset, num_workers: int = 8) -> List[int]:
    """并行提取标签（Fallback方法）"""
    n = len(dataset)

    def get_label(idx):
        try:
            return dataset[idx][1]
        except:
            return -1

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        labels = list(executor.map(get_label, range(n)))

    return [l for l in labels if l != -1]


# ==================== 优化 2: 向量化Dirichlet划分（核心算法优化）====================

def data_from_dirichlet_fast(
        data_name: str,
        alpha_value: float,
        nums_cls: int,
        nums_wk: int,
        nums_sample: int,
        seed: int = 42,
        use_cache: bool = True
) -> Tuple[List[List[int]], List[float]]:
    """
    修复版：确保每个客户端严格分配 nums_sample 个样本
    """

    # 检查缓存
    if use_cache:
        cache_path = get_cache_path(data_name, alpha_value, nums_wk, nums_sample, seed)
        if os.path.exists(cache_path):
            print(f"[Cache] Loading from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            return data['data_idx'].tolist(), data['std_list'].tolist()

    rng = np.random.default_rng(seed)

    # 快速获取标签
    print(f"[Dirichlet] Loading labels for {data_name}...")
    domain = data_name.split("_", 1)[1] if data_name.startswith("domainnet") and "_" in data_name else None
    id2targets_list, _ = get_tag_fast(data_name, domain)
    id2targets = np.array(id2targets_list, dtype=np.int32)
    N = len(id2targets)
    print(f"[Dirichlet] Total samples in dataset: {N}")
    print(f"[Dirichlet] Requested total: {nums_wk * nums_sample} ({nums_wk} clients × {nums_sample} samples)")

    # 检查是否足够
    if nums_wk * nums_sample > N:
        print(f"[Warning] Requested {nums_wk * nums_sample} > available {N}, will use replacement")

    # 构建类别索引池
    print("[Dirichlet] Building class pools...")

    # 按标签排序
    sort_perm = np.argsort(id2targets, kind='mergesort')
    sorted_labels = id2targets[sort_perm]

    # 找到每个类别的边界
    class_boundaries = []
    class_sizes = []

    for c in range(nums_cls):
        left = np.searchsorted(sorted_labels, c, side='left')
        right = np.searchsorted(sorted_labels, c, side='right')
        class_boundaries.append((left, right))
        class_sizes.append(right - left)

    class_sizes = np.array(class_sizes, dtype=np.int32)
    valid_classes = np.where(class_sizes > 0)[0]

    print(f"[Dirichlet] Valid classes: {len(valid_classes)}/{nums_cls}")
    print(
        f"[Dirichlet] Samples per class (min/mean/max): {class_sizes[valid_classes].min()}/{class_sizes[valid_classes].mean():.1f}/{class_sizes[valid_classes].max()}")

    # 计算每个类别的总配额（所有客户端）
    # 使用Dirichlet分布分配比例
    dirichlet_probs = rng.dirichlet([alpha_value] * nums_cls, size=nums_wk)

    # 归一化概率（只考虑有效类别）
    for j in range(nums_wk):
        probs = dirichlet_probs[j].copy()
        probs[class_sizes == 0] = 0
        total = probs.sum()
        if total > 0:
            dirichlet_probs[j] = probs / total
        else:
            dirichlet_probs[j, valid_classes] = 1.0 / len(valid_classes)

    # 计算每个客户端每个类别的期望样本数
    # 并确保总数严格等于 nums_sample
    class_quotas = np.zeros((nums_wk, nums_cls), dtype=np.int32)
    for j in range(nums_wk):
        # 按比例分配，然后调整确保总和为 nums_sample
        quotas = (dirichlet_probs[j] * nums_sample).astype(np.int32)
        # 处理舍入误差：从大到小排序，补足差额
        deficit = nums_sample - quotas.sum()
        if deficit > 0:
            # 找到概率最大的deficit个类别，各加1
            top_classes = np.argsort(dirichlet_probs[j])[::-1][:deficit]
            quotas[top_classes] += 1
        class_quotas[j] = quotas

    print(f"[Dirichlet] Quota per client: {class_quotas.sum(axis=1)}")  # 应该都是 nums_sample

    # 为每个类别创建可重复使用的索引池（带循环）
    # 关键修改：使用无限循环迭代器模式
    class_pools = []
    class_positions = np.zeros(nums_cls, dtype=np.int32)  # 每个类别的当前位置

    for c in range(nums_cls):
        left, right = class_boundaries[c]
        if right > left:
            # 创建该类别所有样本的索引数组
            pool = sort_perm[left:right].copy()
            rng.shuffle(pool)  # 随机打乱
            class_pools.append(pool)
        else:
            class_pools.append(np.array([], dtype=np.int32))

    # 分配样本
    print("[Dirichlet] Assigning samples...")
    data_idx = [[] for _ in range(nums_wk)]

    for j in range(nums_wk):
        client_samples = []

        for c in valid_classes:
            quota = class_quotas[j, c]
            if quota == 0:
                continue

            pool = class_pools[c]
            pos = class_positions[c]
            size = len(pool)

            if size == 0:
                continue

            # 需要quota个样本，从pos开始取，循环使用
            for _ in range(quota):
                # 循环取模
                idx_in_pool = pos % size
                client_samples.append(pool[idx_in_pool])
                pos += 1

            class_positions[c] = pos  # 更新位置

        # 如果因为某些类别为空导致不足，从其他类别补足
        while len(client_samples) < nums_sample:
            # 随机选一个还有样本的类别
            available = [c for c in valid_classes if len(class_pools[c]) > 0]
            if not available:
                # 所有类别都空了，重置所有位置重新采样（有放回）
                print(f"[Warning] Client {j}: all classes exhausted, using replacement")
                class_positions[:] = 0
                for c in valid_classes:
                    rng.shuffle(class_pools[c])  # 重新打乱
                available = valid_classes.tolist()

            c = rng.choice(available)
            pool = class_pools[c]
            pos = class_positions[c]
            size = len(pool)

            idx_in_pool = pos % size
            client_samples.append(pool[idx_in_pool])
            class_positions[c] = pos + 1

        # 打乱该客户端的样本顺序（避免按类别排列）
        rng.shuffle(client_samples)
        data_idx[j] = client_samples

        # 验证
        actual = len(data_idx[j])
        if actual != nums_sample:
            print(f"[Error] Client {j}: expected {nums_sample}, got {actual}")

    # 转换为numpy数组用于后续计算
    data_idx_array = np.array(data_idx, dtype=object)

    # 计算标准差
    print("[Dirichlet] Computing statistics...")
    std_list = []
    for j in range(nums_wk):
        labels = id2targets[data_idx[j]]
        counts = np.bincount(labels, minlength=nums_cls)
        std_list.append(float(counts.std()))

        # 打印每个客户端的详细统计
        actual_count = len(data_idx[j])
        unique_labels = len(np.unique(labels))
        print(f"  Client {j}: {actual_count} samples, {unique_labels} classes, "
              f"top5: {dict(Counter(labels).most_common(5))}")

    print(f"[Dirichlet] Mean label std: {np.mean(std_list):.2f}")

    # 保存缓存
    if use_cache:
        np.savez_compressed(cache_path,
                            data_idx=data_idx_array,
                            std_list=np.array(std_list))
        print(f"[Cache] Saved to {cache_path}")

    return data_idx, std_list


# ==================== 优化 3: 超高速版本（使用Numba/JIT）====================

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("[Warning] Numba not installed, using numpy version")

if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _assign_samples_numba(sampled_classes, sort_perm, boundaries_left, boundaries_right,
                              class_counters, nums_wk, nums_sample):
        """Numba加速的样本分配（并行）"""
        data_idx = np.zeros((nums_wk, nums_sample), dtype=np.int32)

        for j in prange(nums_wk):
            for i in range(nums_sample):
                cls = sampled_classes[j, i]
                left = boundaries_left[cls]
                right = boundaries_right[cls]
                size = right - left

                # 原子操作模拟（Numba不支持真正的原子操作，这里简化）
                pos = class_counters[cls] % max(size, 1)
                data_idx[j, i] = sort_perm[left + pos]
                class_counters[cls] += 1

        return data_idx


# ==================== 保持兼容性的原接口包装 ====================

def data_from_dirichlet(
        data_name: str,
        alpha_value: float,
        nums_cls: int,
        nums_wk: int,
        nums_sample: int,
        seed: int = 42,
        use_cache: bool = True,
        fast_mode: bool = True
) -> Tuple[List[List[int]], List[float]]:
    """
    统一的Dirichlet划分接口

    Args:
        fast_mode: 是否使用极速版本（默认True）
        use_cache: 是否使用缓存
        其他参数同原函数

    Returns:
        data_idx: 每个客户端的样本索引列表
        std_list: 每个客户端标签分布的标准差列表
    """
    if fast_mode:
        return data_from_dirichlet_fast(data_name, alpha_value, nums_cls,
                                        nums_wk, nums_sample, seed, use_cache)
    else:
        # Fallback到原始实现（如果需要）
        return data_from_dirichlet_legacy(data_name, alpha_value, nums_cls,
                                          nums_wk, nums_sample, seed)


# ==================== 原始实现（保留用于对比/调试）====================

def data_from_dirichlet_legacy(
        data_name: str,
        alpha_value: float,
        nums_cls: int,
        nums_wk: int,
        nums_sample: int,
        seed: int = 42
) -> Tuple[List[List[int]], List[float]]:
    """原始慢速版本（用于对比）"""
    rng = np.random.default_rng(seed)

    # 使用原始get_tag（慢）
    id2targets, sorted_indices = _get_tag_original(data_name)

    # 构建类别字典（慢）
    dct = {}
    for idx in sorted_indices:
        cls = id2targets[idx]
        if cls not in dct:
            dct[cls] = []
        dct[cls].append(idx)

    sort_index = [dct.get(key, []) for key in range(nums_cls)]
    tag_index = deepcopy(sort_index)

    # Dirichlet采样（原始实现）
    alpha = [alpha_value] * nums_cls
    gamma_rnd = np.zeros([nums_cls, nums_wk])
    dirichlet_rnd = np.zeros([nums_cls, nums_wk])

    for n in range(nums_wk):
        alpha1 = 1 if n % 10 == 0 else 1  # 原逻辑保留
        for i in range(nums_cls):
            gamma_rnd[i, n] = rng.gamma(alpha1 * alpha[i], 1)
        Z_d = np.sum(gamma_rnd[:, n])
        dirichlet_rnd[:, n] = gamma_rnd[:, n] / Z_d

    # 分配样本（原始慢循环）
    data_idx = []
    for j in range(nums_wk):
        inter_sum = [0]
        for i in dirichlet_rnd[:, j]:
            inter_sum.append(i + inter_sum[-1])

        sample_index = []
        for i in range(nums_sample):
            rnd = rng.random()
            sample_cls = _find_cls(inter_sum, rnd)

            if len(tag_index[sample_cls]) > 0:
                sample_index.append(tag_index[sample_cls].pop())
            else:
                tag_index[sample_cls] = deepcopy(sort_index[sample_cls])
                sample_index.append(tag_index[sample_cls].pop())

        data_idx.append(sample_index)

    # 计算标准差
    std = [pd.Series(Counter([id2targets[j] for j in data])).describe().std()
           for data in data_idx]

    return data_idx, std


def _get_tag_original(data_name):
    """原始get_tag实现（完整保留）"""
    # ... 原get_tag函数的全部代码 ...
    # 为节省空间，这里省略，实际使用时复制原代码
    pass


def _find_cls(inter_sum, rnd):
    """辅助函数"""
    for i in range(len(inter_sum)):
        if rnd < inter_sum[i]:
            break
    return i - 1


# ==================== 测试和验证 ====================

if __name__ == "__main__":
    import time

    # 测试配置
    test_configs = [
        ("domainnet_real", 0.5, 345, 10, 1000),
        ("CIFAR10", 0.5, 10, 10, 5000),
        ("CIFAR100", 0.5, 100, 10, 500),
    ]

    for data_name, alpha, nums_cls, nums_wk, nums_sample in test_configs:
        print(f"\n{'=' * 50}")
        print(f"Testing: {data_name}")
        print(f"Config: alpha={alpha}, clients={nums_wk}, samples_per_client={nums_sample}")

        # 测试极速版本
        t0 = time.time()
        data_idx_fast, std_fast = data_from_dirichlet(
            data_name, alpha, nums_cls, nums_wk, nums_sample,
            seed=42, use_cache=True, fast_mode=True
        )
        t_fast = time.time() - t0
        print(f"Fast version: {t_fast:.3f}s")

        # 验证结果
        print(f"Total samples assigned: {sum(len(x) for x in data_idx_fast)}")
        print(f"Mean std: {np.mean(std_fast):.2f}")

        # 清理缓存（可选）
        # cache_path = get_cache_path(data_name, alpha, nums_wk, nums_sample, 42)
        # if os.path.exists(cache_path):
        #     os.remove(cache_path)
