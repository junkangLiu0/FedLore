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

import numpy as np
import matplotlib.pyplot as plt

from DomainNet import DomainNet


#dataset_path = './data/cola'
#dataset = load_dataset(dataset_path)
#train_dataset = dataset["train"]

def find_cls(inter_sum, rnd):
    for i in range(len(inter_sum)):
        if rnd<inter_sum[i]:
            break
    return i - 1

def get_tag(data_name):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if data_name == 'SVHN':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
        ])
        train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)


    if data_name == 'FashionMNIST':
        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset= datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    if data_name =='CIFAR10':
        train_dataset = datasets.CIFAR10(
            "./data",
            train=True,
            download=True,
            transform=transform_train)
    elif data_name == 'CIFAR100':
        train_dataset = datasets.cifar.CIFAR100(
            "./data",
            train=True,
            download=True,
            transform=transform_train)
    elif data_name =='EMNIST1':
        train_dataset = datasets.EMNIST(
            "./data",
            split='byclass',
            train=True,
            download=True,
            transform=transforms.ToTensor())
    elif data_name =='EMNIST':
        train_dataset = datasets.EMNIST(
            "./data",
            #split='mnist',
            split='balanced',
            #split='byclass',
            train=True,
            download=True,
            transform=transforms.ToTensor())
    elif data_name =='MNIST':
        train_dataset = datasets.EMNIST(
            "./data",
            split='mnist',
            #split='balanced',
            #split='byclass',
            train=True,
            download=True,
            transform=transforms.ToTensor())
    if data_name == 'tiny-imagenet':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821]),
        ])
        train_dataset = local_datasets.TinyImageNetDataset(
            root=os.path.join('./data', 'tiny_imagenet'),
            split='train',
            transform=transform_train
        )
    if data_name == 'imagenet':
        transform_train = transforms.Compose([
            transforms.RandomRotation(10),  # RandomRotation 추가
            transforms.RandomCrop(64, padding=4),
            transforms.RandomResizedCrop((224, 224)),
            # resize 256_comb_coteach_OpenNN_CIFAR -> random_crop 224 ==> crop 32, padding 4
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821]),
        ])
        train_dataset = local_datasets.TinyImageNetDataset(
            root=os.path.join('./data', 'tiny-imagenet-200'),
            split='train',
            transform=transform_train
        )

    if data_name.startswith("domainnet"):
        data_dir = './data/dominnet'
        domain_name = data_name.split("_", 1)[1] if "_" in data_name else "real"
        # 你的路径别再写死 dominnet（容易拼写错），建议统一成 domainnet
        root = "./data/domainnet"
        if not os.path.isdir(root):
            root = "./data/dominnet"  # 兼容你之前的拼写
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = DomainNet(root=root, domain=domain_name, train=True,  transform=transform_train)
        test_dataset  = DomainNet(root=root, domain=domain_name, train=False, transform=transform_train)
        print("Train size:", len(train_dataset))
        print("Test  size:", len(test_dataset))
        print()

    if data_name == 'sst2':
        dataset_path = './data/sst2'
        dataset = load_dataset(dataset_path)
        train_dataset = dataset["train"]
        #train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    if data_name == 'QQP':
        dataset_path = './data/QQP'
        dataset = load_dataset(dataset_path)
        train_dataset = dataset["train"]


    if data_name == 'MNLI':
        dataset_path = './data/MNLI'
        dataset = load_dataset(dataset_path)
        train_dataset = dataset["train"]

    if data_name == 'STS-B':
        dataset_path = './data/sts-b'
        dataset = load_dataset(dataset_path)
        train_dataset = dataset["train"]

    if data_name == 'WNLI':
        dataset_path = './data/WNLI'
        dataset = load_dataset(dataset_path)
        train_dataset = dataset["train"]

    if data_name == 'RTE':
        dataset_path = './data/RTE'
        dataset = load_dataset(dataset_path)
        train_dataset = dataset["train"]

    if data_name == 'MRPC':
        dataset_path = './data/MRPC'
        dataset = load_dataset(dataset_path)
        train_dataset = dataset["train"]

    if data_name == 'qnli':
        dataset_path = './data/qnli'
        dataset = load_dataset(dataset_path)
        train_dataset = dataset["train"]
    if data_name == 'cola':
        dataset_path = './data/cola'
        dataset = load_dataset(dataset_path)
        dataset = dataset.rename_column("Acceptability", "label")
        train_dataset = dataset["train"]
    if data_name == 'SNLI':
        dataset_path = './data/SNLI'
        dataset = load_dataset(dataset_path)
        train_dataset = dataset["train"]
        train_dataset = train_dataset.filter(lambda example: example["label"] != -1)
    if data_name == 'AG_News':
        dataset_path = './data/AG_News'
        dataset = load_dataset(dataset_path)
        train_dataset = dataset["train"]
    if data_name == 'DBPedia_14':
        dataset_path = './data/DBPedia_14'
        dataset = load_dataset(dataset_path)
        train_dataset = dataset["train"]
        #train_dataset = train_dataset.filter(lambda example: example["label"] != -1)
    if data_name == 'IMDB':
        dataset_path = './data/IMDB'
        dataset = load_dataset(dataset_path)
        train_dataset = dataset["train"]
    if data_name == 'ANLI':
        dataset_path = './data/ANLI'
        dataset = load_dataset(dataset_path)
        train_dataset = dataset["train"]



    if data_name in['sst2','qnli','MRPC','RTE','WNLI','STS-B','MNLI','STS-B','cola', 'QQP','SNLI','AG_News','DBPedia_14','IMDB','ANLI']:
        id2targets = [train_dataset[i]['label'] for i in range(len(train_dataset))]
    else:
        id2targets =[train_dataset[i][1] for i in range(len(train_dataset))]
    targets = np.array(id2targets)
    # counter = Counter(targets)
    # print(counter)
    sort_index = np.argsort(targets)

    return id2targets, sort_index


import numpy as np
from collections import Counter
from typing import List, Tuple

def data_from_dirichlet(
        data_name: str,
        alpha_value: float,
        nums_cls: int,
        nums_wk: int,
        nums_sample: int,
        seed: int = 42  # ← 新增参数
) -> Tuple[List[List[int]], List[float]]:
    """
    Deterministic non-IID split with Dirichlet distribution.

    Args:
        data_name   : 数据集名称（用于 get_tag）
        alpha_value : Dirichlet α
        nums_cls    : 类别数
        nums_wk     : 客户端数
        nums_sample : 每客户端样本数
        seed        : 随机种子（默认 42）

    Returns:
        data_idx : list[list[int]]  – 每个客户端的样本索引
        std_list : list[float]      – 每个客户端标签计数的标准差
    """
    # --- 0. 独立随机数生成器，保证不污染全局 ---
    rng = np.random.default_rng(seed)

    # --- 1. 按类别建立索引池 ---
    id2targets, sorted_indices = get_tag(data_name)         # 自定义工具
    class_indices = [[] for _ in range(nums_cls)]
    for idx in sorted_indices:                              # O(N)
        class_indices[id2targets[idx]].append(idx)
    for lst in class_indices:                               # 仅打乱一次
        rng.shuffle(lst)
    ptr = np.zeros(nums_cls, dtype=int)                     # 每类指针

    # --- 2. 采 Dirichlet 概率矩阵 ---
    dirichlet_mat = rng.dirichlet([alpha_value] * nums_cls, size=nums_wk)

    # --- 3. 按客户端批量采样 ---
    #import numpy as np

    # 预先计算哪些类可用
    class_sizes = np.array([len(v) for v in class_indices], dtype=np.int64)
    valid_mask = class_sizes > 0
    valid_cls = np.nonzero(valid_mask)[0]

    if len(valid_cls) == 0:
        raise ValueError("All classes are empty! Check dataset/label mapping.")

    data_idx = []
    for j in range(nums_wk):
        # 只在有效类上采样：将空类概率置 0 后归一化
        p = np.array(dirichlet_mat[j], dtype=np.float64)
        p[~valid_mask] = 0.0
        s = p.sum()
        if s <= 0:
            # 极端情况：该客户端概率全落在空类上 → 退化成均匀采样有效类
            p = None
            cls_samples = rng.choice(valid_cls, size=nums_sample, replace=True)
        else:
            p = p / s
            cls_samples = rng.choice(nums_cls, size=nums_sample, p=p)

        client_indices = []
        for cls in cls_samples:
            # cls 一定是有效类（有数据）
            if len(class_indices[cls]) == 0:
                # 双保险（理论上不会走到）
                continue
            pcur = ptr[cls]
            if pcur >= len(class_indices[cls]):
                rng.shuffle(class_indices[cls])
                ptr[cls] = 0
                pcur = 0
            client_indices.append(class_indices[cls][pcur])
            ptr[cls] += 1

        # 若因为 continue 导致不足 nums_sample，可补齐（可选）
        while len(client_indices) < nums_sample:
            cls = int(rng.choice(valid_cls))
            pcur = ptr[cls]
            if pcur >= len(class_indices[cls]):
                rng.shuffle(class_indices[cls])
                ptr[cls] = 0
                pcur = 0
            client_indices.append(class_indices[cls][pcur])
            ptr[cls] += 1

        data_idx.append(client_indices)

    # --- 4. 统计标签分布标准差 ---
    std_list = []
    for indices in data_idx:
        labels = [id2targets[i] for i in indices]
        counts = np.bincount(labels, minlength=nums_cls)
        std_list.append(counts.std(ddof=0))

    # 可选：打印前 5 个客户端的标签直方图
    for idx, samples in enumerate(data_idx[:min(5, nums_wk)]):
        print(f'Client {idx} label histogram →',
              Counter([id2targets[i] for i in samples]))
    print('mean label std:', np.mean(std_list))
    return data_idx, std_list


