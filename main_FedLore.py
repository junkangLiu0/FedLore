import os
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import SubsetRandomSampler, Subset, random_split, DataLoader
import random
from copy import deepcopy
import ray
import argparse
from tensorboardX import SummaryWriter
from transformers import BertTokenizer, BertForSequenceClassification
from DomainNet import DomainNet
from dirichlet_data2 import data_from_dirichlet
from lora_SVD import aggregate_AB_then_SVD, aggregate_FRLORA
from lora_fair import apply_weights_lora_fair, apply_weights_lora_fair_CV
from models import ResNet18, ResNet18BN, ResNet10, ResNet10BN
from models.DeiTTiny import ViTForCIFAR, deit_tiny_256
from sam import SAM
os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"
from model import swin_tiny_patch4_window7_224 as swin_tiny
from model import swin_small_patch4_window7_224 as swin_small
from model import swin_large_patch4_window7_224_in22k as swin_large
from model import swin_base_patch4_window7_224_in22k as swin_base
from vit_model import vit_base_patch16_224_in21k as vit_B
from vit_model import vit_large_patch16_224_in21k as vit_L
from sam import SAM
import torch, gc
from peft import LoraConfig, get_peft_model, TaskType
gc.collect()
torch.cuda.empty_cache()
#python  new_lora.py --alg FLORA --lr 0.001 --data_name CIFAR100 --alpha_value 0.1 --alpha  0.9  --epoch 101  --extname CIFAR100 --lr_decay 0.98 --gamma 0.3 --CNN swin_tiny --E 1 --batch_size 16  --gpu 0 --p 2 --num_gpus_per 0.25 --selection 0.04 --print 0 --rho 0.1 --num_workers 100 --preprint 5 --lora 1 --r 16 --optimizer SGD
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--lg', default=1.0, type=float, help='learning rate')
parser.add_argument('--epoch', default=100, type=int, help='number of epochs to train')
parser.add_argument('--num_workers', default=100, type=int, help='#workers')
parser.add_argument('--batch_size', default=16, type=int, help='# batch_size')
parser.add_argument('--E', default=1, type=int, help='# batch_size')
parser.add_argument('--alg', default='FedMoment', type=str, help='alg')  # FedMoment cddplus cdd SCAF atte
parser.add_argument('--extname', default='EM', type=str, help='extra_name')
parser.add_argument('--gpu', default='0,1', type=str, help='use which gpus')
parser.add_argument('--lr_decay', default='0.99', type=float, help='lr_decay')
parser.add_argument('--data_name', default='imagenet', type=str, help='lr_decay')
parser.add_argument('--tau', default='0.01', type=float, help='only for FedAdam ')
parser.add_argument('--lr_ps', default='0.15', type=float, help='only for FedAdam ')
parser.add_argument('--alpha_value', default='0.6', type=float, help='for dirichlet')
parser.add_argument('--selection', default='0.06', type=float, help=' C')
parser.add_argument('--check', default=0, type=int, help=' if check')
parser.add_argument('--T_part', default=10, type=int, help=' for mom_step')
parser.add_argument('--alpha', default=1, type=float, help=' for mom_step')
parser.add_argument('--CNN', default='VIT-L', type=str, help=' for mom_step')
parser.add_argument('--gamma', default=0.9, type=float, help=' for mom_step')
parser.add_argument('--weights', type=str, default='./swin_tiny_patch4_window7_224.pth',
                    help='initial weights path')
# 是否冻结权重
parser.add_argument('--p', default=3, type=int, help=' for mom_step')
parser.add_argument('--datapath', type=str,
                    default="./data")
parser.add_argument('--num_gpus_per', default=0.5, type=float, help=' for mom_step')
parser.add_argument('--rho', default=0.1, type=float, help='rho')
parser.add_argument('--optimizer', default='SGD', type=str, help='SGD,AdamW')
parser.add_argument("--preprint", type=int, default=5, help="")
parser.add_argument("--R", type=int, default=1, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--lora", type=int, default=1, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--r", type=int, default=16, help="the perturbation radio for the SAM optimizer.")
parser.add_argument('--K', default=20, type=int, help='#workers')
parser.add_argument("--eps", type=float, default=1e-8, help="the perturbation radio for the SAM optimizer.")
parser.add_argument('--normalization', default='BN', type=str, help=' for mom_step')
args = parser.parse_args()
gpu_idx = args.gpu
print('gpu_idx', gpu_idx)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
num_gpus_per = args.num_gpus_per  # num_gpus_per = 0.16
num_gpus = len(gpu_idx.split(','))
data_name = args.data_name
CNN = args.CNN
if CNN in ['VIT-B', 'swin_tiny', 'swin_large', 'VIT-L', 'swin_small', 'swin_base']:
    lora_config = LoraConfig(
        r=args.r,  # 低秩矩阵的秩，通常在 4 到 64 之间[^18^]
        lora_alpha=args.r*2,  # 缩放参数，通常为 r 的 2 到 32 倍[^18^]
        lora_dropout=0.05,  # Dropout 比率，防止过拟合[^18^]
        bias="none",  # 不训练偏置项[^18^]
        task_type="IMAGE_CLASSIFICATION",  # 任务类型，根据具体任务选择[^18^]
        target_modules=['attn.qkv', 'attn.proj']  # 目标模块，根据模型结构指定[^18^]
    )
lora_config = LoraConfig(
    r=args.r,  # 低秩矩阵的秩，通常在 4 到 64 之间[^18^]
    lora_alpha=args.r*2,  # 缩放参数，通常为 r 的 2 到 32 倍[^18^]
    lora_dropout=0.05,  # Dropout 比率，防止过拟合[^18^]
    bias="none",  # 不训练偏置项[^18^]
    task_type="IMAGE_CLASSIFICATION",  # 任务类型，根据具体任务选择[^18^]
    target_modules=['attn.qkv', 'attn.proj']  # 目标模块，根据模型结构指定[^18^]
)

if CNN in ['VIT-B', 'swin_tiny', 'swin_large', 'VIT-L', 'swin_small', 'swin_base', 'resnet18pre', 'resnet50pre',
           'resnet101pre'] :
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图像大小调整为 ResNet-18 输入的大小
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    if data_name == 'imagenet':
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),  # 将图像大小调整为 ResNet-18 输入的大小
            transforms.ToTensor(),  # 转换为 Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])
    if data_name == 'CIFAR100' or data_name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
else:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )
    transform_test = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    if data_name == 'CIFAR10' or data_name == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        )
        transform_test = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    if data_name == 'imagenet':
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
        ])
        transform_test = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

import dataset as local_datasets

if data_name == 'imagenet':
    train_dataset = local_datasets.TinyImageNetDataset(
        root=os.path.join(args.datapath, 'tiny-imagenet-200'),
        split='train',
        transform=transform_train
    )

if data_name == 'CIFAR10':
    train_dataset = datasets.CIFAR10(
        "./data",
        train=True,
        download=False,
        transform=transform_train)

elif data_name == 'CIFAR100':
    train_dataset = datasets.cifar.CIFAR100(
        "./data",
        train=True,
        download=True,
        transform=transform_train
    )

elif data_name == 'Caltech256':
    from torchvision.datasets import Caltech256
    full_dataset = Caltech256(
        root="./data",
        #download=True,
        transform=transform_train
    )

    n = len(full_dataset)
    print(n)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    test_size = n - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )



if data_name.startswith("domainnet"):
    from PIL import Image

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    import os
    from PIL import Image
    from torch.utils.data import Dataset

    # 自定义DomainNet数据集类
    class DomainNetDataset(torch.utils.data.Dataset):
        def __init__(self, data_dir, domain, split='train', transform=None):
            self.data_dir = data_dir
            self.domain = domain
            self.split = split
            self.transform = transform
            data_dir='./data/dominnet'
            # 读取对应的txt文件
            txt_file = os.path.join(data_dir, f'{domain}_{split}.txt')
            self.samples = []
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            img_path = parts[0]
                            label = int(parts[1])
                            full_img_path = os.path.join(data_dir, img_path)
                            self.samples.append((full_img_path, label))
            print(f"Loaded {len(self.samples)} samples from {txt_file}")
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                image = Image.new('RGB', (224, 224), color='gray')
            if self.transform:
                image = self.transform(image)
            return image, label
    # 根据数据集名称确定域
    domain_name = args.data_name.split('_')[1]
    data_dir = './data/dominnet'
    domain_name = data_name.split("_", 1)[1] if "_" in data_name else "real"
    # 你的路径别再写死 dominnet（容易拼写错），建议统一成 domainnet
    root = "./data/domainnet"
    if not os.path.isdir(root):
        root = "./data/dominnet"  # 兼容你之前的拼写
    train_dataset = DomainNet(root=root, domain=domain_name, train=True,  transform=transform_train)
    test_dataset  = DomainNet(root=root, domain=domain_name, train=False, transform=transform_train)
    args.num_labels = 345  # DomainNet有345个类别

if args.alpha_value==1:
    generator = torch.Generator().manual_seed(42)
    total_size = len(train_dataset)
    print(total_size)
    subset_size = total_size // args.num_workers
    remainder = total_size % args.num_workers  # 计算剩余的样本数
    # 创建分割大小列表
    split_sizes = [subset_size] * (args.num_workers-1)+ [subset_size + remainder]
    subsets = random_split(train_dataset, split_sizes, generator=generator)

    def get_data_loader(pid, data_idx, batch_size, data_name):
        """Safely downloads data. Returns training/validation set dataloader. 使用到了外部的数据"""
        sample_chosed = data_idx[pid]
        train_sampler = SubsetRandomSampler(sample_chosed)
        train_loader = DataLoader(subsets[pid], batch_size=args.batch_size, shuffle=True)
        return train_loader

if args.alpha_value!=1:
    seed=42
    def get_data_loader(pid, data_idx, batch_size, data_name):
        """Safely downloads data. Returns training/validation set dataloader. 使用到了外部的数据"""
        sample_chosed = data_idx[pid]
        train_sampler = SubsetRandomSampler(sample_chosed)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler, num_workers=0, generator=torch.Generator().manual_seed(seed))
        return train_loader




def get_data_loader_test(data_name):
    """Safely downloads data. Returns training/validation set dataloader."""
    if data_name == 'imagenet':
        test_dataset = local_datasets.TinyImageNetDataset(
            root=os.path.join(args.datapath, 'tiny-imagenet-200'),
            split='test',
            transform=transform_train
        )
    if data_name == 'CIFAR10':
        test_dataset = datasets.CIFAR10("./data", train=False, transform=transform_train)

    elif data_name == 'CIFAR100':
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, transform=transform_train
                                               )
    if data_name.startswith("domainnet"):
        root = "./data/domainnet"
        if not os.path.isdir(root):
            root = "./data/dominnet"  # 兼容你之前的拼写
        test_dataset = DomainNet(root=root, domain=domain_name, train=False, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=200,
        shuffle=False,
        num_workers=4)
    return test_loader



def get_data_loader_train(data_name):
    """Safely downloads data. Returns training/validation set dataloader."""
    if data_name == 'imagenet':
        train_dataset = local_datasets.TinyImageNetDataset(
            root=os.path.join(args.datapath, 'tiny-imagenet-200'),
            split='train',
            transform=transform_train
        )
    if data_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10("./data", train=True, transform=transform_train)
        # test_dataset = datasets.cifar.CIFAR100("./data", train=False, transform=transform_test)
    elif data_name == 'CIFAR100':
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, transform=transform_train
                                               )
    if data_name.startswith("domainnet"):
        root = "./data/domainnet"
        if not os.path.isdir(root):
            root = "./data/dominnet"  # 兼容你之前的拼写
        train_dataset = DomainNet(root=root, domain=domain_name, train=True, transform=transform_train)
    train_dataset = torch.utils.data.Subset(train_dataset, range(1000))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=200,
        shuffle=False,
        num_workers=4)
    return train_loader



def evaluate(model, test_loader, train_loader):
    """Evaluates the accuracy of the model on a validation dataset."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    train_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            test_loss += criterion(outputs, target)

        for batch_idx, (data, target) in enumerate(train_loader):
            data_train = data.to(device)
            target_train = target.to(device)
            outputs_train = model(data_train)
            train_loss += criterion(outputs_train, target_train)
    model.to('cpu')
    torch.cuda.empty_cache()
    return 100. * correct / total, test_loss / len(test_loader), train_loss / len(train_loader)

if CNN == 'swin_base':
    def ConvNet():
        return swin_base(num_classes=10)
    def ConvNet100():
        return swin_base(num_classes=100)
    def ConvNet200():
        return swin_base(num_classes=200)

if CNN == 'VIT-B':
    def ConvNet():
        return vit_B(num_classes=10)
    def ConvNet100():
        return vit_B(num_classes=100)
    def ConvNet200():
        return vit_B(num_classes=200)
    def ConvNet345():
        return vit_B(num_classes=345)

if CNN == 'VIT-L':
    def ConvNet():
        return vit_L(num_classes=10)
    def ConvNet100():
        return vit_L(num_classes=100)
    def ConvNet200():
        return vit_L(num_classes=200)

if CNN == 'resnet10':
    if args.normalization == 'BN':
        def ConvNet(num_classes=10):
            return ResNet10BN(num_classes=10)
        def ConvNet100(num_classes=100):
            return ResNet10BN(num_classes=100)
        def ConvNet200(num_classes=200):
            return ResNet10BN(num_classes=200)
    if args.normalization == 'GN':
        def ConvNet(num_classes=10):
            return ResNet10(num_classes=10)
        def ConvNet100(num_classes=100):
            return ResNet10(num_classes=100)
        def ConvNet200(num_classes=200):
            return ResNet10(num_classes=200)


if CNN == 'resnet18':
    if args.normalization == 'BN':
        def ConvNet(num_classes=10, l2_norm=False):
            return ResNet18BN(num_classes=10)
        def ConvNet100(num_classes=100, l2_norm=False):
            return ResNet18BN(num_classes=100)
        def ConvNet200(num_classes=200, l2_norm=False):
            return ResNet18BN(num_classes=200)
    if args.normalization == 'GN':
        def ConvNet(num_classes=10):
            return ResNet18(num_classes=10)
        def ConvNet100(num_classes=100):
            return ResNet18(num_classes=100)
        def ConvNet200(num_classes=200):
            return ResNet18(num_classes=200)

if CNN == 'deit_tiny':
    def ConvNet(num_classes=10):
        return ViTForCIFAR(num_classes=10, img_size=32)
    def ConvNet100(num_classes=100):
        return ViTForCIFAR(num_classes=100, img_size=32)
    def ConvNet200(num_classes=200):
        return ViTForCIFAR(num_classes=200, img_size=64)

if CNN == 'deit_tiny_256':
    def ConvNet(num_classes=10):
        return deit_tiny_256(num_classes=10, img_size=32)
    def ConvNet100(num_classes=100):
        return deit_tiny_256(num_classes=100, img_size=32)
    def ConvNet200(num_classes=200):
        return deit_tiny_256(num_classes=200, img_size=64)

from src.cct import cct_7_3x1_32_c100
import torch
if CNN == 'cct':
    model = cct_7_3x1_32_c100(pretrained=False, progress=True, num_classes=100)
    def ConvNet(num_classes=10):
        return cct_7_3x1_32_c100(pretrained=False, progress=True, num_classes=10)
    def ConvNet100(num_classes=100):
        return cct_7_3x1_32_c100(pretrained=False, progress=True, num_classes=100)
    def ConvNet200(num_classes=200):
        return cct_7_3x1_32_c100(pretrained=False, progress=True, num_classes=200)
import math
import torch
from torch import nn


@ray.remote(num_gpus=num_gpus_per)
class DataWorker(object):
    def __init__(self, pid, data_idx, num_workers, lr, batch_size, alg, data_name, selection, T_part):
        self.alg = alg
        if data_name == 'imagenet':
            self.model = ConvNet200().to(device)
        if data_name == 'CIFAR10':
            self.model = ConvNet().to(device)
        elif data_name == 'CIFAR100':
            self.model = ConvNet100().to(device)
        if data_name.startswith("domainnet"):
            self.model = ConvNet345().to(device)
        if args.lora == 1 and args.alg!="FLORA":
            self.model = get_peft_model(self.model, lora_config)
        self.pid = pid
        self.num_workers = num_workers
        self.data_iterator = None
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.loss = 0
        self.lr_decay = lr_decay
        self.alg = alg
        self.data_idx = data_idx
        self.pre_ps_weight = None
        self.pre_loc_weight = None
        self.flag = False
        self.ci = None
        self.selection = selection
        self.T_part = T_part
        self.Li = None
        self.hi = None
        self.R=1
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.shared_projector_bank = None

    def data_id_loader(self, index):
        '''
        在每轮的开始，该工人装载数据集，以充当被激活的第index个客户端
        '''
        self.data_iterator = get_data_loader(index, self.data_idx, batch_size, data_name)

    def state_id_loader(self, index):
        '''
        在每轮的开始，该工人装载状态，以充当被激活的第index个客户端，使用外部的状态字典
        '''
        if not c_dict.get(index):
            return
        self.ci = c_dict[index]

    def get_train_loss(self):
        return self.loss

    def update_FedIT(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.data_id_loader(index)
        self.model.to(device)
        for name, param in self.model.named_parameters():
            if "classifier" in name or "head" in name or 'lora' in name:
                param.requires_grad = True

        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                               weight_decay=0.01)
        #self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,momentum=0.9,
        #                                       weight_decay=0.001)
        step = 0  # 新增步数计数
        self.loss =0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step += 1  # 步数+1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target.long())
                self.loss += loss.item() / args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                self.optimizer.zero_grad()
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k or 'classifier' in k or 'head' in k}
        if index % 10 == 0:
            print('loss:', self.loss)
        # 6. 模型迁回 CPU，清显存（如果后面这一段时间不用它算梯度的话）
        self.model.to("cpu")
        torch.cuda.empty_cache()
        return delta_w

    def update_FedLORA(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.data_id_loader(index)
        self.model.to(device)
        for name, param in self.model.named_parameters():
            if "classifier" in name or "head" in name:
                param.requires_grad = True
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                               weight_decay=0.01)
        step = 0  # 新增步数计数
        self.loss =0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step += 1  # 步数+1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target.long())
                self.loss += loss.item() / args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                self.optimizer.zero_grad()
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k or "classifier" in k or "head" in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k or "classifier" in k or "head" in k:
                    delta_w[k] = v.cpu() - weights[k].cpu()
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k].cpu()
        if index % 10 == 0:
            print('loss:', self.loss)
        # 6. 模型迁回 CPU，清显存（如果后面这一段时间不用它算梯度的话）
        self.model.to("cpu")
        torch.cuda.empty_cache()
        return delta_w

    def update_Fedgalore(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.data_id_loader(index)
        self.model.to(device)
        from galore_torch import GaLoreAdamW
        head_keywords = ["classifier", "head"]
        head_params = []
        galore_params = []
        other_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            name_lower = name.lower()
            # 1) 分类头参数
            if any(k in name_lower for k in head_keywords):
                head_params.append(p)
            # 2) 其余二维参数 -> GaLore
            elif p.ndim == 2:
                galore_params.append(p)
            # 3) 其他参数
            #else:
            #    other_params.append(p)
        param_groups = []
        if len(head_params) > 0:
            param_groups.append({'params': head_params,'lr': lr})
        if len(galore_params) > 0:
            param_groups.append({'params': galore_params,'rank': args.r,'update_proj_gap': args.K, 'scale': 1,
                'proj_type': 'std',  'lr': lr})
        if len(other_params) > 0:
            param_groups.append({'params': other_params,'lr': lr / 10 })
        self.optimizer = GaLoreAdamW(param_groups,weight_decay=0)
        step = 0  # 新增步数计数
        self.loss =0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step += 1  # 步数+1
                data = data.to(device)
                target = target.to(device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target.long())
                self.loss += loss.item() / args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v.cpu() - weights[k].cpu()
        # 6. 模型迁回 CPU，清显存（如果后面这一段时间不用它算梯度的话）
        self.model.to("cpu")
        torch.cuda.empty_cache()
        return delta_w

    def update_Fedgarare(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.data_id_loader(index)
        self.model.to(device)
        from garare_torch import GaRareAdamW
        head_keywords = ["classifier", "head"]
        head_params = []
        galore_params = []
        other_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            name_lower = name.lower()
            # 1) 分类头参数
            if any(k in name_lower for k in head_keywords):
                head_params.append(p)
            # 2) 其余二维参数 -> GaLore
            elif p.ndim == 2:
                galore_params.append(p)
            # 3) 其他参数
            #else:
            #    other_params.append(p)
        param_groups = []
        if len(head_params) > 0:
            param_groups.append({'params': head_params,'lr': lr})
        if len(galore_params) > 0:
            param_groups.append({'params': galore_params, 'rank': args.r, 'update_proj_gap': args.K, 'scale': 1,
                                 'proj_type':'random', 'lr': lr})
        if len(other_params) > 0:
            param_groups.append({'params': other_params,'lr': lr / 10 })
        self.optimizer = GaRareAdamW(param_groups,weight_decay=0)
        step = 0  # 新增步数计数
        self.loss =0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step += 1  # 步数+1
                data = data.to(device)
                target = target.to(device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target.long())
                self.loss += loss.item() / args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v.cpu() - weights[k].cpu()
        # 6. 模型迁回 CPU，清显存（如果后面这一段时间不用它算梯度的话）
        self.model.to("cpu")
        torch.cuda.empty_cache()
        return delta_w

    def update_FLORA(self, weights, E, index, lr):
        if data_name == 'imagenet':
            self.model = ConvNet200().to(device)
        if data_name == 'CIFAR10':
            self.model = ConvNet().to(device)
        elif data_name == 'CIFAR100':
            self.model = ConvNet100().to(device)
        self.model.load_state_dict(weights)

        self.model.to(device)
        self.model = get_peft_model(self.model, lora_config)
        self.data_id_loader(index)
        for name, param in self.model.named_parameters():
            if "classifier" in name or "head" in name or 'lora' in name:
                param.requires_grad = True
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                               weight_decay=0.01)
        step = 0  # 新增步数计数
        self.loss =0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step += 1  # 步数+1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if
                   'lora' in k or 'classifier' in k or 'head' in k}
        self.model.to("cpu")
        torch.cuda.empty_cache()
        return delta_w

    def update_FFA_LoRA(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        for name, param in self.model.named_parameters():
            if 'lora_A' in name:
                param.requires_grad = False
        for name, param in self.model.named_parameters():
            if "classifier" in name or "head" in name:
                param.requires_grad = True
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                               weight_decay=0.01)
        self.data_id_loader(index)
        step = 0  # 新增步数计数
        self.loss =0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step += 1  # 步数+1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if
                   'lora' in k or 'classifier' in k or 'head' in k}
        # 6. 模型迁回 CPU，清显存（如果后面这一段时间不用它算梯度的话）
        self.model.to("cpu")
        torch.cuda.empty_cache()
        return delta_w
    def update_RoLoRA(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        for name, param in self.model.named_parameters():
            if self.R%2==1:
                if 'lora_A' in name:
                    param.requires_grad = False
                if 'lora_B' in name:
                    param.requires_grad = True
            if self.R%2==0:
                if 'lora_B' in name:
                    param.requires_grad = False
                if 'lora_A' in name:
                    param.requires_grad = True

        self.R=self.R+1
        for name, param in self.model.named_parameters():
            if "classifier" in name or "head" in name:
                param.requires_grad = True

        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                               weight_decay=0.01)
        self.data_id_loader(index)
        step = 0  # 新增步数计数
        self.loss =0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step += 1  # 步数+1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if
                   'lora' in k or 'classifier' in k or 'head' in k}
        # 6. 模型迁回 CPU，清显存（如果后面这一段时间不用它算梯度的话）
        self.model.to("cpu")
        torch.cuda.empty_cache()
        return delta_w

    def update_LoRA_FAIR(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.data_id_loader(index)
        self.model.to(device)
        for name, param in self.model.named_parameters():
            if "classifier" in name or "head" in name:
                param.requires_grad = True
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        step = 0  # 新增步数计数
        self.loss =0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step += 1  # 步数+1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if
                   'lora' in k or 'classifier' in k or 'head' in k}
        # 6. 模型迁回 CPU，清显存（如果后面这一段时间不用它算梯度的话）
        self.model.to("cpu")
        torch.cuda.empty_cache()
        return delta_w

    def update_SAM(self, weights, E, index, lr):
        self.model.load_state_dict(weights)  # y_i = x, x:weights
        self.data_id_loader(index)
        for name, param in self.model.named_parameters():
            if "classifier" in name or "head" in name:
                param.requires_grad = True
        base_optimizer = torch.optim.AdamW
        #base_optimizer = torch.optim.SGD
        self.optimizer = SAM(filter(lambda p: p.requires_grad, self.model.parameters()), base_optimizer,  lr=lr, weight_decay=0.01, rho=args.rho, adaptive=True)
        step = 0  # 新增步数计数
        self.loss =0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step += 1  # 步数+1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                self.criterion(self.model(data), target).backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.second_step(zero_grad=True)
        self.loss = loss.item()
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k or 'classifier' in k or 'head' in k}
        return delta_w

    def update_FedCM(self, weights, E, index, ps_c, lr):
        self.model.set_weights(weights)
        self.model.to(device)
        if ps_c is None:
            ps_c = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        self.gamma = 0.9
        for name, param in self.model.named_parameters():
            if "classifier" in name or "head" in name:
                param.requires_grad = True
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                         weight_decay=0)
        for k, v in self.model.named_parameters():
            ps_c[k] = ps_c[k].to(device)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                for k, v in self.model.named_parameters():
                    v.grad.data = (1 - self.gamma) * v.grad.data + self.gamma * ps_c[k]
                self.optimizer.step()
        send_ci = {}
        for k, v in self.model.named_parameters():
            ps_c[k] = ps_c[k].to('cpu')
        for k, v in self.model.state_dict().items():
            send_ci[k] = - ps_c[k] - 1 / (E * len(self.data_iterator) * lr) * (v - weights[k])
        delta_w = {}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v - weights[k]
        return delta_w, send_ci

    def update_scaf(self, weights, E, index, ps_c, lr):
        self.model.set_weights(weights)
        self.model.to(device)
        if self.ci == None:
            self.ci = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        if ps_c == None:
            ps_c = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        self.state_id_loader(index)
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                               weight_decay=0.01)
        for n, p in model.named_parameters():
            ps_c[n] = ps_c[n].to(device)
            self.ci[n] = self.ci[n].to(device)
            weights[n] = weights[n].to(device)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                lg_loss = 0
                loss_c = self.criterion(output, target)
                for n, p in model.named_parameters():
                    lossh = (p * (self.ci[n] + ps_c[n])).sum()
                    lg_loss += lossh.item()
                loss = loss_c + lg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        send_ci = {}
        ci = deepcopy(self.ci)
        for k, v in self.model.state_dict().items():
            ps_c[k] = ps_c[k].to('cpu')
            self.ci[k] = self.ci[k].to('cpu')
            weights[k] = weights[k].to('cpu')
            ci[k] = ci[k].to('cpu')
            self.ci[k] = (weights[k] - v) / (E * len(self.data_iterator) * lr) + ci[k] - ps_c[k]
        for k, v in self.model.state_dict().items():
            send_ci[k] = -ci[k] + self.ci[k]
        delta_w = {}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v - weights[k]
        c_dict[index] = deepcopy(self.ci)
        return delta_w, send_ci


    def update_FedACG(self, weights, E, index, ps_c, lr):
        if ps_c == {}:
            ps_c = {k: torch.zeros_like(v, device='cpu') for k, v in self.model.state_dict().items()}
        for k, v in ps_c.items():
            weights[k] = weights[k].cpu() + ps_c[k].cpu() * args.gamma
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        for name, param in self.model.named_parameters():
            if "classifier" in name or "head" in name:
                param.requires_grad = True
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                               weight_decay=0.01, eps=1e-8)
        step = 0  # 新增步数计数
        self.loss=0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step=step +1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                #reg_loss = 0
                #for n, p in model.named_parameters():
                #    weights[n] = weights[n].to(device)
                #    L1 = ((p - weights[n].detach()) ** 2).sum()
                #    reg_loss += L1.item()
                #loss = self.criterion(output, target)+0.01*reg_loss
                loss = self.criterion(output, target)
                self.loss += loss.item() / args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k or 'classifier' in k or 'head' in k}
        self.model.to("cpu")
        torch.cuda.empty_cache()
        return delta_w

    def load_dict(self):
        self.func_dict = {
            'FedMoment': self.update_FedIT,  # add moment
            'SCAFFOLD': self.update_scaf,  # scaf
            'FedAdam': self.update_FedIT,  # FedAdam
            'FedCM': self.update_FedCM,
            'FedACG': self.update_FedACG,
            'FedSAM': self.update_SAM,
            'FLORA': self.update_FLORA,
            'FedIT': self.update_FedIT,
            'FFA_LoRA': self.update_FFA_LoRA,
            'LORA_FAIR': self.update_FedIT,
            'FedSVD': self.update_FedIT,
            'RoLoRA': self.update_RoLoRA,
            'FRLoRA': self.update_FedIT,
            'FedAvg': self.update_FedLORA,
            'Fedfull': self.update_FedLORA,
            'FedGalore': self.update_Fedgalore,
            'FedGarare': self.update_Fedgarare,

        }

    def update_func(self, alg, weights, E, index, lr, ps_c=None):
        self.load_dict()
        if alg in {'SCAFFOLD', 'FedCM','FedACG'}:
            return self.func_dict.get(alg, None)(weights, E, index, ps_c, lr)
        else:
            return self.func_dict.get(alg, None)(weights, E, index, lr)

def apply_weights_FLORA( num_workers, weights,model):
    sum_weights ={}
    lora_config = LoraConfig(
        r=int(args.r*args.selection*num_workers),  # 低秩矩阵的秩，通常在 4 到 64 之间[^18^]
        lora_alpha=args.r*2,  # 缩放参数，通常为 r 的 2 到 32 倍[^18^]
        lora_dropout=0.05,  # Dropout 比率，防止过拟合[^18^]
        bias="none",  # 不训练偏置项[^18^]
        task_type="IMAGE_CLASSIFICATION",  # 任务类型，根据具体任务选择[^18^]
        target_modules=['attn.qkv', 'attn.proj']  # 目标模块，根据模型结构指定[^18^]
    )
    for weight in weights:
        for k, v in weight.items():
            if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                if 'lora_A' in k :
                    new = [sum_weights[k], v / (num_workers * args.selection)]
                    sum_weights[k] = torch.cat(new, dim=0)
                    #print(sum_weights[k].shape)
                elif 'lora_B' in k:
                    new = [sum_weights[k], v ]
                    sum_weights[k] = torch.cat(new, dim=1)
                else:
                    sum_weights[k]+= v/ (num_workers * args.selection)
            else:
                if 'lora_A' in k:
                    sum_weights[k] = v / (num_workers * args.selection)
                elif 'lora_B' in k:
                    sum_weights[k] = v * 1
                else:
                    sum_weights[k]=v/ (num_workers * args.selection)
    model = get_peft_model(model, lora_config)
    model.load_state_dict(sum_weights,strict=False)
    model.merge_and_unload()
    return {k: v.cpu() for k, v in model.state_dict().items()}

def apply_weights_LORA_SVD(num_workers, weights, model, selection=1.0):
    # 1) 先用 AB->∆W 平均->SVD 的方式得到新的 A/B
    new_lora_state = aggregate_AB_then_SVD(
        weights=weights,
        r=int(args.r),
        num_workers=num_workers,
        selection=selection
    )
    lora_only = {k: v for k, v in new_lora_state.items() if "lora" in k}
    scale = 1.0 / (num_workers * args.selection)
    # 聚合 delta_wi
    for weight in weights:
        for k, v in weight.items():
            if ('classifier' in k) or ('head' in k):
                if k not in lora_only.keys():
                    lora_only[k] = torch.zeros_like(v, device='cpu')
                lora_only[k].add_(v, alpha=scale)  # inplace 加法
    model.load_state_dict(lora_only,strict=False)
    return {k: v.cpu() for k, v in model.state_dict().items()}

def apply_weights_FRLoRA(num_workers, weights, model, selection=args.selection):
    new_lora_state = aggregate_FRLORA(
        weights=weights,
        r=int(args.r),
        num_workers=num_workers,
        selection=selection
    )
    lora_only = {k: v for k, v in new_lora_state.items() if "lora" in k}
    model.load_state_dict(lora_only,strict=False)
    return {k: v.cpu() for k, v in model.state_dict().items()}


def apply_weights_SCAF(num_workers, weights,model,ps_c):
    model.to('cpu')
    m = [mi for _, mi in weights]
    weightss = [w for w,_ in weights]
    sum_c = {}
    # 首先以第一个客户端为基础初始化 sum_c（避免判断逻辑）
    for k, v in m[0].items():
        sum_c[k] =v / (num_workers * selection)
    # 之后叠加剩余客户端的梯度
    for ci in m[1:]:
        for k, v in ci.items():
            sum_c[k]+= v / (num_workers * selection)
    if ps_c == {}:
        ps_c = {k: torch.zeros_like(v.cpu()) for k, v in model.named_parameters()}
        for k, v in m[0].items():
            ps_c[k]=sum_c[k]
    else:
        for k, v in m[0].items():
            if alg in {'SCAFFOLD'}:
                ps_c[k]=ps_c[k]+sum_c[k]*selection
            if alg in {'SCAFFOLD+'}:
                ps_c[k] = ps_c[k] + sum_c[k] * 0.2
    ps_w = model.state_dict()  # w : ps_w

    sum_weights = {k: torch.zeros_like(v) for k, v in ps_w.items() if "lora" in k}
    for weight in weightss:
        for k, v in weight.items():
            if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                sum_weights[k] += v / (num_workers * selection)
            else:
                sum_weights[k] = v / (num_workers * selection)
    model.load_state_dict(sum_weights,strict=False)
    return model.state_dict(),ps_c


@torch.no_grad()
def apply_weights_avg(num_workers, weights,model):
    ps_w = {k: v.cpu() for k, v in model.state_dict().items()}
    sum_weights = {k: torch.zeros_like(v) for k, v in ps_w.items() if
               ('lora' in k) or ('classifier' in k) or ('head' in k)}
    scale = 1.0 / (num_workers * args.selection)
    # 聚合 delta_wi
    for weight in weights:
        for k, v in weight.items():
            if ('lora' in k) or ('classifier' in k) or ('head' in k):
                sum_weights[k].add_(v, alpha=scale)  # inplace 加法
    model.load_state_dict(sum_weights,strict=False)
    return {k: v.cpu() for k, v in model.state_dict().items()}

@torch.no_grad()
def apply_weights_FedLORA(num_workers, weights,model):
    model.to('cpu')
    ps_w = {k: v.cpu() for k, v in model.state_dict().items()}
    sum_weights = {k: torch.zeros_like(v) for k, v in ps_w.items() if
                   ('lora' in k) or ('classifier' in k) or ('head' in k)}
    for weight in weights:
        for k, v in weight.items():
            if not torch.is_floating_point(ps_w[k]):
                continue
            if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                sum_weights[k] += v / (num_workers * selection)
            else:
                sum_weights[k] = v / (num_workers * selection)
    for k, v in sum_weights.items():  # w = w + delta_w
        ps_w[k] = ps_w[k] + sum_weights[k]
    model.load_state_dict(ps_w)
    return {k: v.cpu() for k, v in model.state_dict().items()}

@torch.no_grad()
def apply_weights_avg_full(num_workers, weights,model):
    ps_w = model.state_dict()  # w : ps_w
    sum_weights = {}  # delta_w : sum_weights
    global_weights = {}
    for weight in weights:
        for k, v in weight.items():
            if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                sum_weights[k] += v / (num_workers * selection)
            else:
                sum_weights[k] = v / (num_workers * selection)
    for k, v in sum_weights.items():  # w = w + delta_w
        global_weights[k] = ps_w[k].cpu() + sum_weights[k].cpu()
    model.load_state_dict(global_weights)
    return model.state_dict()

@torch.no_grad()
def apply_weights_avgACG(num_workers, weights,model,momen_m):
    model.to('cpu')
    gamma = args.gamma
    ps_w = {k: v.cpu() for k, v in model.state_dict().items()}
    sum_weights = {}
    for weight in weights:
        for k, v in weight.items():
            if k in sum_weights.keys():
                sum_weights[k] += 1 / (args.num_workers * args.selection) * v
            else:
                sum_weights[k] = 1 / (args.num_workers * args.selection) * v
    if momen_m == {}:
        momen_m = deepcopy(sum_weights)
    else:
        for k, v in sum_weights.items():
            momen_m[k] = gamma * v + sum_weights[k]
    for k, v in sum_weights.items():
        ps_w[k] = ps_w[k] + momen_m[k]
    model.load_state_dict(ps_w)
    return {k: v.cpu() for k, v in model.state_dict().items()}, momen_m


def set_random_seed(seed=42):
    """
    设置随机种子以确保实验的可重复性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
if __name__ == "__main__":
    # 获取args
    #ray.init(ignore_reinit_error=True)
    set_random_seed(seed=42)
    epoch = args.epoch
    num_workers = args.num_workers
    batch_size = args.batch_size
    lr = args.lr
    E = args.E
    lr_decay = args.lr_decay  # for CIFAR10
    alg = args.alg
    data_name = args.data_name
    selection = args.selection
    tau = args.tau
    lr_ps = args.lr_ps
    alpha_value = args.alpha_value
    alpha = args.alpha
    extra_name = args.extname
    check = args.check
    T_part = args.T_part
    c_dict = {}
    lr_decay = args.lr_decay
    hi_dict = {}
    Li_dict = {}
    import time
    localtime = time.asctime(time.localtime(time.time()))
    checkpoint_path = './checkpoint/ckpt-{}-{}-{}-{}-{}-{}'.format(alg, lr, extra_name, alpha_value, extra_name,
                                                                   localtime)
    c_dict = {}  # state dict
    assert alg in {
        'FedAvg',
        'FedMoment',
        'SCAFFOLD',
        'FedCM',
        'FedSAM',
        'FedACG',
        'FLORA',
        'FFA_LoRA',
        'FedIT',
        'FedSVD',
        'LORA_FAIR',
        'RoLoRA',
        'FRLoRA',
        'Fedfull',
        'FedLORA',
        'FedGalore',
        'FedGarare',
    }
    #  配置logger
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("./log/{}-{}-{}-{}-{}-{}-{}.txt"
                                  .format(alg, data_name, lr, num_workers, batch_size, E, lr_decay))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    writer = SummaryWriter(comment=alg)

    nums_cls = 100
    if data_name == 'CIFAR10':
        nums_cls = 10
    if data_name == 'CIFAR100':
        nums_cls = 100
    if data_name == 'imagenet':
        nums_cls = 200
    if data_name.startswith("domainnet"):
        nums_cls =345

    nums_sample = 500
    if data_name == 'CIFAR10':
        nums_sample = int(50000 / (args.num_workers))
    if data_name == 'CIFAR100':
        nums_sample = int(50000 / (args.num_workers))
    if data_name == 'imagenet':
        nums_sample = int(100000 / (args.num_workers))
    if data_name == 'domainnet_infograph':
        nums_sample = int(37087 / (args.num_workers))
    if data_name == 'domainnet_clipart':
        nums_sample = int(34019 / (args.num_workers))
    if data_name == 'domainnet_quickdraw':
        nums_sample = int(120750/ (args.num_workers))
    if data_name == 'domainnet_real':
        nums_sample = int(122563/ (args.num_workers))
    if data_name == 'domainnet_sketch':
        nums_sample = int(49115/ (args.num_workers))
    if data_name == 'domainnet_painting':
        nums_sample = int(50416 / (args.num_workers))


    import pickle
    filename = 'num_workers_{}-alpha_value_{}-data_{}'.format(num_workers, args.alpha_value, data_name)
    if args.alpha_value == 1:
        filename = 'data_idx.data'
        f = open(filename, 'rb')
        data_idx = pickle.load(f)
    else:
        import os
        import pickle
        filename = f'num_workers_{num_workers}-alpha_value_{alpha_value}-data_{data_name}'
        if os.path.exists(filename):
            # 文件存在，直接加载
            with open(filename, 'rb') as f:
                data_idx = pickle.load(f)
            print(f"加载已有数据索引文件: {filename}")
            std = None  # 若你需要 std，则存成 tuple 后一起加载
        else:
            # 文件不存在，生成并保存
            # 默认使用极速版本（推荐）
            data_idx, std = data_from_dirichlet(data_name, alpha_value, nums_cls, nums_wk=num_workers, nums_sample=nums_sample, use_cache=False,fast_mode=True)
            with open(filename, 'wb') as f:
                pickle.dump(data_idx, f)
            print(f"生成并保存新数据索引文件: {filename}")
    # logger.info('std:{}'.format(std))
    #ray.init(ignore_reinit_error=True, num_gpus=num_gpus)
    import os
    import ray

    base_tmp = os.path.abspath("./tmp")
    ray_tmp = os.path.join(base_tmp, "ray")

    os.makedirs(ray_tmp, exist_ok=True)

    os.environ["TMPDIR"] = base_tmp

    ray.init(
        ignore_reinit_error=True,
        num_gpus=num_gpus,
        _temp_dir=ray_tmp,
        include_dashboard=False,
    )
    if data_name == 'imagenet':
        model = ConvNet200().to(device)
    if data_name == 'CIFAR10':
        model = ConvNet().to(device)
    elif data_name == 'CIFAR100':
        model = ConvNet100().to(device)
    if data_name.startswith("domainnet"):
        model = ConvNet345().to(device)

    epoch_s = 0
    workers = [DataWorker.remote(i, data_idx, num_workers,
                                 lr, batch_size=batch_size, alg=alg, data_name=data_name, selection=selection,
                                 T_part=T_part) for i in range(int(num_workers * selection / args.p))]
    logger.info('extra_name:{},alg:{},E:{},data_name:{}, epoch:{}, lr:{},alpha_value:{},alpha:{},CNN:{},gamma:{}'
                .format(extra_name, alg, E, data_name, epoch, lr, alpha_value, alpha, args.CNN, args.gamma))

    test_loader = get_data_loader_test(data_name)
    train_loader = get_data_loader_train(data_name)
    print("@@@@@ Running synchronous parameter server training @@@@@@")

    if args.CNN == 'VIT-B':
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('vit_base_patch16_224_in21k.pth', map_location=device)
            # 删除不需要的权重
            del_keys = ['head.weight', 'head.bias'] if model.has_logits \
                else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            for k in del_keys:
                del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

    if args.CNN == 'VIT-L':
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('jx_vit_large_patch16_224_in21k-606da67d.pth', map_location=device)
            # 删除不需要的权重
            del_keys = ['head.weight', 'head.bias'] if model.has_logits \
                else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            for k in del_keys:
                del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

    if args.CNN == 'swin_tiny':
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('swin_tiny_patch4_window7_224.pth', map_location=device)["model"]
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

    if args.CNN == 'swin_small':
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('swin_small_patch4_window7_224.pth', map_location=device)["model"]
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

    if args.CNN == 'swin_base':

        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('swin_base_patch4_window7_224_22k.pth', map_location=device)["model"]
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

    if args.CNN == 'swin_large':
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('swin_large_patch4_window7_224_22k.pth', map_location=device)["model"]
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

    if args.lora == 1 and args.alg!='FLORA':
        model = get_peft_model(model, lora_config)

    result_list, X_list = [], []
    result_list_loss = []
    test_list_loss = []
    start = time.time()
    best_acc = 0
    no_improve = 0
    zero = model.state_dict()
    for k, v in model.state_dict().items():
        zero[k] = zero[k] - zero[k]
    ps_c = deepcopy(zero)
    del zero
    div = []
    sim = []
    momen_m = {}
    current_weights=model.state_dict()
    for epochidx in range(epoch_s, epoch):
        start_time1 = time.time()
        #lr = lr * lr_decay
        if args.lr_decay==2:
            eta_max=args.lr
            eta_min=0
            t=epochidx
            T=args.epoch
            lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * t / T))

        index = np.arange(num_workers)  # 100
        np.random.shuffle(index)
        index = index[:int(num_workers * selection)]  # 10id
        #index = np.sort(index)

        if alg in {'SCAFFOLD'}:
            weights_and_ci = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights_and_ci = weights_and_ci + [
                    worker.update_func.remote(alg, current_weights, E, idx, lr, ps_c=ps_c)
                    for worker, idx in zip(workers, index_sel)]
            weights_and_ci = ray.get(weights_and_ci)
            current_weights, ps_c = apply_weights_SCAF(num_workers, weights_and_ci, model, ps_c)
            model.load_state_dict(current_weights)
            del weights_and_ci

        elif alg in { 'FedSAM'}:
            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights = weights + [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                     zip(workers, index_sel)]
            weights=ray.get(weights)
            current_weights = apply_weights_FedLORA(num_workers, weights,model)
            model.load_state_dict(current_weights)

        elif alg in { 'FedIT','RoLoRA','FFA_LoRA'}:

            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights = weights + [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                     zip(workers, index_sel)]
            weights=ray.get(weights)
            current_weights = apply_weights_avg(num_workers, weights,model)
            model.load_state_dict(current_weights)

        elif alg in { 'FedLORA'}:
            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights = weights + [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                     zip(workers, index_sel)]
            weights=ray.get(weights)
            current_weights = apply_weights_FedLORA(num_workers, weights,model)
            model.load_state_dict(current_weights)


        elif alg in {'FedSVD'}:
            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights = weights + [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                     zip(workers, index_sel)]
            weights = ray.get(weights)
            time3 = time.time()
            print(epochidx, '    ', time3 - start_time1)
            current_weights = apply_weights_LORA_SVD(num_workers, weights, model, selection=selection)

            model.load_state_dict(current_weights)

        elif alg in {'LORA_FAIR'}:
            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights = weights + [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                     zip(workers, index_sel)]
            weights=ray.get(weights)
            time3 = time.time()
            print(epochidx, '    ', time3 - start_time1)
            current_weights = apply_weights_lora_fair_CV(
                num_workers=num_workers,
                weights=weights,  # 客户端上报的 LoRA(绝对)参数字典列表
                model=model,
                selection=selection,  # 参与比例（此处仅做均值，不额外缩放）
                iters=200,  # 可调
                lr=0.1,  # 可调
                lambda_reg=0.01  # 可调
            )
            model.load_state_dict(current_weights)

        elif alg in {'FRLoRA'}:
            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights = weights + [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                     zip(workers, index_sel)]
            weights = ray.get(weights)
            time3 = time.time()
            current_weights = apply_weights_FRLoRA(num_workers, weights, model, selection=selection)
            model.load_state_dict(current_weights)

        elif alg in {'FLORA'}:
            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights = weights + [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                     zip(workers, index_sel)]
            weights = ray.get(weights)
            current_weights = apply_weights_FLORA(num_workers, weights,model)
            current_weights =  {
                    k.replace("base_model.model.", ""): v
                    for k, v in current_weights.items()
                    #if "lora_" not in k
                }
            model.load_state_dict(current_weights)


        elif alg in {'Fedfull','FedGalore','FedAvg','FedGarare'}:
            weights = []
            index_sel = index
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights = weights + [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                 zip(workers, index_sel)]
            weights = ray.get(weights)
            current_weights = apply_weights_avg_full(num_workers, weights, model)
            model.load_state_dict(current_weights)

        if alg in { 'FedACG'}:
            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights = weights + [worker.update_func.remote(alg, current_weights, E, idx, lr, momen_m) for
                                     worker, idx in
                                     zip(workers, index_sel)]
            weights = ray.get(weights)
            current_weights,momen_m = apply_weights_avgACG(num_workers,weights,model,momen_m)
            model.load_state_dict(current_weights)
            del weights

        end_time1 = time.time()
        #print(epochidx, '    ', end_time1 - time3)
        print(epochidx, '    ', end_time1 - start_time1)
        args.i = 1
        if epochidx % args.preprint == 0:
            start_time1 = time.time()
            print('测试')
            test_loss = 0
            train_loss = 0
            accuracy, test_loss, train_loss = evaluate(model, test_loader, train_loader)
            end_time1 = time.time()
            print('测试完毕', '    ', end_time1 - start_time1)
            test_loss = test_loss.to('cpu')
            loss_train_median = train_loss.to('cpu')
            # early stop
            if accuracy > best_acc:
                best_acc = accuracy
                no_improve = 0
            else:
                no_improve += 1
                if no_improve == 1000:
                    break
            writer.add_scalar('accuracy', accuracy, epochidx * E)
            writer.add_scalar('loss median', loss_train_median, epochidx * E)
            logger.info(
                "Iter {}: \t accuracy is {:.2f}, train loss is {:.5f}, test loss is {:.5f}, data:{}, name:{},lr:{:.7f},CNN:{},GPU:{},gamma:{},r:{},alpha_value:{},data:{}".format(
                    epochidx, accuracy,
                    loss_train_median, test_loss,
                    args.data_name, args.alg, lr, args.CNN, args.gpu, args.gamma, args.r, args.alpha_value,
                    args.data_name))

            print(
                "Iter {}: \t accuracy is {:.2f}, train loss is {:.5f}, test loss is {:.5f}, data:{}, name:{},lr:{:.7f},CNN:{},GPU:{},data:{},gamma:{},r:{},alpha_value:{}".format(
                    epochidx, accuracy,
                    loss_train_median, test_loss,
                    args.data_name, args.alg, lr, args.CNN, args.gpu, args.data_name, args.gamma,
                    args.r, args.alpha_value))

            if np.isnan(loss_train_median):
                logger.info('nan~~')
                break
            X_list.append(epochidx)
            result_list.append(accuracy)
            result_list_loss.append(loss_train_median)
            test_list_loss.append(test_loss)
    logger.info("Final accuracy is {:.2f}.".format(accuracy))
    endtime = time.time()
    logger.info('time is pass:{}'.format(endtime - start))
    x = np.array(X_list)
    result = np.array(result_list)
    result_loss = np.array(result_list_loss)
    test_list_loss = np.array(test_list_loss)
    save_name = './plot/alg_{}-data_{}-#wk_{}-K_{}-lr_{}-alpha_value_{}-selec_{}-alpha{}-{}-gamma{}-r{}-CNN{}-time{}'.format(
        alg, args.data_name, num_workers, args.K,
        args.lr, args.alpha_value, args.selection, args.alpha,
        extra_name, args.gamma, args.r, args.CNN, endtime)
    save_name = save_name + '.npy'
    np.save(save_name, (x, result, result_loss, test_list_loss))
    ray.shutdown()