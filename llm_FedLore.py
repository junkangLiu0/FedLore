import math
import os
from copy import deepcopy
#from Fedmerge_llm import apply_weights_TA
from lora_SVD import aggregate_AB_then_SVD, aggregate_FRLORA
from lora_fair import apply_weights_lora_fair
from sam import SAM

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--lg', default=1.0, type=float, help='learning rate')
parser.add_argument('--epoch', default=100, type=int, help='number of R to train')
parser.add_argument('--num_workers', default=100, type=int, help='#workers')
parser.add_argument('--batch_size', default=16, type=int, help='# batch_size')
parser.add_argument('--E', default=1, type=int, help='# number of local epoch to train')
parser.add_argument('--alg', default='FedMoment', type=str, help='alg')  # FedMoment cddplus cdd SCAF atte
parser.add_argument('--extname', default='EM', type=str, help='extra_name')
parser.add_argument('--gpu', default='0,1', type=str, help='use which gpus')
parser.add_argument('--lr_decay', default='0.99', type=float, help='lr_decay')
parser.add_argument('--data_name', default='imagenet', type=str, help='lr_decay')
parser.add_argument('--alpha_value', default='0.6', type=float, help='for dirichlet')
parser.add_argument('--selection', default='0.06', type=float, help=' C')
parser.add_argument('--check', default=0, type=int, help=' if check')
parser.add_argument('--T_part', default=10, type=int, help=' for mom_step')
parser.add_argument('--alpha', default=1, type=float, help=' for mom_step')
parser.add_argument('--CNN', default='VIT-L', type=str, help=' for model')
parser.add_argument('--gamma', default=0.9, type=float, help=' for mom_step')
parser.add_argument('--weights', type=str, default='./swin_tiny_patch4_window7_224.pth',
                    help='initial weights path')
parser.add_argument('--p', default=1, type=int, help=' for mom_step')
parser.add_argument('--datapath', type=str, default="./data")
parser.add_argument('--num_gpus_per', default=0.5, type=float, help=' for mom_step')
parser.add_argument('--rho', default=0.1, type=float, help='rho')
parser.add_argument('--optimizer', default='SGD', type=str, help='SGD,AdamW')
parser.add_argument("--preprint", type=int, default=5, help="")
parser.add_argument("--R", type=int, default=1, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--lora", type=int, default=0, help="")
parser.add_argument("--r", type=int, default=16, help="the perturbation radio for the SAM optimizer.")
parser.add_argument('--K', default=20, type=int, help='#workers')
parser.add_argument('--freeze', default=1, type=int, help='# batch_size')
parser.add_argument("--pre", type=int, default=1, help="the perturbation radio for the SAM optimizer.")
parser.add_argument('--print', default=0, type=int, help=' for mom_step')

args = parser.parse_args()
print(args.lora)
gpu_idx = args.gpu
print('gpu_idx', gpu_idx)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import SubsetRandomSampler, random_split
import random
import ray
from tensorboardX import SummaryWriter
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, \
    RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from dirichlet_data import data_from_dirichlet
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Adafactor, Trainer, TrainingArguments
from datasets import load_dataset, tqdm
from peft import LoraConfig, get_peft_model, TaskType
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
num_gpus_per = args.num_gpus_per  # num_gpus_per = 0.16
num_gpus = len(gpu_idx.split(','))
data_name = args.data_name
CNN = args.CNN

if args.CNN=='roberta_base':
    model_path='./roberta_base'
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    #model = RobertaForSequenceClassification.from_pretrained(model_path)
    lora_config = LoraConfig(
        r=args.r,  # LoRA attention dimension
        lora_alpha=args.r*2,  # Alpha scaling
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,  # Sequence classification
        #target_modules=['query', 'value', 'key','intermediate.dense','output.dense'] , # Target modules to apply LoRA
        #target_modules=['query', 'value', 'key'],
        target_modules=['query', 'value', 'key'],
        modules_to_save=None
        #modules_to_save = ["classifier"],
    )

if data_name in ['MNLI', 'SNLI','ANLI']:
    num_labels=3
elif data_name == 'STS-B':
    num_labels= 1
elif data_name == 'AG_News':
    num_labels = 4
elif data_name == 'DBPedia_14':
    num_labels = 14
else:
    num_labels= 2

if args.data_name=='QQP':
    dataset_path = './data/QQP'
    # 加载数据集
    dataset = load_dataset(dataset_path)
    # 数据预处理
    def preprocess_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True, padding="max_length",
                         max_length=128)
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]

if args.data_name=='MNLI':
    dataset_path = './data/MNLI'
    # 加载数据集
    dataset = load_dataset(dataset_path)
    # 数据预处理
    def preprocess_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True, padding="max_length",
                         max_length=128)
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]

if args.data_name=='STS-B':
    dataset_path = './data/sts-b'
    # 加载数据集
    dataset = load_dataset(dataset_path)
    dataset = dataset.rename_column("score", "label")
    # 数据预处理
    def preprocess_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding="max_length",
                         max_length=128)
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    #tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]

if args.data_name=='WNLI':
    dataset_path = './data/WNLI'
    # 加载数据集
    dataset = load_dataset(dataset_path)
    # 数据预处理
    def preprocess_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True, padding="max_length",
                         max_length=128)
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]


if args.data_name=='RTE':
    dataset_path = './data/RTE'
    dataset = load_dataset(dataset_path)
    def preprocess_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True, padding="max_length",
                         max_length=128)
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

if args.data_name=='MRPC':
    def preprocess_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True, padding="max_length",max_length=128)
    dataset_path = './data/MRPC'
    dataset = load_dataset(dataset_path)
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

if args.data_name=='qnli':
    def preprocess_function(examples):
        # 拼接问题和句子
        inputs = tokenizer(
            examples["text1"],
            examples["text2"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        labels = [1 if label == "entailment" else 0 for label in examples["label"]]
        inputs["labels"] = labels
        return inputs
    dataset_path = './data/qnli'
    dataset = load_dataset(dataset_path)
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]


if args.data_name=='sst2':
    def preprocess_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", return_tensors="pt",max_length=128)
    dataset_path = './data/sst2'
    dataset = load_dataset(dataset_path)
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]

if args.data_name=='cola':
    dataset_path = './data/cola'
    def preprocess_function(examples):
        return tokenizer(examples["Sentence"], padding="max_length", truncation=True,max_length=128)

    dataset = load_dataset(dataset_path)
    dataset = dataset.rename_column("Acceptability", "label")
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    #tokenized_dataset = tokenized_dataset.rename_column("Acceptability", "label")
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]

if args.data_name == 'SNLI':
    dataset_path = './data/SNLI'
    # 加载数据集
    dataset = load_dataset(dataset_path)
    # 过滤掉无效标签，SNLI 中有些样本 label = -1
    dataset = dataset.filter(lambda example: example["label"] != -1)
    # 数据预处理
    def preprocess_function(example):
        return tokenizer(
            example["premise"],
            example["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )
    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]


if args.data_name == 'AG_News':
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=4)
    dataset_path = './data/AG_News'
    # 加载数据集
    dataset = load_dataset(dataset_path)
    # 数据预处理
    def preprocess_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"])
    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["test"]

if args.data_name == 'DBPedia_14':
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=14)
    dataset_path = './data/DBPedia_14'
    # 加载数据集
    dataset = load_dataset(dataset_path)
    # 数据预处理
    def preprocess_function(example):
        return tokenizer(
            example["title"],
            example["content"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )
    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["test"]

if args.data_name == 'IMDB':
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2)
    dataset_path = './data/IMDB'
    # 加载数据集
    dataset = load_dataset(dataset_path)

    # 数据预处理
    def preprocess_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["test"]

if args.data_name == 'ANLI':
    dataset_path = './data/ANLI'
    dataset = load_dataset(dataset_path)

    def preprocess_function(example):
        return tokenizer(
            example["premise"],
            example["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # 过滤无效标签（如果有）
    if "label" in train_dataset.column_names:
        train_dataset = train_dataset.filter(lambda example: example["label"] != -1)
        test_dataset = test_dataset.filter(lambda example: example["label"] != -1)

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )
    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    tokenized_dataset = {
        "train": train_dataset,
        "test": test_dataset
    }
seed = 42
if args.alpha_value==1:
    def get_data_loader(pid, data_idx, batch_size, data_name):
        """Safely downloads data. Returns training/validation set dataloader. 使用到了外部的数据"""
        generator = torch.Generator().manual_seed(42)
        train_dataset = tokenized_dataset["train"]
        total_size = len(train_dataset)
        #print(total_size)
        subset_size = total_size // args.num_workers
        remainder = total_size % args.num_workers  # 计算剩余的样本数

        # 创建分割大小列表
        split_sizes = [subset_size] * (args.num_workers - 1) + [subset_size + remainder]
        subsets = random_split(train_dataset, split_sizes, generator=generator)
        sample_chosed = data_idx[pid]
        #train_sampler = SubsetRandomSampler(sample_chosed)
        #train_dataset = tokenized_dataset["train"]
        train_loader = DataLoader(subsets[pid], batch_size=args.batch_size, shuffle=True)
        return train_loader

if args.alpha_value!=1:
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
    #train_dataset = tokenized_dataset["train"]
    #test_dataset = tokenized_dataset["validation"]
    if "validation" in tokenized_dataset:
        test_dataset = tokenized_dataset["validation"]
    elif "test" in tokenized_dataset:
        test_dataset = tokenized_dataset["test"]
    else:
        raise ValueError(f"No validation/test split found for dataset: {data_name}")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4)
    return test_loader

def get_data_loader_train(data_name):
    #train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))
    train_dataset = tokenized_dataset["train"].select(range(1000))
    #test_dataset = tokenized_dataset["validation"]
    test_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4)
    return test_loader


def evaluate(model, test_loader, train_loader):
    """Evaluates the accuracy of the model on a validation dataset."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    test_loss = 0
    train_loss = 0
    start_time1 = time.time()
    print('evaluate')
    with torch.no_grad():
        for batch in tqdm(test_loader,disable=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["label"].to(device)
            model.zero_grad()
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            test_loss+= criterion(logits, target)
        for batch in tqdm(train_loader, disable=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["label"].to(device)
            model.zero_grad()
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            _, predicted = torch.max(logits.data, 1)
            train_loss += criterion(logits, target)
    accuracy = 100. * correct / total
    end_time1 = time.time()
    print('evaluate完毕', '    ', end_time1 - start_time1)
    model.to('cpu')
    torch.cuda.empty_cache()
    return  accuracy , test_loss / len(test_loader), train_loss / len(train_loader)

import torch
#@ray.remote(num_cpus=1,num_gpus=num_gpus_per)
@ray.remote(num_gpus=num_gpus_per)
class DataWorker(object):

    def __init__(self, pid, data_idx, num_workers, lr, batch_size, alg, data_name, selection, T_part):
        self.alg = alg
        if args.CNN == 'roberta_base':
            model_path = './roberta_base'
            self.model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

        if args.lora == 1 and args.alg!="FLORA":
            self.model = get_peft_model(self.model, lora_config)
            print(args.lora)
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
        self.ci = {}
        self.selection = selection
        self.T_part = T_part
        self.Li = None
        self.hi = None
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.momen_v = {}
        self.momen_m = {}
        self.R =1
        self.t ={k:  torch.tensor([0], dtype=torch.float32, device='cpu') for k, v in self.model.named_parameters()}
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

    def data_id_loader(self, index):
        '''
        在每轮的开始，该工人装载数据集，以充当被激活的第index个客户端
        '''
        self.data_iterator = get_data_loader(index, self.data_idx, batch_size, data_name)

    def state_id_loader(self, index, shared_state):
        '''
        在每轮的开始，该工人装载状态，以充当被激活的第index个客户端，使用外部的状态字典
        '''
        # c_dict = ray.get(c_dict_id)
        self.ci = ray.get(shared_state.get_ci_dict.remote(index))

    def get_train_loss(self):
        return self.loss

    def get_param_name(self, param):
        # 获取参数的名称
        for name, p in self.model.named_parameters():
            if p is param:
                return name
        return None


    def update_FedIT(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=0.01)
        step = 0  # 新增步数计数
        self.loss =0
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                self.loss += loss.item() / args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1)
                self.optimizer.step()
                step += 1  # 步数+1
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k].cpu()
        if index % 4 == 0:
            print('loss:',self.loss)
        # 6. 模型迁回 CPU，清显存（如果后面这一段时间不用它算梯度的话）
        self.model.to("cpu")
        torch.cuda.empty_cache()
        return delta_w

    def update_scaf(self, weights, E, index, ps_c, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        if self.ci == {}:
            self.ci = {k: torch.zeros_like(v,device='cpu') for k, v in self.model.named_parameters()}
        if ps_c == {}:
            ps_c = {k: torch.zeros_like(v,device='cpu') for k, v in self.model.named_parameters()}
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=0.01)
        for k in ps_c:
            ps_c[k] = ps_c[k].to(device)
            self.ci[k] = self.ci[k].to(device)
            weights[k] = weights[k].to(device)
        self.loss = 0
        step = 0  # 新增步数计数
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                step = step + 1
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                self.loss+=loss.item()/args.K
                loss =  self.criterion(logits, target.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                with torch.no_grad():
                    for n, p in self.model.named_parameters():
                        p.add_(self.ci[n]*lr -ps_c[n]*lr)  # 再加 c
        send_ci = {}
        ci = {}
        for k, v in self.model.named_parameters():
            v_cpu = v.detach().to('cpu')
            ps_c[k] = ps_c[k].to('cpu')
            self.ci[k] = self.ci[k].to('cpu')
            weights[k] = weights[k].to('cpu')
            ci[k] = self.ci[k]
            self.ci[k] = (weights[k] - v_cpu) / (args.K * lr) + ci[k] - ps_c[k]

        for k, v in self.model.named_parameters():
            if 'lora' not in k:
                continue
            send_ci[k] = -ci[k] + self.ci[k]
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]
        if index % 4 == 0:
            print('loss:',self.loss)
        #del  target, output, loss, ci
        #torch.cuda.empty_cache()
        return delta_w, send_ci

    def update_SAM(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        base_optimizer = torch.optim.AdamW
        self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=lr, weight_decay=0.01, rho=args.rho,adaptive=True)
        step = 0  # 新增步数计数
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                self.criterion(self.model(input_ids, attention_mask=attention_mask).logits, target).backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1)
                self.optimizer.second_step(zero_grad=True)
                step += 1  # 步数+1
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]
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
        self.optimizer = torch.optim.AdamW([
            {"params": [p for n, p in self.model.named_parameters() if "lora_A" in n], "lr": lr},
            {"params": [p for n, p in self.model.named_parameters() if "lora_B" in n], "lr": lr * 2},
            {"params": [p for n, p in self.model.named_parameters() if "lora_" not in n], "lr": lr}
        ], weight_decay=0.01)
        self.data_id_loader(index)
        step = 0  # 新增步数计数
        self.loss =0
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                step += 1  # 步数+1
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                self.loss += loss.item() / args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1)
                self.optimizer.step()
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
        if index % 4 == 0:
            print('loss:',self.loss)
        # 6. 模型迁回 CPU，清显存（如果后面这一段时间不用它算梯度的话）
        self.model.to("cpu")
        torch.cuda.empty_cache()
        return delta_w
    def update_FLORA(self, weights, E, index, lr):
        config = RobertaConfig.from_pretrained('./roberta_base')
        config.num_labels = num_labels
        self.model = RobertaForSequenceClassification(config)
        self.model.load_state_dict(weights, strict=False)
        self.model.to(device)
        if args.lora == 1:
            self.model = get_peft_model(self.model, lora_config)
        self.data_id_loader(index)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        step = 0  # 新增步数计数
        self.loss =0
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                step += 1  # 步数+1
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1)
                self.optimizer.step()
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
        if index % 4 == 0:
            print('loss:',self.loss)
        # 6. 模型迁回 CPU，清显存（如果后面这一段时间不用它算梯度的话）
        self.model.to("cpu")
        torch.cuda.empty_cache()
        return delta_w

    def update_FFA_LoRA(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        for name, param in self.model.named_parameters():
            if 'lora_A' in name:
                param.requires_grad = False
        self.data_id_loader(index)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        step = 0  # 新增步数计数
        self.loss =0
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                step += 1  # 步数+1
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1)
                self.optimizer.step()
                step += 1  # 步数+1
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
        if index % 4 == 0:
            print('loss:',self.loss)
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
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                self.loss += loss.item() / args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                step += 1  # 步数+1
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v.cpu() - weights[k].cpu()
        if index % 4 == 0:
            print('loss:', self.loss)
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

    def update_FedACG(self, weights, E, index, ps_c, lr):
        if ps_c == {}:
            ps_c = {k: torch.zeros_like(v, device='cpu') for k, v in self.model.state_dict().items()}
        for k, v in ps_c.items():
            weights[k] = weights[k].cpu() + ps_c[k].cpu() * args.gamma
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                               weight_decay=0.01, eps=1e-8)
        step = 0  # 新增步数计数
        self.loss =0
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                self.loss += loss.item() / args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1)
                self.optimizer.step()
                step += 1  # 步数+1
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k].cpu()
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k].cpu()
        if index % 4 == 0:
            print('loss:', self.loss)
        # 6. 模型迁回 CPU，清显存（如果后面这一段时间不用它算梯度的话）
        self.model.to("cpu")
        torch.cuda.empty_cache()
        return delta_w

    def load_dict(self):
        self.func_dict = {
            'FedSAM': self.update_SAM,
            'SCAFFOLD': self.update_scaf,  # scaf
            'FedLORA': self.update_FedIT,
            'FLORA': self.update_FLORA,
            'FFA_LoRA': self.update_FFA_LoRA,
            'FedIT': self.update_FedIT,
            'FedSVD': self.update_FedIT,
            'LORA_FAIR': self.update_FedIT,
            'RoLoRA': self.update_RoLoRA,
            'FRLoRA': self.update_FedIT,
            'Fedfull': self.update_FedIT,
            'FedACG': self.update_FedACG,
            'FedGalore': self.update_Fedgalore,
            'FedGarare': self.update_Fedgarare,

        }

    def update_func(self, alg, weights, E, index, lr, ps_c=None, v=None,step=None,shared_state=None):
        self.load_dict()
        if alg in { 'FedCM','FedACG'}:
            return self.func_dict.get(alg, None)(weights, E, index, ps_c, lr)
        if alg in {'SCAFFOLD'}:
            return self.func_dict.get(alg, None)(weights, E, index, ps_c, lr)
        else:
            return self.func_dict.get(alg, None)(weights, E, index, lr)

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


def apply_weights_FLORA( num_workers, weights,model):

    sum_weights ={}
    lora_config = LoraConfig(
        r=int(args.r*args.selection*num_workers),  # LoRA attention dimension
        lora_alpha=args.r*2,  # Alpha scaling
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,  # Sequence classification
        target_modules=['query', 'value', 'key'],  # Target modules to apply LoRA
        modules_to_save=None
    )
    for weight in weights:
        for k, v in weight.items():
            if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                if 'lora_A' in k :
                    new = [sum_weights[k], v / (num_workers * selection)]
                    sum_weights[k] = torch.cat(new, dim=0)
                elif 'lora_B' in k:
                    new = [sum_weights[k], v ]
                    sum_weights[k] = torch.cat(new, dim=1)
                else:
                    sum_weights[k] = v/(num_workers * selection)
            else:
                if 'lora_A' in k:
                    sum_weights[k] = v / (num_workers * selection)
                elif 'lora_B' in k:
                    sum_weights[k] = v * 1
                else:
                    sum_weights[k] += v/ (num_workers * selection)
    model = get_peft_model(model, lora_config)
    model.load_state_dict(sum_weights,strict=False)
    model.merge_and_unload()
    return {k: v.cpu() for k, v in model.state_dict().items()}
def apply_weights_LORA_SVD(num_workers, weights, model, selection=args.selection):
    # 1) 先用 AB->∆W 平均->SVD 的方式得到新的 A/B
    new_lora_state = aggregate_AB_then_SVD(
        weights=weights,
        r=int(args.r),
        num_workers=num_workers,
        selection=selection
    )
    lora_only = {k: v for k, v in new_lora_state.items() if "lora" in k}
    #set_peft_model_state_dict(model, lora_only)
    model.load_state_dict(lora_only,strict=False)
    return {k: v.cpu() for k, v in model.state_dict().items()}

def apply_weights_LORA_SVD2(num_workers, weights, model, selection=args.selection):
    # 1) 先用 AB->∆W 平均->SVD 的方式得到新的 A/B
    new_lora_state = aggregate_AB_then_SVD(
        weights=weights,
        r=int(args.r),
        num_workers=num_workers,
        selection=selection
    )
    lora_only = {k: v for k, v in new_lora_state.items() if "lora" in k}
    global_weights = model.state_dict()
    for k, v in lora_only.items():  # w = w + delta_w
        global_weights[k] = global_weights[k].cpu() + lora_only[k].cpu()
    model.load_state_dict(global_weights)
    return {k: v.cpu() for k, v in model.state_dict().items()}
def apply_weights_FRLoRA(num_workers, weights, model, selection=1.0):
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
    #for k, v in sum_weights.items():  # w = w + delta_w
    #    ps_w[k] = ps_w[k] + sum_weights[k]
    model.load_state_dict(sum_weights,strict=False)
    return model.state_dict(),ps_c
@torch.no_grad()
def apply_weights_avg(num_workers, weights,model):
    ps_w = {k: v.cpu() for k, v in model.state_dict().items()}
    sum_weights = {k: torch.zeros_like(v) for k, v in ps_w.items() if "lora" in k}
    scale = 1.0 / (num_workers * selection)
    # 聚合 delta_wi
    for weight in weights:
        for k, v in weight.items():
            if 'lora' in k and args.lora==1:
                sum_weights[k].add_(v, alpha=scale)  # inplace 加法
    model.load_state_dict(sum_weights,strict=False)
    return {k: v.cpu() for k, v in model.state_dict().items()}
@torch.no_grad()
def apply_weights_FedLORA(num_workers, weights,model):
    ps_w = {k: v.cpu() for k, v in model.state_dict().items()}
    sum_weights = {k: torch.zeros_like(v) for k, v in ps_w.items()}
    scale = 1.0 / (num_workers * selection)
    # 聚合 delta_wi
    for weight in weights:
        for k, v in weight.items():
            if 'lora' in k and args.lora==1:
                sum_weights[k].add_(v, alpha=scale)  # inplace 加法
            else:
                sum_weights[k].add_(v, alpha=scale)
    for k in ps_w.keys():
        ps_w[k].add_(sum_weights[k])  # inplace 加法
    model.load_state_dict(ps_w)
    return {k: v.cpu() for k, v in model.state_dict().items()}
@torch.no_grad()
def apply_weights_avg_full(num_workers, weights,model):
    ps_w = model.state_dict()  # w : ps_w
    #print(ps_w.keys())
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



if __name__ == "__main__":
    # 获取args
    gpu_idx = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    set_random_seed(seed=seed)
    epoch = args.epoch
    num_workers = args.num_workers
    batch_size = args.batch_size
    lr = args.lr
    E = args.E
    lr_decay = args.lr_decay  # for CIFAR10
    # lr_decay = 1
    alg = args.alg
    data_name = args.data_name
    selection = args.selection
    alpha_value = args.alpha_value
    alpha = args.alpha
    extra_name = args.extname
    check = args.check
    T_part = args.T_part
    c_dict = {}
    lr_decay = args.lr_decay
    import time
    localtime = time.asctime(time.localtime(time.time()))
    checkpoint_path = './checkpoint/ckpt-{}-{}-{}-{}-{}-{}'.format(alg, lr, extra_name, alpha_value, extra_name,
                                                                   localtime)
    c_dict = {}  # state dict
    assert alg in {
        'FedSAM',
        'SCAFFOLD',
        'FedLORA',
        'FLORA',
        'FFA_LoRA',
        'FedIT',
        'FedSVD',
        'LORA_FAIR',
        'RoLoRA',
        'FRLoRA',
        'Fedfull',
        'FedACG',
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

    nums_cls = 2
    if data_name == 'sst2':
        nums_cls = 2
    if data_name == 'MNLI' or args.data_name == 'SNLI':
        nums_cls = 3

    nums_sample = 500
    if data_name == 'sst2':
        nums_sample = int(67349/ (args.num_workers))
    if data_name == ' STS-B':
        nums_sample = int(67349/ (args.num_workers))
    if data_name == 'cola':
        nums_sample = int(8551/ (args.num_workers))
    if data_name == 'qnli':
        nums_sample = int(104743/ (args.num_workers))
    if data_name == 'MRPC':
        nums_sample = int(5801/ (args.num_workers))
    if data_name == 'RTE':
        nums_sample = int(2490/ (args.num_workers))
    if data_name == 'MNLI':
        nums_sample = int(392702/ (args.num_workers))
    if data_name == 'QQP':
        nums_sample = int(363846 / (args.num_workers))
    if args.data_name == 'SNLI':
        nums_sample = int(363846 / (args.num_workers))
    if data_name == 'AG_News':
        nums_sample = int(120000 / args.num_workers)
    if data_name == 'DBPedia_14':
        nums_sample = int(560000 / args.num_workers)
    if args.data_name == 'IMDB':
        nums_sample = int(50000 / args.num_workers)
    if args.data_name == 'ANLI':
        nums_sample = int(len(tokenized_dataset["train"]) / args.num_workers)

    import pickle
    filename = 'data_idx.data'
    if args.alpha_value==1:
        f = open(filename, 'rb')
        data_idx = pickle.load(f)
    else:
        data_idx, std = data_from_dirichlet(data_name, alpha_value,num_labels, num_workers, nums_sample)
    ray.init(ignore_reinit_error=True, num_gpus=num_gpus)

    if args.CNN == 'bert':
        model_path = '../glfl/BERT'
        model = BertForSequenceClassification.from_pretrained(model_path)
    if args.CNN == 'roberta_base':
        model_path = './roberta_base'
        model = RobertaForSequenceClassification.from_pretrained(model_path,num_labels=num_labels)
        #if args.data_name == 'MNLI' or args.data_name == 'SNLI':
        #    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=3)


    model=model.to(device)
    epoch_s = 0
    # c_dict = None,None
    workers = [DataWorker.remote(i, data_idx, num_workers,
                                 lr, batch_size=batch_size, alg=alg, data_name=data_name, selection=selection,
                                 T_part=T_part) for i in range(int(num_workers * selection))]

    logger.info('extra_name:{},alg:{},E:{},data_name:{}, epoch:{}, lr:{},alpha_value:{},alpha:{},CNN:{},gamma:{}'
                .format(extra_name, alg, E, data_name, epoch, lr, alpha_value, alpha, args.CNN, args.gamma))
    # logger.info('data_idx{}'.format(data_idx))
    test_loader = get_data_loader_test(data_name)
    train_loader = get_data_loader_train(data_name)
    print("@@@@@ Running synchronous parameter server training @@@@@@")

    if args.lora == 1 and args.alg!='FLORA':
        model = get_peft_model(model, lora_config)
    current_weights=model.state_dict()
    ps_c=None
    result_list, X_list = [], []
    result_list_loss = []
    test_list_loss = []
    start = time.time()
    # for early stop
    best_acc = 0
    no_improve = 0
    momen_m={}
    momen_v = {}
    ps_c={}
    div = []
    sim = []
    step = torch.tensor([0], dtype=torch.float32, device='cpu')
    for epochidx in range(epoch_s, epoch):
        start_time1 = time.time()
        lr = lr * lr_decay
        if args.lr_decay==2:
            eta_max=args.lr
            eta_min=0
            t=epochidx
            T=args.epoch
            lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * t / T))
        index = np.arange(num_workers)  # 100
        np.random.shuffle(index)
        index = index[:int(num_workers * selection)]  # 10id
        index = np.sort(index)

        if alg in {'FedAvg','FedSAM'}:
            weights = []
            index_sel = index
            weights =  [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                 zip(workers, index_sel)]
            weights=ray.get(weights)
            time3 = time.time()
            current_weights = apply_weights_FedLORA(num_workers, weights,model)
            model.load_state_dict(current_weights)


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

        elif alg in { 'FedIT','RoLoRA','FFA_LoRA'}:
            weights = []
            index_sel = index
            weights =  [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                 zip(workers, index_sel)]
            weights=ray.get(weights)
            time3 = time.time()
            print(epochidx, '    ', time3 - start_time1)
            current_weights = apply_weights_avg(num_workers, weights,model)
            model.load_state_dict(current_weights)

        elif alg in { 'FedLORA'}:
            weights = []
            index_sel = index
            weights =  [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                 zip(workers, index_sel)]
            weights=ray.get(weights)
            time3 = time.time()
            print(epochidx, '    ', time3 - start_time1)
            #current_weights = apply_weights_LORA_SVD2(num_workers, weights, model, selection=1.0)
            current_weights = apply_weights_FedLORA(num_workers, weights,model)
            model.load_state_dict(current_weights)


        elif alg in {'FedSVD'}:
            weights = []
            index_sel = index
            weights =  [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                 zip(workers, index_sel)]
            weights=ray.get(weights)
            time3 = time.time()
            print(epochidx, '    ', time3 - start_time1)
            current_weights = apply_weights_LORA_SVD(num_workers, weights, model, selection=args.selection)
            model.load_state_dict(current_weights)

        elif alg in {'LORA_FAIR'}:
            weights = []
            index_sel = index
            weights =  [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                 zip(workers, index_sel)]
            weights=ray.get(weights)
            time3 = time.time()
            print(epochidx, '    ', time3 - start_time1)
            current_weights = apply_weights_lora_fair(
                num_workers=num_workers,
                weights=weights,  # 客户端上报的 LoRA(绝对)参数字典列表
                model=model,
                selection=selection,  # 参与比例（此处仅做均值，不额外缩放）
                iters=50,  # 可调
                lr=0.01,  # 可调
                lambda_reg=0.01  # 可调
            )
            model.load_state_dict(current_weights)

        elif alg in {'FRLoRA'}:
            weights = []
            index_sel = index
            weights = [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                       zip(workers, index_sel)]
            weights = ray.get(weights)
            time3 = time.time()
            print(epochidx, '    ', time3 - start_time1)
            current_weights = apply_weights_FRLoRA(num_workers, weights, model, selection=args.selection)
            model.load_state_dict(current_weights)


        elif alg in {'FLORA'}:
            weights = []
            index_sel = index
            weights =  [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                 zip(workers, index_sel)]
            weights=ray.get(weights)
            time3 = time.time()
            #print(epochidx, '    ', time3 - start_time1)
            current_weights = apply_weights_FLORA(num_workers, weights,model)
            current_weights =  {
                    k.replace("base_model.model.", ""): v
                    for k, v in current_weights.items()
                    #if "lora_" not in k
                }
            model.load_state_dict(current_weights)

        elif alg in {'Fedfull','FedGalore','FedGarare'}:
            weights = []
            index_sel = index
            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights = weights + [worker.update_func.remote(alg, current_weights, E, idx, lr) for
                                     worker, idx in
                                     zip(workers, index_sel)]
            #weights =  [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
            #                     zip(workers, index_sel)]
            weights=ray.get(weights)
            time3 = time.time()
            print(epochidx, '    ', time3 - start_time1)
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
        print(epochidx, '    ', end_time1 - start_time1)
        args.i = 1
        if epochidx % args.preprint == 0:
            start_time1 = time.time()
            print('测试')
            test_loss = 0
            train_loss = 0
            accuracy, test_loss, train_loss = evaluate(model, test_loader, train_loader)
            model.to('cpu')
            torch.cuda.empty_cache()
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
                "Iter {}: \t accuracy is {:.2f}, train loss is {:.5f}, test loss is {:.5f}, no improve:{}, name:{},lr:{:.7f},CNN:{},GPU:{},gamma:{},alpha:{},r:{},alpha_value:{},data:{}".format(
                    epochidx, accuracy,
                    loss_train_median, test_loss,
                    no_improve, args.alg, lr, args.CNN, args.gpu, args.gamma, args.alpha, args.r, args.alpha_value,
                    args.data_name))
            print(
                "Iter {}: \t accuracy is {:.2f}, train loss is {:.5f}, test loss is {:.5f}, no improve:{}, name:{},lr:{:.7f},CNN:{},GPU:{},data:{},gamma:{},alpha:{},r:{},alpha_value:{}".format(
                    epochidx, accuracy,
                    loss_train_median, test_loss,
                    no_improve, args.alg, lr, args.CNN, args.gpu, args.data_name, args.gamma, args.alpha,
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
    save_name = './plot/alg_{}-data_{}-E_{}-#wk_{}-ep_{}-lr_{}-alpha_value_{}-selec_{}-alpha{}-{}-gamma{}-r{}-CNN{}-optimizer{}-time{}'.format(
        alg,args.data_name, E, num_workers, epoch,
        lr, alpha_value, selection, alpha,
        extra_name, args.gamma, args.r, args.CNN, args.optimizer, endtime)
    save_name = save_name + '.npy'
    np.save(save_name, (x, result, result_loss, test_list_loss))
    ray.shutdown()