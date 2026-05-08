import math

from dataclasses import dataclass
from typing import Dict, List, Any

import argparse
from soap import SOAP
from sophia import SophiaG
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 必须放在任何 tokenizer/transformers import 之前


import os
os.environ["RAY_TMPDIR"] = "/data/ray_tmp"   # 换成你有空间的路径
os.makedirs(os.environ["RAY_TMPDIR"], exist_ok=True)
from peft import LoraConfig, get_peft_model, TaskType

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
parser.add_argument('--CNN', default='llama_60M', type=str, help=' for mom_step')
parser.add_argument('--gamma', default=0.9, type=float, help=' for mom_step')
parser.add_argument('--weights', type=str, default='./swin_tiny_patch4_window7_224.pth',
                    help='initial weights path')
# 是否冻结权重
parser.add_argument('--p', default=1, type=int, help=' for mom_step')
parser.add_argument('--freeze-layers', type=bool, default=False)
parser.add_argument('--datapath', type=str,
                    default="./data")
parser.add_argument('--num_gpus_per', default=0.5, type=float, help=' for mom_step')
parser.add_argument('--rho', default=0.1, type=float, help='rho')
parser.add_argument('--optimizer', default='SGD', type=str, help='SGD,AdamW')
parser.add_argument("--preprint", type=int, default=5, help="")
parser.add_argument("--R", type=int, default=1, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--lora", type=int, default=0, help="")
parser.add_argument("--AdaLora", type=int, default=0, help="")
parser.add_argument("--r", type=int, default=16, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--beta1", type=float, default=0.9, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--beta2", type=float, default=0.999, help="the perturbation radio for the SAM optimizer.")
parser.add_argument('--K', default=20, type=int, help='#workers')
parser.add_argument('--freeze', default=1, type=int, help='# batch_size')
parser.add_argument("--pre", type=int, default=1, help="the perturbation radio for the SAM optimizer.")
parser.add_argument('--print', default=0, type=int, help=' for mom_step')

parser.add_argument("--tokenizer_name_or_path", type=str, default="hf-internal-testing/llama-tokenizer")
parser.add_argument("--local_c4_glob", type=str, default="./C4/c4-train.*.json.gz")
parser.add_argument("--text_column", type=str, default="text")

parser.add_argument("--output_dir", type=str, default='/data/llama60m_c4')
parser.add_argument("--seq_len", type=int, default=1024)

parser.add_argument("--per_device_train_batch_size", type=int, default=2)
parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--warmup_steps", type=int, default=0)
parser.add_argument("--max_steps", type=int, default=50)
parser.add_argument("--logging_steps", type=int, default=50)
parser.add_argument("--save_steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--shuffle_buffer", type=int, default=10_000)
parser.add_argument("--map_batch_size", type=int, default=1000)
parser.add_argument("--take_text_samples", type=int, default=0)


parser.add_argument("--local_c4_val_glob", type=str, default="./C4/c4-validation.*.json.gz")
parser.add_argument("--eval_max_batches", type=int, default=50, help="评估最多跑多少个batch（控制时间）")
parser.add_argument("--eval_batch_size", type=int, default=2, help="评估batch size")


args = parser.parse_args()
print(args.lora)
gpu_idx = args.gpu
print('gpu_idx', gpu_idx)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import functional as F
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast  # 用于混合精度训练
#from muon import MuonWithAuxAdam
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import SubsetRandomSampler, random_split
import random
from math import exp
from copy import deepcopy
import ray
import os
import os

# 1) 先改 Ray 的临时目录（换成你有空间且可写的路径）
os.environ["RAY_TMPDIR"] = "/data/ray_tmp"   # 例如 /data, /scratch, /home/zjc/tmp 都行
os.environ["TMPDIR"] = os.environ["RAY_TMPDIR"]  # 保险：有些组件也看 TMPDIR
os.makedirs(os.environ["RAY_TMPDIR"], exist_ok=True)

# 2) 你的其它 env 也放这里
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from tensorboardX import SummaryWriter
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, \
    RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig, AutoTokenizer, LlamaConfig, LlamaForCausalLM
from dirichlet_data import data_from_dirichlet

from transformers import RobertaTokenizer, RobertaForSequenceClassification, Adafactor, Trainer, TrainingArguments
from datasets import load_dataset, tqdm


print(torch.cuda.is_available())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(device)
num_gpus_per = args.num_gpus_per  # num_gpus_per = 0.16
# num_gpus_per = 0.5
num_gpus = len(gpu_idx.split(','))

data_name = args.data_name
CNN = args.CNN

target_modules = [
    "q_proj", "k_proj", "v_proj",
    "up_proj", "down_proj", "gate_proj", "o_proj"
]
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=args.r,
    lora_alpha=args.r,
    lora_dropout=0.1,
    target_modules=target_modules,
    modules_to_save=None,
    bias="none",
)


def build_llama_130m_config(vocab_size: int, seq_len: int) -> LlamaConfig:
    # ~130M（会随 vocab_size 略有浮动）
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=640,
        intermediate_size=2560,   # 4x hidden
        num_hidden_layers=12,
        num_attention_heads=10,   # 640 / 10 = 64
        num_key_value_heads=10,
        max_position_embeddings=seq_len,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
        tie_word_embeddings=True,
    )

def build_llama_350m_config(vocab_size: int, seq_len: int) -> LlamaConfig:
    # ~350M（随 vocab_size 略浮动）
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=1024,
        intermediate_size=4096,   # 4x hidden
        num_hidden_layers=16,
        num_attention_heads=16,   # 1024 / 16 = 64
        num_key_value_heads=16,
        max_position_embeddings=seq_len,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
        tie_word_embeddings=True,
    )


def build_llama_150m_config(vocab_size: int, seq_len: int) -> LlamaConfig:
    # ~160M（随 vocab_size 略浮动）
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=768,
        intermediate_size=3072,  # 4x hidden
        num_hidden_layers=12,
        num_attention_heads=12,  # 768/12=64
        num_key_value_heads=12,
        max_position_embeddings=seq_len,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
        tie_word_embeddings=True,
    )


def build_llama_60m_config(vocab_size: int, seq_len: int) -> LlamaConfig:
    # 约 60M（随 vocab_size 略浮动）
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=10,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=seq_len,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
        tie_word_embeddings=True,
    )



def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


import itertools
from torch.utils.data import IterableDataset
class SkipTorchIterableDataset(IterableDataset):
    def __init__(self, base_iterable, skip_n: int):
        self.base_iterable = base_iterable
        self.skip_n = int(skip_n)

    def __iter__(self):
        return itertools.islice(iter(self.base_iterable), self.skip_n, None)



@dataclass
class CausalCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # features 必须都有 input_ids/labels/attention_mask
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)

        labels = labels.masked_fill(input_ids.eq(self.pad_token_id), -100)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}



if args.data_name=='C4':
    seq_len = args.seq_len
    text_col = args.text_column

    args.tokenizer_name_or_path='/data/zjc/LORA+SAM2/tiny_llama/'


    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Dataset: 本地 json.gz（streaming）----
    train_ds = load_dataset(
        "json",
        data_files={"train": args.local_c4_glob},
        split="train",
        streaming=True,
    )

    tokenizer.model_max_length = 10 ** 9  # 禁止这种“超过 max_length”的警告

    #next(iter(train_ds))

if args.CNN=='llama_350M':
    config = build_llama_350m_config(vocab_size=len(tokenizer), seq_len=seq_len)
if args.CNN=='llama_130M':
    config = build_llama_130m_config(vocab_size=len(tokenizer), seq_len=seq_len)
if args.CNN=='llama_150M':
    config = build_llama_150m_config(vocab_size=len(tokenizer), seq_len=seq_len)
if args.CNN=='llama_60M':
    config = build_llama_60m_config(vocab_size=len(tokenizer), seq_len=seq_len)

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
use_fp16 = torch.cuda.is_available() and (not use_bf16)


seed = 42

_C4_TRAIN_FILES = None
import glob
import itertools
from datasets import load_dataset
_C4_TRAIN_FILES = None
def _get_c4_train_files():
    global _C4_TRAIN_FILES
    if _C4_TRAIN_FILES is None:
        files = sorted(glob.glob(args.local_c4_glob))
        if not files:
            raise FileNotFoundError(f"No files matched: {args.local_c4_glob}")
        _C4_TRAIN_FILES = files
    return _C4_TRAIN_FILES


def _get_all_columns(ds, fallback_text_col: str):
    """
    尽量不消耗样本地拿到所有列名，用于 remove_columns（必须删掉 timestamp/url 等）
    """
    cols = getattr(ds, "column_names", None)
    if cols:
        return list(cols)

    feats = getattr(ds, "features", None)
    if feats:
        return list(feats.keys())

    # 极端兜底：只在拿不到 metadata 时消耗 1 条样本（建议只发生极少数情况）
    try:
        sample = next(iter(ds))
        return list(sample.keys())
    except Exception:
        return [fallback_text_col]


def get_data_loader(index, data_idx, batch_size, data_name, skip_samples: int = 0):
    # ---- 1) 按文件分配，不做样本级 shard ----
    files = _get_c4_train_files()
    client_files = files[index::args.num_workers]
    if len(client_files) == 0:
        client_files = [files[index % len(files)]]

    train_ds = load_dataset(
        "json",
        data_files={"train": client_files},
        split="train",
        streaming=True,
    )

    if args.take_text_samples and args.take_text_samples > 0:
        train_ds = train_ds.take(args.take_text_samples)

    eos_id = tokenizer.eos_token_id
    seq_len = args.seq_len
    text_col = args.text_column

    def tokenize_batch(examples):
        texts = examples[text_col]
        out = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=False,   # 我们后面自己切块
        )
        out["input_ids"] = [ids + [eos_id] for ids in out["input_ids"]]
        return out

    def group_texts(examples):
        concatenated = list(itertools.chain.from_iterable(examples["input_ids"]))
        total_len = (len(concatenated) // seq_len) * seq_len
        if total_len == 0:
            return {"input_ids": [], "labels": [], "attention_mask": []}

        concatenated = concatenated[:total_len]
        blocks = [concatenated[i:i + seq_len] for i in range(0, total_len, seq_len)]
        return {
            "input_ids": blocks,
            "labels": blocks.copy(),
            "attention_mask": [[1] * seq_len for _ in range(len(blocks))],
        }

    # ---- 2) 关键：tokenize 这一步必须 remove 掉所有原始列（含 timestamp/url/...）----
    original_cols = _get_all_columns(train_ds, fallback_text_col=text_col)

    train_ds = train_ds.map(
        tokenize_batch,
        batched=True,
        batch_size=args.map_batch_size,
        remove_columns=original_cols,   # ★ 必须删全，否则会出现 timestamp mismatch
    )

    # group_texts 后只会生成 3 列
    train_ds = train_ds.map(
        group_texts,
        batched=True,
        batch_size=64,
    )

    # 旧逻辑兼容：如果你还没完全删掉 offset/skip，可先保留
    if skip_samples and skip_samples > 0:
        skip_samples = int(skip_samples)
        if hasattr(train_ds, "skip"):
            train_ds = train_ds.skip(skip_samples)
        else:
            train_ds = SkipTorchIterableDataset(train_ds, skip_samples)

    return train_ds





from transformers import TrainerCallback

class LossTrackingCallback(TrainerCallback):
    def __init__(self):
        self.losses = []  # 用于存储每一步的损失

    def on_log(self, args, state, control, logs=None, **kwargs):
        # 检查日志中是否包含 'loss' 字段
        if 'loss' in logs:
            self.losses.append(logs['loss'])  # 存储损失值
            print(f"Step {state.global_step}, Loss: {logs['loss']}")  # 打印每步的损失

    def get_average_loss(self):
        # 计算并返回所有损失的平均值
        return sum(self.losses) / len(self.losses) if self.losses else None





#计算困惑度指标
def get_eval_dataset_c4():
    val_ds = load_dataset(
        "json",
        data_files={"validation": args.local_c4_val_glob},
        split="validation",
        streaming=True,
    )

    eos_id = tokenizer.eos_token_id
    seq_len = args.seq_len
    text_col = args.text_column

    def tokenize_batch(examples):
        texts = examples[text_col]
        out = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)
        out["input_ids"] = [ids + [eos_id] for ids in out["input_ids"]]
        return out

    def group_texts(examples):
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)

        total_len = (len(concatenated) // seq_len) * seq_len
        if total_len == 0:
            return {"input_ids": [], "labels": [], "attention_mask": []}

        concatenated = concatenated[:total_len]
        blocks = [concatenated[i: i + seq_len] for i in range(0, total_len, seq_len)]
        return {
            "input_ids": blocks,
            "labels": blocks.copy(),
            "attention_mask": [[1] * seq_len for _ in range(len(blocks))],
        }

    # remove_columns：按你 train 的写法拿列名
    cols = getattr(val_ds, "column_names", None)
    if cols:
        original_cols = list(cols)
    else:
        feats = getattr(val_ds, "features", None)
        if feats:
            original_cols = list(feats.keys())
        else:
            sample = next(iter(val_ds))
            original_cols = list(sample.keys())

    val_ds = val_ds.map(
        tokenize_batch,
        batched=True,
        batch_size=args.map_batch_size,
        remove_columns=original_cols,
    )

    val_ds = val_ds.map(
        group_texts,
        batched=True,
        batch_size=64,
    )

    return val_ds
import torch.nn.functional as F

@torch.no_grad()
def compute_ppl_on_streaming_dataset(model, eval_ds, collator, device, max_batches=50, batch_size=2):
    model.eval()
    model.to(device)

    loader = DataLoader(eval_ds, batch_size=batch_size, collate_fn=collator)

    total_nll = 0.0
    total_tokens = 0
    n_batches = 0

    for batch in loader:
        n_batches += 1
        if max_batches and n_batches > max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # [B, T, V]

        # causal shift
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        mask = (shift_labels != -100)
        token_count = mask.sum().item()
        if token_count == 0:
            continue

        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)

        batch_nll = (loss_per_token * mask).sum().item()
        total_nll += batch_nll
        total_tokens += token_count

    avg_nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(avg_nll)

    return avg_nll, ppl, total_tokens, n_batches


import itertools
import torch.utils.data as tud
import torch.utils.data as tud

class MaterializedSliceDataset(tud.Dataset):
    """
    从 iterator 里取出 n 条样本，物化成可索引 Dataset，兼容 HF Trainer/accelerate。
    每条样本应是 dict: {'input_ids','labels','attention_mask'}。
    """
    def __init__(self, it, n: int, max_tries_mul: int = 20):
        super().__init__()
        n = int(n)
        data = []
        tries = 0
        max_tries = n * max_tries_mul

        while len(data) < n and tries < max_tries:
            tries += 1
            try:
                ex = next(it)
            except StopIteration:
                break

            # 跳过空样本（group_texts 可能产生空输出）
            if not isinstance(ex, dict):
                continue
            ids = ex.get("input_ids", None)
            if ids is None or len(ids) == 0:
                continue

            data.append(ex)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


@ray.remote(num_gpus=num_gpus_per)
class DataWorker(object):

    def __init__(self, pid, data_idx, num_workers, lr, batch_size, alg, data_name, selection, T_part):
        self.alg = alg

        self.model = LlamaForCausalLM(config)
        if args.lora == 1:
            self.model = get_peft_model(self.model, peft_config)
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
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01,
                                           betas=(args.beta1, args.beta2), amsgrad=False)

        self.client_iters = {}  # client_id -> iterator
        self.client_ds = {}  # client_id -> hf iterable ds (optional cache)

    def _build_client_ds(self, client_id: int):
        ds = get_data_loader(client_id, self.data_idx, batch_size, data_name, skip_samples=0)  # 不要skip
        return ds

    def _get_client_iter(self, client_id: int):
        if client_id not in self.client_iters:
            ds = self._build_client_ds(client_id)
            self.client_ds[client_id] = ds
            self.client_iters[client_id] = iter(ds)
        return self.client_iters[client_id]

    def data_id_loader(self, index, shared_state=None):
        # 取该 client 的“已消费样本数”
        skip_n = 0
        if shared_state is not None:
            skip_n = ray.get(shared_state.get_offset.remote(index))

        self.train_ds = get_data_loader(index, self.data_idx, batch_size, data_name, skip_samples=skip_n)

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

    from torch.utils.data import DataLoader
    class LossTrackingCallback(TrainerCallback):
        def __init__(self):
            self.losses = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if 'loss' in logs:
                self.losses.append(logs['loss'])
                print(f"Step {state.global_step}, Loss: {logs['loss']}")

        def get_average_loss(self):
            return sum(self.losses) / len(self.losses) if self.losses else None


    def update_FedIT(self, weights, E, index, lr, shared_state=None):
        self.model.load_state_dict(weights)
        self.model.to(device)
        before_lora = {}
        for k, v in self.model.state_dict().items():
            if "lora_A" in k or "lora_B" in k:
                before_lora[k] = v.detach().cpu().clone()

        it = self._get_client_iter(index)
        # 你要跑的优化步数
        target_steps = int(args.max_steps)
        micro_bs = int(args.per_device_train_batch_size)
        grad_acc = int(args.gradient_accumulation_steps)
        # 一步优化需要 micro_bs * grad_acc 个样本（近似）
        need_samples = target_steps * micro_bs * grad_acc
        # 给点冗余（因为 group_texts 可能产出空样本）
        train_slice = MaterializedSliceDataset(it, need_samples * 2)
        # 根据实际拿到的样本量，限制这轮最多能跑多少个优化 step
        max_steps_possible = len(train_slice) // (micro_bs * grad_acc)
        if max_steps_possible <= 0:
            zero_delta = {k: torch.zeros_like(v).cpu() for k, v in weights.items()}
            return zero_delta, None
        real_max_steps = min(target_steps, max_steps_possible)
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            disable_tqdm=True,
            per_device_train_batch_size=micro_bs,
            gradient_accumulation_steps=grad_acc,
            learning_rate=lr,
            weight_decay=0.01,
            warmup_steps=0,
            max_steps=real_max_steps,  # ★ 关键：用 real_max_steps
            lr_scheduler_type="constant",
            logging_steps=args.logging_steps,
            save_strategy="no",
            eval_strategy="no",
            bf16=use_bf16,
            fp16=use_fp16,
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.99,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
        )
        collator = CausalCollator(pad_token_id=tokenizer.pad_token_id)
        # 初始化自定义回调
        loss_tracking_callback = LossTrackingCallback()
        # 初始化 Trainer，传入回调
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_slice,  # ★ 用物化后的 map-style dataset
            data_collator=collator,
            processing_class=tokenizer,
            callbacks=[loss_tracking_callback],
        )
        # 重置 AdamW 优化器状态（重新初始化二阶矩）
        optimizer = trainer.optimizer
        if optimizer is not None:
            optimizer.state = {}  # 清空优化器的状态，重置动量估计
        trainer.train()
        consumed_samples = (
            training_args.max_steps
            * training_args.gradient_accumulation_steps
            * training_args.per_device_train_batch_size
        )
        if shared_state is not None:
            shared_state.inc_offset.remote(index, consumed_samples)
        avg_train_loss = loss_tracking_callback.get_average_loss()
        delta_w = {}
        for k, v in self.model.state_dict().items():
            if "lora_A" in k or "lora_B" in k:
                delta_w[k] = v.detach().cpu() - before_lora[k]
        #delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
        #for k, v in self.model.state_dict().items():
        #    delta_w[k] = v.cpu() - weights[k]
        print(f"Average Training Loss: {avg_train_loss}")
        return delta_w, avg_train_loss

    def update_fedavg_adamw(self, weights, E, index, lr, shared_state=None):
        self.model.load_state_dict(weights)
        self.model.to(device)
        it = self._get_client_iter(index)
        # 你要跑的优化步数
        target_steps = int(args.max_steps)
        micro_bs = int(args.per_device_train_batch_size)
        grad_acc = int(args.gradient_accumulation_steps)
        # 一步优化需要 micro_bs * grad_acc 个样本（近似）
        need_samples = target_steps * micro_bs * grad_acc
        # 给点冗余（因为 group_texts 可能产出空样本）
        train_slice = MaterializedSliceDataset(it, need_samples * 2)
        # 根据实际拿到的样本量，限制这轮最多能跑多少个优化 step
        max_steps_possible = len(train_slice) // (micro_bs * grad_acc)
        if max_steps_possible <= 0:
            # 本轮数据不足：返回 0 delta，避免 trainer 崩
            zero_delta = {k: torch.zeros_like(v).cpu() for k, v in weights.items()}
            return zero_delta, None

        real_max_steps = min(target_steps, max_steps_possible)

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            disable_tqdm=True,
            per_device_train_batch_size=micro_bs,
            gradient_accumulation_steps=grad_acc,
            learning_rate=lr,
            weight_decay=0.01,
            warmup_steps=0,
            max_steps=real_max_steps,  # ★ 关键：用 real_max_steps
            lr_scheduler_type="constant",
            logging_steps=args.logging_steps,
            save_strategy="no",
            eval_strategy="no",
            bf16=use_bf16,
            fp16=use_fp16,
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.99,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
        )
        collator = CausalCollator(pad_token_id=tokenizer.pad_token_id)
        # 初始化自定义回调
        loss_tracking_callback = LossTrackingCallback()
        # 初始化 Trainer，传入回调
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_slice,  # ★ 用物化后的 map-style dataset
            data_collator=collator,
            processing_class=tokenizer,
            callbacks=[loss_tracking_callback],
        )
        # 重置 AdamW 优化器状态（重新初始化二阶矩）
        optimizer = trainer.optimizer
        if optimizer is not None:
            optimizer.state = {}  # 清空优化器的状态，重置动量估计
        # 开始训练
        trainer.train()
        # ---- 本轮消费的“样本条数”估算：dataloader batch 数 = max_steps * grad_accum
        consumed_samples = (
            training_args.max_steps
            * training_args.gradient_accumulation_steps
            * training_args.per_device_train_batch_size
        )
        if shared_state is not None:
            shared_state.inc_offset.remote(index, consumed_samples)
        # 获取训练后的平均损失
        avg_train_loss = loss_tracking_callback.get_average_loss()
        # 计算权重差异
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v.cpu() - weights[k]
        # 输出平均训练损失
        print(f"Average Training Loss: {avg_train_loss}")
        return delta_w, avg_train_loss

    def update_fedavg_galore(self, weights, E, index, lr, shared_state=None):
        self.model.load_state_dict(weights)
        self.model.to(device)

        it = self._get_client_iter(index)

        target_steps = int(args.max_steps)
        micro_bs = int(args.per_device_train_batch_size)
        grad_acc = int(args.gradient_accumulation_steps)

        need_samples = target_steps * micro_bs * grad_acc
        train_slice = MaterializedSliceDataset(it, need_samples * 2)

        max_steps_possible = len(train_slice) // (micro_bs * grad_acc)
        if max_steps_possible <= 0:
            zero_delta = {k: torch.zeros_like(v).cpu() for k, v in weights.items()}
            return zero_delta, None

        real_max_steps = min(target_steps, max_steps_possible)

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            disable_tqdm=True,
            per_device_train_batch_size=micro_bs,
            gradient_accumulation_steps=grad_acc,
            learning_rate=lr,
            weight_decay=0.01,
            warmup_steps=0,
            max_steps=real_max_steps,
            lr_scheduler_type="constant",
            logging_steps=args.logging_steps,
            save_strategy="no",
            eval_strategy="no",
            bf16=use_bf16,
            fp16=use_fp16,
            optim="adamw_torch",   # 占位，不会真的用它
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
        )
        collator = CausalCollator(pad_token_id=tokenizer.pad_token_id)
        loss_tracking_callback = LossTrackingCallback()
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
            # 2) 其余二维参数 -> 用 GaLore
            elif p.ndim == 2:
                galore_params.append(p)
            # 3) 其他参数 -> 普通 AdamW 更新
            else:
                other_params.append(p)
        head_lr = lr  # 分类头
        galore_lr = lr  # GaLore 主体
        other_lr = lr / 10  # 其他参数更小
        param_groups = []
        if len(head_params) > 0:
            param_groups.append({
                "params": head_params,
                "lr": head_lr,
            })
        if len(galore_params) > 0:
            param_groups.append({
                "params": galore_params,
                "lr": galore_lr,
                "rank": args.r,
                "update_proj_gap": args.K,
                "scale": 1,
                "proj_type": "std",
            })
        if len(other_params) > 0:
            param_groups.append({
                "params": other_params,
                "lr": other_lr,
            })
        self.optimizer = GaLoreAdamW(
            param_groups,
            betas=(0.9, 0.999),
            weight_decay=0,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_slice,
            data_collator=collator,
            processing_class=tokenizer,
            callbacks=[loss_tracking_callback],
            optimizers=(self.optimizer, None),
        )
        trainer.train()
        consumed_samples = (
            training_args.max_steps
            * training_args.gradient_accumulation_steps
            * training_args.per_device_train_batch_size
        )
        if shared_state is not None:
            shared_state.inc_offset.remote(index, consumed_samples)
        avg_train_loss = loss_tracking_callback.get_average_loss()
        delta_w = {}
        new_state = self.model.state_dict()
        for k, v in new_state.items():
            delta_w[k] = v.cpu() - weights[k]
        print(f"[GaLore] Average Training Loss: {avg_train_loss}")
        return delta_w, avg_train_loss

    def update_fedavg_muon(self, weights, E, index, lr, shared_state=None):
        self.model.load_state_dict(weights)
        self.model.to(device)
        it = self._get_client_iter(index)
        # 你要跑的优化步数
        target_steps = int(args.max_steps)
        micro_bs = int(args.per_device_train_batch_size)
        grad_acc = int(args.gradient_accumulation_steps)
        # 一步优化需要 micro_bs * grad_acc 个样本（近似）
        need_samples = target_steps * micro_bs * grad_acc
        # 给点冗余（因为 group_texts 可能产出空样本）
        train_slice = MaterializedSliceDataset(it, need_samples * 2)
        # 根据实际拿到的样本量，限制这轮最多能跑多少个优化 step
        max_steps_possible = len(train_slice) // (micro_bs * grad_acc)
        if max_steps_possible <= 0:
            # 本轮数据不足：返回 0 delta，避免 trainer 崩
            zero_delta = {k: torch.zeros_like(v).cpu() for k, v in weights.items()}
            return zero_delta, None
        real_max_steps = min(target_steps, max_steps_possible)
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            disable_tqdm=True,
            per_device_train_batch_size=micro_bs,
            gradient_accumulation_steps=grad_acc,
            learning_rate=lr,
            weight_decay=0.01,
            warmup_steps=0,
            max_steps=real_max_steps,  # ★ 关键：用 real_max_steps
            lr_scheduler_type="constant",
            logging_steps=args.logging_steps,
            save_strategy="no",
            eval_strategy="no",
            bf16=use_bf16,
            fp16=use_fp16,
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
        )
        collator = CausalCollator(pad_token_id=tokenizer.pad_token_id)
        # 初始化自定义回调
        loss_tracking_callback = LossTrackingCallback()
        # 创建 MuON 优化器实例
        from muon import SingleDeviceMuonWithAuxAdam
        hidden_weights = [p for p in self.model.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in self.model.parameters() if p.ndim < 2]
        nonhidden_params = []
        param_groups = [
            dict(params=hidden_weights, use_muon=True,
                 lr=lr*100, weight_decay=0.01),
            dict(params=hidden_gains_biases + nonhidden_params, use_muon=False,
                 lr=lr, betas=(0.9, 0.95), weight_decay=0.01),
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        # 初始化 Trainer，传入回调和自定义优化器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_slice,
            data_collator=collator,
            processing_class=tokenizer,
            callbacks=[loss_tracking_callback],  # 注册回调
            optimizers=(optimizer, None)  # 这里传递自定义优化器和学习率调度器（没有使用）
        )
        # 开始训练
        trainer.train()
        # ---- 本轮消费的“样本条数”估算：dataloader batch 数 = max_steps * grad_accum
        consumed_samples = (
                training_args.max_steps
                * training_args.gradient_accumulation_steps
                * training_args.per_device_train_batch_size
        )
        if shared_state is not None:
            shared_state.inc_offset.remote(index, consumed_samples)
        # 获取训练后的平均损失
        avg_train_loss = loss_tracking_callback.get_average_loss()
        # 计算权重差异
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v.cpu() - weights[k]
        # 输出平均训练损失
        print(f"Average Training Loss: {avg_train_loss}")
        return delta_w, avg_train_loss
    def update_fedavg_SGD(self, weights, E, index, lr, shared_state=None):
        self.model.load_state_dict(weights)
        self.model.to(device)
        it = self._get_client_iter(index)
        # 你要跑的优化步数
        target_steps = int(args.max_steps)
        micro_bs = int(args.per_device_train_batch_size)
        grad_acc = int(args.gradient_accumulation_steps)
        # 一步优化需要 micro_bs * grad_acc 个样本（近似）
        need_samples = target_steps * micro_bs * grad_acc
        # 给点冗余（因为 group_texts 可能产出空样本）
        train_slice = MaterializedSliceDataset(it, need_samples * 2)
        # 根据实际拿到的样本量，限制这轮最多能跑多少个优化 step
        max_steps_possible = len(train_slice) // (micro_bs * grad_acc)
        if max_steps_possible <= 0:
            # 本轮数据不足：返回 0 delta，避免 trainer 崩
            zero_delta = {k: torch.zeros_like(v).cpu() for k, v in weights.items()}
            return zero_delta, None
        real_max_steps = min(target_steps, max_steps_possible)
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            disable_tqdm=True,
            per_device_train_batch_size=micro_bs,
            gradient_accumulation_steps=grad_acc,
            learning_rate=lr,
            weight_decay=0.01,
            warmup_steps=0,
            max_steps=real_max_steps,  # ★ 关键：用 real_max_steps
            lr_scheduler_type="constant",
            logging_steps=args.logging_steps,
            save_strategy="no",
            eval_strategy="no",
            bf16=use_bf16,
            fp16=use_fp16,
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.99,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
        )
        collator = CausalCollator(pad_token_id=tokenizer.pad_token_id)
        # 初始化自定义回调
        loss_tracking_callback = LossTrackingCallback()
        # ---- SGD 优化器（关键改动）
        # momentum 可按需设置：0.0 表示纯 SGD；0.9 常用
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=getattr(args, "sgd_momentum", 0.0),
            weight_decay=0.01,
            nesterov=getattr(args, "sgd_nesterov", False),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_slice,
            data_collator=collator,
            processing_class=tokenizer,
            callbacks=[loss_tracking_callback],
            optimizers=(optimizer, None),  # 不用 scheduler；如果要用再传
        )
        # 开始训练
        trainer.train()
        # ---- 本轮消费的“样本条数”估算：dataloader batch 数 = max_steps * grad_accum
        consumed_samples = (
                training_args.max_steps
                * training_args.gradient_accumulation_steps
                * training_args.per_device_train_batch_size
        )
        if shared_state is not None:
            shared_state.inc_offset.remote(index, consumed_samples)
        # 获取训练后的平均损失
        avg_train_loss = loss_tracking_callback.get_average_loss()
        # 计算权重差异
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v.cpu() - weights[k]
        # 输出平均训练损失
        print(f"Average Training Loss: {avg_train_loss}")
        return delta_w, avg_train_loss
    def update_fedavg_SOAP(self, weights, E, index, lr, shared_state=None):
        self.model.load_state_dict(weights)
        self.model.to(device)
        it = self._get_client_iter(index)
        # 你要跑的优化步数
        target_steps = int(args.max_steps)
        micro_bs = int(args.per_device_train_batch_size)
        grad_acc = int(args.gradient_accumulation_steps)
        # 一步优化需要 micro_bs * grad_acc 个样本（近似）
        need_samples = target_steps * micro_bs * grad_acc
        # 给点冗余（因为 group_texts 可能产出空样本）
        train_slice = MaterializedSliceDataset(it, need_samples * 2)
        # 根据实际拿到的样本量，限制这轮最多能跑多少个优化 step
        max_steps_possible = len(train_slice) // (micro_bs * grad_acc)
        if max_steps_possible <= 0:
            # 本轮数据不足：返回 0 delta，避免 trainer 崩
            zero_delta = {k: torch.zeros_like(v).cpu() for k, v in weights.items()}
            return zero_delta, None
        real_max_steps = min(target_steps, max_steps_possible)
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            disable_tqdm=True,
            per_device_train_batch_size=micro_bs,
            gradient_accumulation_steps=grad_acc,
            learning_rate=lr,
            weight_decay=0.01,
            warmup_steps=0,
            max_steps=real_max_steps,  # ★ 关键：用 real_max_steps
            lr_scheduler_type="constant",
            logging_steps=args.logging_steps,
            save_strategy="no",
            eval_strategy="no",
            bf16=use_bf16,
            fp16=use_fp16,
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.99,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
        )
        collator = CausalCollator(pad_token_id=tokenizer.pad_token_id)
        # 初始化自定义回调
        loss_tracking_callback = LossTrackingCallback()
        # ---- SGD 优化器（关键改动）
        # momentum 可按需设置：0.0 表示纯 SGD；0.9 常用
        optimizer = SOAP(params=self.model.parameters(),lr=lr, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)


        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_slice,
            data_collator=collator,
            processing_class=tokenizer,
            callbacks=[loss_tracking_callback],
            optimizers=(optimizer, None),  # 不用 scheduler；如果要用再传
        )
        # 开始训练
        trainer.train()
        # ---- 本轮消费的“样本条数”估算：dataloader batch 数 = max_steps * grad_accum
        consumed_samples = (
                training_args.max_steps
                * training_args.gradient_accumulation_steps
                * training_args.per_device_train_batch_size
        )
        if shared_state is not None:
            shared_state.inc_offset.remote(index, consumed_samples)
        # 获取训练后的平均损失
        avg_train_loss = loss_tracking_callback.get_average_loss()
        # 计算权重差异
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v.cpu() - weights[k]
        # 输出平均训练损失
        print(f"Average Training Loss: {avg_train_loss}")
        return delta_w, avg_train_loss

    def load_dict(self):
        self.func_dict = {
            'FedAvg_adamw': self.update_fedavg_adamw,
            'FedAvg_Galore': self.update_fedavg_galore,
            'FedIT': self.update_FedIT,
            'FedAvg_muon': self.update_fedavg_muon,
            'FedAvg_SGD': self.update_fedavg_SGD,
            'FedAvg_SOAP': self.update_fedavg_SOAP,
        }

    def update_func(self, alg, weights, E, index, lr, ps_c=None, v=None,step=None,shared_state=None):
        self.load_dict()
        if alg in { 'FedCM','FedCM'}:
            return self.func_dict.get(alg, None)(weights, E, index, ps_c, lr)
        else:
            if alg in {"FedAvg_adamw", "FedAvg_muon",'FedAvg_SGD','FedAvg_SOAP','FedAvg_Galore','FedIT'}:
                return self.func_dict.get(alg, None)(weights, E, index, lr, shared_state)
            return self.func_dict.get(alg, None)(weights, E, index, lr)




@ray.remote
class SharedClientState:
    def __init__(self, num_clients: int):
        self.offset = {i: 0 for i in range(num_clients)}

    def get_offset(self, client_id: int) -> int:
        return int(self.offset.get(client_id, 0))

    def inc_offset(self, client_id: int, delta: int):
        self.offset[client_id] = int(self.offset.get(client_id, 0)) + int(delta)
        return self.offset[client_id]

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


@torch.no_grad()
def apply_weights_avg(num_workers, weights,model):
    start_time1 = time.time()
    loss = [mi for _, mi in weights]
    average_loss = sum(loss) / len(loss) if loss else None  # 防止除以 0 的情况
    print(f"Average m: {average_loss}")
    weights = [w for w, _ in weights]

    ps_w = {k: v.cpu() for k, v in model.state_dict().items()}
    sum_weights = {k: torch.zeros_like(v) for k, v in ps_w.items()}
    scale = 1.0 / (num_workers * selection)
    # 聚合 delta_wi
    for weight in weights:
        for k, v in weight.items():
                sum_weights[k].add_(v, alpha=scale)
    # 将 server 模型加上 delta_w
    for k in ps_w.keys():
        ps_w[k].add_(sum_weights[k])  # inplace 加法
    model.load_state_dict(ps_w)
    end_time1 = time.time()
    print('聚合完毕', '    ', end_time1 - start_time1)
    return {k: v.cpu() for k, v in model.state_dict().items()},average_loss



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
    mi_dict = {}
    vi_dict = {}
    ti_dict = {}
    import time

    localtime = time.asctime(time.localtime(time.time()))

    checkpoint_path = './checkpoint/ckpt-{}-{}-{}-{}-{}-{}'.format(alg, lr, extra_name, alpha_value, extra_name,
                                                                   localtime)
    c_dict = {}  # state dict
    assert alg in {
        'FedAvg',
        'FedAvg_adamw',
        'FedAvg_Galore',
        'FedIT',
        'FedMuon',
        'FedLADA',
        'FedCM',
        'FedAdamW',
        'Local_Soap',
        'Local_Muon',
        'Local_Sophia',
        'FedSoap',
        'FedAvg_muon',
        'FedAvg_SGD',
        'FedAvg_SOAP',
        'FedIT'

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






    import pickle
    if args.alpha_value == 0.6:
        filename = 'data_idx.data'
    if args.alpha_value == 0.1:
        filename = 'data_idx100000_0.1.data'
    filename = 'data_idx100000_0.1.data'
    if args.alpha_value==1:
        f = open(filename, 'rb')
        data_idx = pickle.load(f)
    else:
        data_idx, std = data_from_dirichlet(data_name, alpha_value, nums_cls, num_workers, nums_sample)

    #ray.init(ignore_reinit_error=True, num_gpus=num_gpus)
    ray.init(ignore_reinit_error=True, num_gpus=num_gpus, _temp_dir="/data/ray_tmp")

    shared_state = SharedClientState.remote(num_workers)

    epoch_s = 0
    workers = [DataWorker.remote(i, data_idx, num_workers,
                                 lr, batch_size=batch_size, alg=alg, data_name=data_name, selection=selection,
                                 T_part=T_part) for i in range(int(num_workers * selection))]


    logger.info('extra_name:{},alg:{},E:{},data_name:{}, epoch:{}, lr:{},alpha_value:{},alpha:{},CNN:{},gamma:{}'
                .format(extra_name, alg, E, data_name, epoch, lr, alpha_value, alpha, args.CNN, args.gamma))
    # logger.info('data_idx{}'.format(data_idx))


    print("@@@@@ Running synchronous parameter server training @@@@@@")


    if args.CNN == 'llama_350M':
        config = build_llama_350m_config(vocab_size=len(tokenizer), seq_len=seq_len)
    if args.CNN == 'llama_130M':
        config = build_llama_130m_config(vocab_size=len(tokenizer), seq_len=seq_len)
    if args.CNN == 'llama_60M':
        config = build_llama_60m_config(vocab_size=len(tokenizer), seq_len=seq_len)

    model = LlamaForCausalLM(config).to('cpu')
    print(f"[Info] Model parameters: {count_params(model) / 1e6:.2f}M")


    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and (not use_bf16)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters (M): {total_params / 1e6:.2f}M")



    if args.lora==1:
        model = get_peft_model(model, peft_config)



    current_weights=model.state_dict()
    ps_c=None

    result_list, X_list = [], []
    result_list_loss = []

    train_loss = []
    val_loss = []
    val_ppl = []
    test_list_loss = []
    start = time.time()
    # for early stop
    best_acc = 0
    no_improve = 0
    m = {k: torch.tensor([0], dtype=torch.float32, device='cpu') for k, v in
                    model.named_parameters()}
    v = {k: torch.tensor([0], dtype=torch.float32, device='cpu') for k, v in
                    model.named_parameters()}

    momen_m={}
    momen_v = {}
    ps_c={}

    div = []
    sim = []
    eval_ds = None
    collator_eval = None
    if args.data_name == "C4":
        eval_ds = get_eval_dataset_c4()
        collator_eval = CausalCollator(pad_token_id=tokenizer.pad_token_id)

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

        if alg in {

            'FedAvg', 'FedMoment',
            'FedAvg_adam', 'FedMuon', 'Local_Soap','Local_Muon', 'Local_Sophia',
        }:
            weights = []
            index_sel = index
            weights =  [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                 zip(workers, index_sel)]
            weights=ray.get(weights)
            time3 = time.time()
            current_weights = apply_weights_avg(num_workers, weights,model)
            time4 = time.time()
            model.load_state_dict(current_weights)



        elif alg in { 'FedAvg_adamw','FedAvg_muon','FedAvg_SGD','FedAvg_SOAP', 'FedAvg_Galore',
        'FedIT'}:
            weights = []
            index_sel = index
            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights = weights + [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                     zip(workers, index_sel)]

            weights=ray.get(weights)
            time3 = time.time()
            print(epochidx, '    ', time3 - start_time1)
            current_weights,average_loss = apply_weights_avg(num_workers, weights,model)
            model.load_state_dict(current_weights)




        end_time1 = time.time()
        print(epochidx, '    ', end_time1 - start_time1)
        args.i = 1

        if epochidx % 1 == 0:
            start_time1 = time.time()
            print('测试')
            model.to('cpu')
            torch.cuda.empty_cache()
            end_time1 = time.time()
            print('测试完毕', '    ', end_time1 - start_time1)
            #train_loss_ = average_loss
            loss_train_median = average_loss
            accuracy=0

            # --- 构建验证集（只做一次）---
            #eval_ds = None
            #if args.data_name == "C4":
            #    eval_ds = get_eval_dataset_c4()
            print("评估全局模型 PPL...")
            # 注意：你上面 model.to('cpu') + empty_cache 会影响评估
            # 建议评估时把全局 model 临时搬到 GPU，评估完再搬回 CPU
            if eval_ds is not None:
                collator_eval = CausalCollator(pad_token_id=tokenizer.pad_token_id)


                avg_nll, ppl, ntok, nb = compute_ppl_on_streaming_dataset(
                    model=model,
                    eval_ds=eval_ds,
                    collator=collator_eval,
                    device=device,  # cuda
                    max_batches=args.eval_max_batches,
                    batch_size=args.eval_batch_size,
                )

                print(f"[Global Eval] avg_nll={avg_nll:.4f}, ppl={ppl:.4f}, tokens={ntok}, batches={nb}")

                # 记录到 tensorboard / logger
                writer.add_scalar("eval/avg_nll", avg_nll, epochidx * E)
                writer.add_scalar("eval/ppl", ppl, epochidx * E)
                logger.info(
                    f"Iter {epochidx}: global eval avg_nll={avg_nll:.6f}, ppl={ppl:.6f}, tokens={ntok}, batches={nb}")

            # 评估后再放回 CPU（如果你想省显存）
            model.to("cpu")
            torch.cuda.empty_cache()






            writer.add_scalar('accuracy', accuracy, epochidx * E)
            writer.add_scalar('loss median', loss_train_median, epochidx * E)
            logger.info(
                "Iter {}: \t ppl is {:.2f}, train loss is {:.5f}, val loss is {:.5f}, no improve:{}, name:{},lr:{:.7f},CNN:{},GPU:{},gamma:{},rho:{},alpha_value:{},data:{}".format(
                    epochidx, ppl,
                    loss_train_median, avg_nll,
                    no_improve, args.alg, lr, args.CNN, args.gpu, args.gamma, args.rho, args.alpha_value,
                    args.data_name))

            print(
                "Iter {}: \t ppl is {:.2f}, train loss is {:.5f}, val loss is {:.5f}, no improve:{}, name:{},lr:{:.7f},CNN:{},GPU:{},data:{},gamma:{},rho:{},alpha_value:{}".format(
                    epochidx, ppl,
                    loss_train_median, avg_nll,
                    no_improve, args.alg, lr, args.CNN, args.gpu, args.data_name, args.gamma,
                    args.rho, args.alpha_value))

            if np.isnan(loss_train_median):
                logger.info('nan~~')
                break
            X_list.append(epochidx)
            train_loss.append(loss_train_median)
            val_loss.append(avg_nll)
            val_ppl.append(ppl)

    logger.info("Final accuracy is {:.2f}.".format(accuracy))
    endtime = time.time()
    logger.info('time is pass:{}'.format(endtime - start))
    x = np.array(X_list)
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    val_ppl = np.array(val_ppl)
    save_name = './plot/alg_{}-data_{}-E_{}-#wk_{}-ep_{}-lr_{}-alpha_value_{}-selec_{}-alpha{}-{}-gamma{}-r{}-CNN{}-optimizer{}-time{}'.format(
        alg,args.data_name, E, num_workers, epoch,
        lr, alpha_value, selection, alpha,
        extra_name, args.gamma, args.r, args.CNN, args.optimizer, endtime)
    save_name = save_name + '.npy'
    np.save(save_name, (x, train_loss,val_loss , val_ppl))
    ray.shutdown()