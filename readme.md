
<div align="center">

# FedLore Communication- and Memory-Efficient Federated Learning via Gradient Low-Rank Projection

[![Conference](https://img.shields.io/badge/NeurIPS-2026-blueviolet.svg)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)]()
[![Federated Learning](https://img.shields.io/badge/Federated%20Learning-Low--Rank%20Projection-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

</div>

---

## Overview

**FedLore** is a communication- and memory-efficient federated learning framework for large-scale models.

Federated learning enables collaborative training without centralizing private data, but scaling FL to large vision and language models is difficult because clients face severe memory and communication bottlenecks. Full-parameter federated optimization requires transmitting large model updates and maintaining expensive optimizer states, while LoRA-based FL methods reduce cost by freezing the backbone and training only fixed low-rank adapters.

FedLore rethinks low-rank learning in federated optimization.

Instead of treating low-rank matrices as trainable adapter modules, FedLore uses low-rank projection as a dynamic optimization coordinate system. At each communication round, a shared low-rank projector is constructed and sent to participating clients. Clients optimize in the shared projected space, upload compact low-rank updates, and the server reconstructs these updates into the original parameter space for aggregation.

In this way, FedLore preserves the flexibility of full-parameter learning while keeping the client-side budget close to LoRA-style methods.

---


## Repository Structure

```text
FedLore/
├── main_FedLore.py              # Main vision-side federated training script
├── dirichlet_data2.py           # Dirichlet non-IID data partitioning
├── lora_SVD.py                  # Low-rank aggregation utilities
├── lora_fair.py                 # LoRA-FAIR aggregation utilities
├── DomainNet.py                 # DomainNet dataset loader
├── dataset.py                   # Tiny-ImageNet dataset loader
├── model.py                     # Swin Transformer backbone
├── vit_model.py                 # ViT-Base backbone
├── models/
│   └── DeiTTiny.py              # ViT-Tiny / DeiT-Tiny style backbone
├── sam.py                       # SAM optimizer utilities
├── requirements.txt
└── README.md
```
下载模型权重网址：
下载下来的权重直接放主文件夹下面就行，你也可以自己该目类

vit-base：
https://huggingface.co/Junkang2/vit/tree/main

swin_transformer 
https://huggingface.co/Junkang2/swin_transformer/tree/main

## Dataset

数据集下载网址

Tiny-ImageNet：
https://huggingface.co/datasets/Junkang2/Tiny-ImageNet/upload/main



## Supported Backbones

The current vision-side implementation keeps the models used in the paper's vision experiments.

| Argument          | Backbone                         |
| :---------------- | :------------------------------- |
| `--CNN VIT-B`     | ViT-Base                         |
| `--CNN swin_base` | Swin-Base                        |
| `--CNN deit_tiny` | ViT-Tiny / DeiT-Tiny style model |


## Supported Datasets

| Argument                         | Dataset                    |
| :------------------------------- | :------------------------- |
| `--data_name CIFAR10`            | CIFAR-10                   |
| `--data_name CIFAR100`           | CIFAR-100                  |
| `--data_name imagenet`           | Tiny-ImageNet-style loader |

## Installation

### Create Environment

```bash
conda create -n fedlore python=3.8 -y
conda activate fedlore
```

### Install PyTorch

Choose the CUDA version according to your machine. For example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Install Dependencies

```bash
pip install numpy scipy scikit-learn matplotlib tqdm tensorboardX ray peft transformers
```

## Installation

```bash
conda create -n fedmuon python=3.8 -y
conda activate fedmuon

pip install torch torchvision
pip install numpy matplotlib filelock tensorboardX ray==1.0.0
pip install peft transformers
```

Recommended package versions used by the original implementation:

```text
python >= 3.8
torch >= 2.0
torchvision >= 0.15
ray == 1.0.0
tensorboardX == 2.6.2.2
peft == 0.13.2
transformers == 4.46.3
```

```bash
pip install -r requirements.txt
```

---

## Data Preparation

### CIFAR-100

The script supports CIFAR-100 through torchvision.

```bash
--data_name CIFAR100
```

### Tiny-ImageNet

Place Tiny-ImageNet under:

```text
./data/tiny-imagenet-200/
```

Expected structure:

```text
data/
└── tiny-imagenet-200/
    ├── train/
    ├── val/
    └── test/
```
---

## Quick Start

### FedLore on CIFAR-100 with ViT-Base

This is the recommended concise command.

```bash
python main_FedLore.py --alg FedGalore --data_name CIFAR100 --alpha_value 0.1 --num_workers 50 --selection 0.1 --K 50 --lr 1e-3 --r 8 --lora 0 --gpu 0 --num_gpus_per 0.2 --extname FedLore_CIFAR100_ViTB
```

This command relies on the following default values in the code:

| Default Argument | Value   |
| :--------------- | :------ |
| `--CNN`          | `VIT-B` |
| `--epoch`        | `100`   |
| `--E`            | `1`     |
| `--batch_size`   | `16`    |
| `--lr_decay`     | `0.99`  |

Equivalent expanded form:

```bash
python  main_FedLore.py --alg FedIT --lr 1e-3 --data_name CIFAR100 --alpha_value 0.1  --epoch 101  --extname FedMerge --lr_decay 2 --gamma 0.9  --CNN  VIT-B --E 5 --batch_size 16   --gpu 0 --p 1 --num_gpus_per 0.2 --selection 0.1 --pre 1 --num_workers 50 --preprint 10 --rho 0.01 --lora 1 --K 50 --r 8  --alpha  1
```

---

## More Running Examples

### FedLore on Tiny-ImageNet with Swin-Base

```bash
python  main_FedLore.py --alg FedIT --lr 1e-3 --data_name CIFAR100 --alpha_value 0.1  --epoch 101  --extname FedMerge --lr_decay 2 --gamma 0.9  --CNN  swin_base --E 5 --batch_size 16   --gpu 0 --p 1 --num_gpus_per 0.2 --selection 0.1 --pre 1 --num_workers 50 --preprint 10 --rho 0.01 --lora 1 --K 50 --r 8  --alpha  1
```

### FedLore on CIFAR-100 with Swin-Base

```bash
python  main_FedLore.py --alg FedIT --lr 1e-3 --data_name imagenet --alpha_value 0.1  --epoch 101  --extname FedMerge --lr_decay 2 --gamma 0.9  --CNN  swin_base --E 5 --batch_size 16   --gpu 0 --p 1 --num_gpus_per 0.2 --selection 0.1 --pre 1 --num_workers 50 --preprint 10 --rho 0.01 --lora 1 --K 50 --r 8  --alpha  1
```

### FedLore on CIFAR-100 with ViT-Tiny

```bash
python  main_FedLore.py --alg FedIT --lr 1e-3 --data_name CIFAR100 --alpha_value 0.1  --epoch 101  --extname FedMerge --lr_decay 2 --gamma 0.9  --CNN  swin_base --E 5 --batch_size 16   --gpu 0 --p 1 --num_gpus_per 0.2 --selection 0.1 --pre 1 --num_workers 50 --preprint 10 --rho 0.01 --lora 1 --K 50 --r 8  --alpha  1
```

---

## Baseline Commands

### FedIT

```bash
python  main_FedLore.py --alg FedIT --lr 1e-3 --data_name CIFAR100 --alpha_value 0.1  --epoch 101  --extname FedMerge --lr_decay 2 --gamma 0.9  --CNN  swin_base --E 5 --batch_size 16   --gpu 0 --p 1 --num_gpus_per 0.2 --selection 0.1 --pre 1 --num_workers 50 --preprint 10 --rho 0.01 --lora 1 --K 50 --r 8  --alpha  1
```

### LoRA-FAIR

```bash
python  main_FedLore.py --alg LORA_FAIR --lr 1e-3 --data_name CIFAR100 --alpha_value 0.1  --epoch 101  --extname FedMerge --lr_decay 2 --gamma 0.9  --CNN  swin_base --E 5 --batch_size 16   --gpu 0 --p 1 --num_gpus_per 0.2 --selection 0.1 --pre 1 --num_workers 50 --preprint 10 --rho 0.01 --lora 1 --K 50 --r 8  --alpha  1
```

### FFA-LoRA

```bash
python  main_FedLore.py --alg FFA_LoRA --lr 1e-3 --data_name CIFAR100 --alpha_value 0.1  --epoch 101  --extname FedMerge --lr_decay 2 --gamma 0.9  --CNN  swin_base --E 5 --batch_size 16   --gpu 0 --p 1 --num_gpus_per 0.2 --selection 0.1 --pre 1 --num_workers 50 --preprint 10 --rho 0.01 --lora 1 --K 50 --r 8  --alpha  1
```

### RoLoRA

```bash
python  main_FedLore.py --alg RoLoRA --lr 1e-3 --data_name CIFAR100 --alpha_value 0.1  --epoch 101  --extname FedMerge --lr_decay 2 --gamma 0.9  --CNN  swin_base --E 5 --batch_size 16   --gpu 0 --p 1 --num_gpus_per 0.2 --selection 0.1 --pre 1 --num_workers 50 --preprint 10 --rho 0.01 --lora 1 --K 50 --r 8  --alpha  1
```

### FLoRA

```bash
python main_FedLore.py --alg FLORA --data_name CIFAR100 --alpha_value 0.1 --num_workers 50 --selection 0.1 --K 50 --lr 1e-3 --r 8 --lora 1 --gpu 0 --num_gpus_per 0.2 --extname FLoRA_CIFAR100_ViTB
```

### FRLoRA

```bash
python main_FedLore.py --alg FRLoRA --data_name CIFAR100 --alpha_value 0.1 --num_workers 50 --selection 0.1 --K 50 --lr 1e-3 --r 8 --lora 1 --gpu 0 --num_gpus_per 0.2 --extname FRLoRA_CIFAR100_ViTB
```

### FedFull

```bash
python main_FedLore.py --alg Fedfull --data_name CIFAR100 --alpha_value 0.1 --num_workers 50 --selection 0.1 --K 50 --lr 1e-4 --r 8 --lora 0 --gpu 0 --num_gpus_per 0.2 --extname FedFull_CIFAR100_ViTB
```

---

## Important Arguments

| Argument         | Description                                                                                                        |
| :--------------- | :----------------------------------------------------------------------------------------------------------------- |
| `--alg`          | Federated algorithm, such as `FedGalore`, `FedIT`, `Fedfull`, `LORA_FAIR`, `FFA_LoRA`, `RoLoRA`, `FLORA`, `FRLoRA` |
| `--CNN`          | Backbone model. Choose from `VIT-B`, `swin_base`, `deit_tiny`                                                      |
| `--data_name`    | Dataset name, such as `CIFAR10`, `CIFAR100`, `imagenet`, `domainnet_real`                                          |
| `--alpha_value`  | Dirichlet concentration parameter for non-IID data partitioning                                                    |
| `--num_workers`  | Total number of federated clients                                                                                  |
| `--selection`    | Client participation ratio per communication round                                                                 |
| `--epoch`        | Number of communication rounds                                                                                     |
| `--E`            | Number of local epochs                                                                                             |
| `--K`            | Maximum number of local update steps                                                                               |
| `--batch_size`   | Local mini-batch size                                                                                              |
| `--lr`           | Client learning rate                                                                                               |
| `--lr_decay`     | Learning-rate decay factor                                                                                         |
| `--r`            | Low-rank dimension                                                                                                 |
| `--lora`         | Whether to enable LoRA adapters. Use `1` for LoRA-style methods and `0` for full/projected training                |
| `--gpu`          | GPU id, for example `0` or `0,1`                                                                                   |
| `--num_gpus_per` | Fractional GPU resource allocated to each Ray worker                                                               |
| `--extname`      | Extra experiment name for logs and checkpoints                                                                     |
| `--weights`      | Optional pretrained weights path for ViT-Base or Swin-Base                                                         |

---

## Non-IID Federated Partition

FedLore uses Dirichlet partitioning to simulate heterogeneous federated data.

```bash
--alpha_value 0.1
```

means strong non-IID heterogeneity.

```bash
--alpha_value 1
```

corresponds to a more balanced or IID-like split.

| `alpha_value` | Data Heterogeneity      |
| :------------ | :---------------------- |
| `0.1`         | Strong non-IID          |
| `0.5`         | Moderate non-IID        |
| `1.0`         | Mild non-IID / near-IID |

---

## Paper-Style Experimental Setting

Representative vision-side settings:

| Setting             | Value                           |
| :------------------ | :------------------------------ |
| Number of clients   | `50`                            |
| Participation ratio | `0.1`                           |
| Local steps         | `K = 50`                        |
| Batch size          | `16`                            |
| Low-rank dimension  | `r = 8`                         |
| Non-IID split       | Dirichlet                       |
| Backbone            | ViT-Base / Swin-Base / ViT-Tiny |

Recommended command:

```bash
python main_FedLore.py --alg FedGalore --data_name CIFAR100 --alpha_value 0.1 --num_workers 50 --selection 0.1 --K 50 --lr 1e-3 --r 8 --lora 0 --gpu 0 --num_gpus_per 0.2 --extname FedLore_CIFAR100_ViTB
```

---

## Output Files

The code typically saves logs, checkpoints, curves, and model files under the following directories:

```text
log/
checkpoint/
plot/
model/
tmp/ray/
```

Suggested organization:

| Directory     | Description          |
| :------------ | :------------------- |
| `log/`        | Training logs        |
| `checkpoint/` | Checkpoints          |
| `plot/`       | Accuracy/loss arrays |
| `model/`      | Saved model weights  |
| `tmp/ray/`    | Ray runtime files    |

---

## Practical Tips

### GPU Allocation

`--num_gpus_per` controls how much GPU resource each Ray worker occupies.

For example:

```bash
--num_gpus_per 0.2
```

means each worker is assigned approximately 20% of one GPU resource. On a 24GB GPU, this roughly corresponds to scheduling about five workers per GPU, depending on the actual memory usage of the backbone.

For larger models or out-of-memory issues, try:

```bash
--num_gpus_per 0.25
```

or:

```bash
--num_gpus_per 0.5
```

### Batch Size

A safe default is:

```bash
--batch_size 16
```

If CUDA memory is insufficient, reduce it to:

```bash
--batch_size 8
```

### Rank

A typical vision-side rank is:

```bash
--r 8
```

Larger ranks may improve optimization but increase memory and communication cost.

### LoRA Flag

For LoRA-style methods:

```bash
--lora 1
```

For FedLore / GaLore-style projected training:

```bash
--lora 0
```

---

## Troubleshooting

### CUDA Out of Memory

Reduce one or more of the following arguments:

```bash
--batch_size
--num_gpus_per
--selection
--K
```

Example:

```bash
python main_FedLore.py --alg FedGalore --data_name CIFAR100 --alpha_value 0.1 --num_workers 50 --selection 0.1 --K 50 --lr 1e-3 --r 8 --lora 0 --gpu 0 --num_gpus_per 0.5 --extname FedLore_CIFAR100_ViTB
```

### Ray Memory Monitor Error

The script disables Ray memory monitor by default:

```python
os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"
```

If Ray temporary files cause issues, remove old runtime files:

```bash
rm -rf ./tmp/ray
```

### Dataset Path Not Found

Check whether the dataset is placed under:

```text
./data/
```

For Tiny-ImageNet:

```text
./data/tiny-imagenet-200/
```

For DomainNet:

```text
./data/domainnet/
```

or:

```text
./data/dominnet/
```

---

## Reproducibility Checklist

To reproduce paper-style experiments, ensure that:

* The same backbone is used.
* The same dataset is used.
* The same Dirichlet `alpha_value` is used.
* The same number of clients is used.
* The same client participation ratio is used.
* The same local update steps `K` are used.
* The same rank `r` is used.
* Multiple seeds are averaged for final reporting.

Recommended seeds:

```text
42, 43, 44, 45, 46
```

---

## Example Experiment Matrix

| Method    | `--alg`     | `--lora` | Typical LR |
| :-------- | :---------- | :------- | :--------- |
| FedIT     | `FedIT`     | `1`      | `1e-3`     |
| LoRA-FAIR | `LORA_FAIR` | `1`      | `1e-3`     |
| FFA-LoRA  | `FFA_LoRA`  | `1`      | `1e-3`     |
| RoLoRA    | `RoLoRA`    | `1`      | `1e-3`     |
| FLoRA     | `FLORA`     | `1`      | `1e-3`     |
| FRLoRA    | `FRLoRA`    | `1`      | `1e-3`     |
| FedFull   | `Fedfull`   | `0`      | `1e-4`     |
| FedLore   | `FedGalore` | `0`      | `1e-3`     |

---

## Citation

If you find this repository useful, please consider citing:

```bibtex
@inproceedings{fedlore2026,
  title     = {FedLore: Communication and Memory Efficient Federated Learning via Gradient Low-Rank Projection},
  author    = {Anonymous Authors},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

---

## Acknowledgements

FedLore builds on ideas from federated learning, parameter-efficient fine-tuning, LoRA-style adaptation, and gradient low-rank projection.

The goal of this project is to make large-scale federated optimization more practical through:

* low-rank communication,
* memory-efficient optimization,
* shared subspace alignment,
* and full-space model updates.

---

## License

This project is released under the MIT License.

---

<div align="center">

### FedLore

**Low-rank client budgets. Full-space federated learning.**

</div>
```
