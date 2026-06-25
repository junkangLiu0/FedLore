
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

## Key Idea

LoRA-style federated learning is efficient, but it restricts all optimization to a fixed adapter subspace. This fixed-subspace constraint can become too rigid under strong non-IID data and is especially limiting for pre-training.

FedLore introduces a dynamic shared-subspace paradigm.

At round `t`, the server constructs a shared projector `P_t`. Each selected client receives the global model and this projector, performs local optimization in the low-rank coordinate system, and uploads only the compact low-rank update `ΔZ_i`. Since the server already knows `P_t`, it can reconstruct the full-space update `P_t ΔZ_i` and aggregate updates in the original parameter space.

```text
Server builds shared projector P_t
        ↓
Clients optimize in the shared low-rank space
        ↓
Clients upload compact low-rank updates ΔZ_i
        ↓
Server reconstructs full-space updates P_t ΔZ_i
        ↓
Server aggregates and updates the global model
````

---

## Highlights

* Dynamic low-rank subspace switching across communication rounds
* Shared projection across clients to reduce projection drift
* Low-rank client upload with full-space server aggregation
* Low optimizer memory through projected-coordinate optimization
* Strong compatibility with transformer-based vision backbones
* Ray-based parallel federated client simulation
* Dirichlet non-IID data partitioning
* Cleaned vision-side implementation aligned with the paper setting

---

## Method Comparison

| Method          | Optimization Space                 | Client Upload | Optimizer Memory | Key Limitation                               |
| :-------------- | :--------------------------------- | :------------ | :--------------- | :------------------------------------------- |
| FedIT / LoRA-FL | Fixed adapter subspace             | Low           | Low              | Frozen backbone and fixed subspace           |
| FedFull         | Full parameter space               | High          | High             | Expensive communication and optimizer states |
| Local GaLore    | Client-specific projected space    | High          | Low              | Projection drift across clients              |
| **FedLore**     | **Dynamic shared projected space** | **Low**       | **Low**          | Efficient full-space learning                |

FedLore is designed to answer a central question:

> Can federated learning recover the flexibility of full-parameter optimization while keeping client cost close to LoRA?

FedLore answers this through dynamic shared low-rank projection.

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

By default, the script uses:

```bash
--CNN VIT-B
```

Therefore, when running ViT-Base experiments, this argument can be omitted.

---

## Supported Datasets

| Argument                         | Dataset                    |
| :------------------------------- | :------------------------- |
| `--data_name CIFAR10`            | CIFAR-10                   |
| `--data_name CIFAR100`           | CIFAR-100                  |
| `--data_name imagenet`           | Tiny-ImageNet-style loader |
| `--data_name domainnet_real`     | DomainNet Real             |
| `--data_name domainnet_clipart`  | DomainNet Clipart          |
| `--data_name domainnet_painting` | DomainNet Painting         |

The paper-style vision experiments focus on transformer backbones under heterogeneous federated settings, especially CIFAR-100 and Tiny-ImageNet with Dirichlet non-IID partitioning.

---

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

Or install from `requirements.txt`:

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

Run with:

```bash
--data_name imagenet --datapath ./data
```

### DomainNet

Place DomainNet under:

```text
./data/domainnet/
```

The code is also compatible with the older path:

```text
./data/dominnet/
```

Example usage:

```bash
--data_name domainnet_real
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
python main_FedLore.py --alg FedGalore --CNN VIT-B --data_name CIFAR100 --alpha_value 0.1 --num_workers 50 --selection 0.1 --epoch 100 --E 1 --K 50 --batch_size 16 --lr 1e-3 --lr_decay 0.99 --r 8 --lora 0 --gpu 0 --num_gpus_per 0.2 --extname FedLore_CIFAR100_ViTB
```

---

## More Running Examples

### FedLore on Tiny-ImageNet with Swin-Base

```bash
python main_FedLore.py --alg FedGalore --CNN swin_base --data_name imagenet --datapath ./data --alpha_value 0.1 --num_workers 50 --selection 0.1 --K 50 --lr 1e-3 --r 8 --lora 0 --gpu 0 --num_gpus_per 0.2 --extname FedLore_TinyImageNet_SwinBase
```

### FedLore on CIFAR-100 with Swin-Base

```bash
python main_FedLore.py --alg FedGalore --CNN swin_base --data_name CIFAR100 --alpha_value 0.1 --num_workers 50 --selection 0.1 --K 50 --lr 1e-3 --r 8 --lora 0 --gpu 0 --num_gpus_per 0.2 --extname FedLore_CIFAR100_SwinBase
```

### FedLore on CIFAR-100 with ViT-Tiny

```bash
python main_FedLore.py --alg FedGalore --CNN deit_tiny --data_name CIFAR100 --alpha_value 0.1 --num_workers 50 --selection 0.1 --K 50 --lr 1e-3 --r 8 --lora 0 --gpu 0 --num_gpus_per 0.2 --extname FedLore_CIFAR100_ViTTiny
```

---

## Baseline Commands

### FedIT

```bash
python main_FedLore.py --alg FedIT --data_name CIFAR100 --alpha_value 0.1 --num_workers 50 --selection 0.1 --K 50 --lr 1e-3 --r 8 --lora 1 --gpu 0 --num_gpus_per 0.2 --extname FedIT_CIFAR100_ViTB
```

### LoRA-FAIR

```bash
python main_FedLore.py --alg LORA_FAIR --data_name CIFAR100 --alpha_value 0.1 --num_workers 50 --selection 0.1 --K 50 --lr 1e-3 --r 8 --lora 1 --gpu 0 --num_gpus_per 0.2 --extname LoRAFAIR_CIFAR100_ViTB
```

### FFA-LoRA

```bash
python main_FedLore.py --alg FFA_LoRA --data_name CIFAR100 --alpha_value 0.1 --num_workers 50 --selection 0.1 --K 50 --lr 1e-3 --r 8 --lora 1 --gpu 0 --num_gpus_per 0.2 --extname FFALoRA_CIFAR100_ViTB
```

### RoLoRA

```bash
python main_FedLore.py --alg RoLoRA --data_name CIFAR100 --alpha_value 0.1 --num_workers 50 --selection 0.1 --K 50 --lr 1e-3 --r 8 --lora 1 --gpu 0 --num_gpus_per 0.2 --extname RoLoRA_CIFAR100_ViTB
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
