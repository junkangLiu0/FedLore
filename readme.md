
# FedLore: Communication and Memory Efficient  Federated Learning  via Gradient Low-Rank  Projection
# Federated Learning Framework README


---

## Quick Start
## Requirements

* Python 3.8
* PyTorch
* torchvision
* numpy
* matplotlib
* tensorboardX
* ray==1.0.0
* filelock

You can install the dependencies with:

```bash
pip install -r requirements.txt
```

### 2.  Vision Transformer Training
```bash
python  main_FedLore.py --alg LORA_FAIR --lr 1e-3 --data_name imagenet --alpha_value 0.1  --epoch 101  --extname FedMerge --lr_decay 2 --gamma 0.9  --CNN  swin_base --E 5 --batch_size 16   --gpu 1 --p 1 --num_gpus_per 0.2 --selection 0.1 --pre 1 --num_workers 50 --preprint 10 --rho 0.01 --lora 1 --K 50 --r 8  --alpha  1
python  main_FedLore.py --alg Fedfull --lr 1e-4 --data_name imagenet --alpha_value 0.1  --epoch 101  --extname FedMerge --lr_decay 2 --gamma 0.9  --CNN  swin_base --E 5 --batch_size 16   --gpu 2 --p 1 --num_gpus_per 0.2 --selection 0.1 --pre 1 --num_workers 50 --preprint 10 --rho 0.01 --lora 0 --K 50 --r 8  --alpha  1
python  main_FedLore.py --alg LORA_FAIR --lr 1e-3 --data_name imagenet --alpha_value 0.1  --epoch 101  --extname FedMerge --lr_decay 2 --gamma 0.9  --CNN  swin_base --E 5 --batch_size 16   --gpu 1 --p 1 --num_gpus_per 0.2 --selection 0.1 --pre 1 --num_workers 50 --preprint 10 --rho 0.01 --lora 1 --K 50 --r 8  --alpha  1
python  main_FedLore.py --alg FedIT --lr 1e-3 --data_name imagenet --alpha_value 0.1  --epoch 101  --extname FedMerge --lr_decay 2 --gamma 0.9  --CNN  VIT-B --E 5 --batch_size 16   --gpu 0 --p 1 --num_gpus_per 0.2 --selection 0.1 --pre 1 --num_workers 50 --preprint 10 --rho 0.01 --lora 1 --K 50 --r 8  --alpha  1
python  main_FedLore.py --alg FedSVD --lr 1e-3 --data_name imagenet --alpha_value 0.1  --epoch 101  --extname FedMerge --lr_decay 2 --gamma 0.9  --CNN  VIT-B --E 5 --batch_size 16   --gpu 0 --p 1 --num_gpus_per 0.2 --selection 0.1 --pre 1 --num_workers 50 --preprint 10 --rho 0.01 --lora 1 --K 50 --r 8  --alpha  1
python  main_FedLore.py --alg RoLoRA --lr 1e-3 --data_name imagenet --alpha_value 0.1  --epoch 101  --extname FedMerge --lr_decay 2 --gamma 0.9  --CNN  VIT-B --E 5 --batch_size 16   --gpu 0 --p 1 --num_gpus_per 0.2 --selection 0.1 --pre 1 --num_workers 50 --preprint 10 --rho 0.01 --lora 1 --K 50 --r 8  --alpha  1
python  main_FedLore.py --alg FRLoRA --lr 1e-3 --data_name imagenet --alpha_value 0.1  --epoch 101  --extname FedMerge --lr_decay 2 --gamma 0.9  --CNN  VIT-B --E 5 --batch_size 16   --gpu 0 --p 1 --num_gpus_per 0.2 --selection 0.1 --pre 1 --num_workers 50 --preprint 10 --rho 0.01 --lora 1 --K 50 --r 8  --alpha  1
python  main_FedLore.py --alg FedGalore --lr 1e-3 --data_name imagenet --alpha_value 0.1  --epoch 101  --extname FedMerge --lr_decay 2 --gamma 0.9  --CNN  VIT-B --E 5 --batch_size 16   --gpu 0 --p 1 --num_gpus_per 0.2 --selection 0.1 --pre 1 --num_workers 50 --preprint 10 --rho 0.01 --lora 0 --K 50 --r 8  --alpha  1
```

* 这里解释一下 --num_gpus_per 0.2的意思是如果你用的是4090显卡24g显存，那么你每个客户端将分配0.2张显卡，即4.8g显存。
* --lr_decay 2 解释一下，这个是余弦学习率下降
* --gpu 0 是指使用的是第0块gpu（gpu序号）
* --alpha_value 0.1 是迪利克雷非立同分布常数
* --alpha_value 1 这个时候是iid情况
* --lora 1 是否使用lora微调
* --data_name timy imagenet数据集需要自己下载，网址在下面
* --lora 1 使用lora微调
* --batch_size 16 显存限制原因，16效果还可以
* --num_gpus_per 0.2 五个客户端，每个客户端使用0.2张卡
* --lr 1e-3 这个学习率微调lora最好

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

The code supports multiple datasets:

* **CIFAR-10 / CIFAR-100**
* **Tiny-ImageNet**

## 🤖 **大语言模型训练示例（RoBERTa-base + GLUE-SST2）**
```bash
python  llm_FedLore.py --alg LORA_FAIR --lr 2e-4 --data_name QQP  --alpha_value 0.8  --epoch 101  --extname FedMerge --lr_decay 2 --gamma 0.9  --CNN  roberta_base --E 5 --batch_size 16   --gpu 0 --p 1 --num_gpus_per 0.25 --selection 0.2 --pre 1 --num_workers 20 --preprint 10 --rho 0.01 --lora 1 --K 50 --r 8  --alpha  1
python  llm_FedLore.py --alg LORA_FAIR --lr 2e-4 --data_name  sst2  --alpha_value 0.8  --epoch 101  --extname FedMerge --lr_decay 2 --gamma 0.9  --CNN  roberta_base --E 5 --batch_size 16   --gpu 0 --p 1 --num_gpus_per 0.25 --selection 0.2 --pre 1 --num_workers 20 --preprint 10 --rho 0.01 --lora 1 --K 50 --r 8  --alpha  1
python  llm_FedLore.py --alg FFA_LoRA --lr 2e-4 --data_name QQP  --alpha_value 0.8  --epoch 101  --extname FedMerge --lr_decay 2 --gamma 0.9  --CNN  roberta_base --E 5 --batch_size 16   --gpu 4 --p 1 --num_gpus_per 0.25 --selection 0.2 --pre 1 --num_workers 20 --preprint 10 --rho 0.01 --lora 1 --K 50 --r 8  --alpha  1
python  llm_FedLore.py --alg FedSVD --lr 2e-4 --data_name QQP  --alpha_value 0.8  --epoch 101  --extname FedMerge --lr_decay 2 --gamma 0.9  --CNN  roberta_base --E 5 --batch_size 16   --gpu 4 --p 1 --num_gpus_per 0.25 --selection 0.2 --pre 1 --num_workers 20 --preprint 10 --rho 0.01 --lora 1 --K 50 --r 8  --alpha  1
python  llm_FedLore.py --alg RoLoRA --lr 2e-4 --data_name QQP  --alpha_value 0.8  --epoch 101  --extname FedMerge --lr_decay 2 --gamma 0.9  --CNN  roberta_base --E 5 --batch_size 16   --gpu 5 --p 1 --num_gpus_per 0.25 --selection 0.2 --pre 1 --num_workers 20 --preprint 10 --rho 0.01 --lora 1 --K 50 --r 8  --alpha  1
python  llm_FedLore.py --alg FedGalore --lr 2e-4 --data_name QQP  --alpha_value 0.8  --epoch 101  --extname FedMerge --lr_decay 2 --gamma 0.9  --CNN  roberta_base --E 5 --batch_size 16   --gpu 4 --p 1 --num_gpus_per 0.25 --selection 0.2 --pre 1 --num_workers 20 --preprint 10 --rho 0.01 --lora 0 --K 50 --r 8  --alpha  1
```
数据集和模型权重下载地址：
* RoBERTa_base模型权重下载地址，下载完之后放入 roberta_base 文件夹即可。
https://huggingface.co/FacebookAI/roberta-base/tree/main

* 数据集下载地址在hugging face上
  sst2 https://huggingface.co/datasets/SetFit/sst2/tree/main
 全部数据集下载地址：
https://huggingface.co/datasets/Junkang2/glue/tree/main



## 🤖 **大语言模型预训练**
```bash
python  FedLore_llama.py --alg FedAvg_adamw --lr 3e-4 --data_name C4 --alpha_value 1 --alpha  100  --epoch 101  --extname FedMuon --lr_decay 2 --gamma 0.5  --CNN   llama_60M --E 5 --batch_size 16   --gpu 0 --p 1 --num_gpus_per 0.1 --selection 1 --print 0 --pre 1 --num_workers 4 --preprint 2 --beta1 0.9 --beta2 0.999 --rho 0.01 --lora 0 --K 50
python  FedLore_llama.py --alg FedAvg_Galore --lr 3e-3 --data_name C4 --alpha_value 1 --alpha  100  --epoch 101  --extname FedMuon --lr_decay 2 --gamma 0.5  --CNN   llama_60M --E 5 --batch_size 16   --gpu 0 --p 1 --num_gpus_per 0.1 --selection 1 --print 0 --pre 1 --num_workers 4 --preprint 2 --rho 0.01 --lora 0 --K 50 --r 128
python  FedLore_llama.py --alg FedIT --lr 1e-3 --data_name C4 --alpha_value 1 --alpha  100  --epoch 101  --extname FedMuon --lr_decay 2 --gamma 0.5  --CNN   llama_60M --E 5 --batch_size 16   --gpu 0 --p 1 --num_gpus_per 0.1 --selection 1 --print 0 --pre 1 --num_workers 4 --preprint 2 --rho 0.01 --lora 1 --K 50 --r 128
```
## Parameter Reference

### Core Federated Learning Parameters
| Parameter | Description |
|-----------|-------------|
| `--alg` | Algorithm choice: `FedIT`, `FedSVD`, etc. |
| `--lr` | Client learning rate |
| `--lr_decay` | Learning rate decay strategy (1=exponential, 2=cosine annealing) |
| `--gamma` | Momentum parameter for certain algorithms |
| `--alpha` | Weight decay coefficient for AdamW optimizer |

### Data Parameters
| Parameter | Description |
|-----------|-------------|
| `--data_name` | Dataset:  `CIFAR100`, `imagenet`, `QQP`, `MNLI`, etc. |
| `--alpha_value` | Dirichlet distribution parameter for non-IID data splitting (0.1=highly non-IID, 1=IID) |
| `--num_workers` | Total number of clients |
| `--selection` | Fraction of clients selected per round (0.1=10%) |

### Model Parameters
| Parameter | Description |
|-----------|-------------|
| `--CNN` | Model architecture: `roberta_base` |
| `--pre` | Use pretrained weights (1=True, 0=False) |
| `--normalization` | Normalization type: `BN` (BatchNorm) or `GN` (GroupNorm) |

### Training Parameters
| Parameter | Description |
|-----------|-------------|
| `--epoch` | Total communication rounds |
| `--E` | Local epochs per client |
| `--batch_size` | Client batch size |
| `--K` | Maximum local steps per round (overrides E if smaller) |
| `--p` | Parallelism factor for client updates |

### LoRA Parameters
| Parameter | Description |
|-----------|-------------|
| `--lora` | Enable LoRA fine-tuning (1=True, 0=False) |
| `--r` | LoRA rank |
| `--lora_alpha` | LoRA scaling parameter |

### Optimization Parameters
| Parameter | Description |
|-----------|-------------|
| `--rho` | SAM optimizer perturbation radius |
| `--optimizer` | Base optimizer: `SGD` or `AdamW` |

### System Parameters
| Parameter | Description |
|-----------|-------------|
| `--gpu` | GPU device IDs (e.g., "0,1,2") |
| `--num_gpus_per` | GPU fraction per client (0.2=20% of a GPU) |
| `--print` | Print detailed logs (1=True, 0=False) |
| `--preprint` | Evaluation frequency (in epochs) |

---

## Output Files

- **Logs**: `./log/alg-dataset-lr-workers-batch-epochs-lr_decay.txt`
- **Checkpoints**: `./checkpoint/ckpt-alg-lr-extname-alpha_value-timestamp/`
- **Plots**: `./plot/alg-dataset-...-timestamp.npy` (contains accuracy/loss arrays)
- **Models**: `./model/model-alg-...-timestamp.pth`

---

## Notes

1. **LoRA Usage**: When `--lora 1`, only LoRA parameters are trainable by default
2. **Pretrained Models**: Automatically downloads required pretrained weights
3. **Data Splitting**: Uses Dirichlet distribution for non-IID splits when `--alpha_value < 1`
4. **Memory**: Adjust `--num_gpus_per` based on your GPU memory capacity

For transformer training with GLUE tasks, use `new_llm.py` with appropriate `--data_name` (QQP, MNLI, SST2, etc.).


# 🌌 **联邦学习实验平台 · 中文文档**  
*（支持 CNN & Transformer 双栈训练）*

---

## 📂 一键安装依赖
```bash
# 基础环境
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 联邦学习 & 日志
pip install ray==1.0.0 tensorboardX==2.6.2.2 tqdm==4.67.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Transformer & 数据集
pip install transformers==4.46.3 datasets==3.1.0 peft==0.13.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 科学计算
pip install scikit-learn==1.3.2 scipy==1.9.3 matplotlib==3.7.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
