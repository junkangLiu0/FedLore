import os

from PIL import Image  # 确保导入了正确的 PIL.Image 模块
from torch.utils.data import Dataset  # ✅ 这是 PyTorch Dataset


class DomainNet(Dataset):
    """
    CIFAR-like interface:
      - root: 数据根目录（里面有 real_train.txt / real_test.txt 等）
      - domain: 'real'/'clipart'/'sketch'/'quickdraw'/'infograph'/'painting'
      - train: True->xxx_train.txt, False->xxx_test.txt
      - transform: torchvision transforms
    It also provides:
      - targets: list[int]
    """

    def __init__(self, root, domain="real", train=True, transform=None):
        self.root = root
        self.domain = domain
        self.train = train
        self.transform = transform

        split = "train" if train else "test"
        txt_file = os.path.join(root, f"{domain}_{split}.txt")
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"DomainNet txt not found: {txt_file}")

        self.samples = []
        self.targets = []

        with open(txt_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                rel_path = parts[0]
                y = int(parts[1])

                # txt 里一般存相对路径，比如 "real/airplane/xxx.png"
                img_path = os.path.join(root, rel_path)
                self.samples.append((img_path, y))
                self.targets.append(y)

        # 可选：提供 classes 信息（如果你没有 class name 映射，就先不强求）
        # DomainNet 常见是 345 类
        self.num_classes = 345

        print(f"[DomainNet] domain={domain} split={split} loaded {len(self.samples)} samples from {txt_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, y = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # 不中断训练：给个灰图兜底
            print(f"[DomainNet] Error loading {img_path}: {e}")
            img = Image.new("RGB", (224, 224), color="gray")

        if self.transform is not None:
            img = self.transform(img)
        return img, y
