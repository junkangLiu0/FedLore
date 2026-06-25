import torch
from collections import defaultdict
from peft import set_peft_model_state_dict

@torch.no_grad()
def _pair_by_layer(weights):
    """
    把每个客户端的 lora_A/lora_B 先在“客户端内部”按层前缀配成对，再汇总到全局。
    返回: groups[prefix]["pairs"] = [(A_k, B_k), ...] （均为 float32，保留原 dtype/device 用 proto）
    以及每层的原型 proto 与 key（回写时用）
    """
    groups = defaultdict(lambda: {
        "pairs": [],
        "A_key": None, "B_key": None,
        "A_proto": None, "B_proto": None,
    })
    for w in weights:                    # 逐客户端
        local = {}
        for k, v in w.items():            # 逐参数键
            if not isinstance(v, torch.Tensor):
                continue
            if "lora_A" in k:
                prefix = k.split("lora_A")[0].rstrip(".")
                d = local.setdefault(prefix, {})
                d["A"] = v.detach().to(torch.float32)
                if groups[prefix]["A_key"] is None:
                    groups[prefix]["A_key"] = k
                    groups[prefix]["A_proto"] = v
            elif "lora_B" in k:
                prefix = k.split("lora_B")[0].rstrip(".")
                d = local.setdefault(prefix, {})
                d["B"] = v.detach().to(torch.float32)
                if groups[prefix]["B_key"] is None:
                    groups[prefix]["B_key"] = k
                    groups[prefix]["B_proto"] = v
        # 本客户端遍历完后，只把既有 A 又有 B 的层 push 到全局
        for prefix, d in local.items():
            if "A" in d and "B" in d:
                groups[prefix]["pairs"].append((d["A"], d["B"]))
    return groups

def _safe_product(B, A):
    """
    计算 ΔW = B @ A，带常见形状兜底，返回 (d, l)。
    约定/常见：B∈R^{d×r}, A∈R^{r×l}.
    """
    if B.ndim != 2 or A.ndim != 2:
        raise ValueError(f"A/B 必须二维，得到 A{A.shape}, B{B.shape}")
    if B.shape[1] == A.shape[0]:
        return B @ A
    if B.shape[0] == A.shape[1]:
        return B @ A.t()       # A 可能是 (l×r)
    if B.shape[1] == A.shape[1]:
        return B @ A.t()       # 另一种错位
    raise ValueError(f"无法匹配乘法维度：A{A.shape}, B{B.shape}")

def apply_weights_lora_fair(
    num_workers,
    weights,                 # List[Dict[str, Tensor]] 各客户端(绝对)LoRA参数
    model,
    *,
    selection=1.0,           # 参与比例；这里做“均值”不需要额外缩放
    iters=200,               # 优化 ΔB 的迭代步数
    lr=1e-2,                 # 学习率
    lambda_reg=1.0,          # 正则系数 λ
    eps=1e-12
):
    """
    LoRA-FAIR 聚合：对各层先求 A/B 均值与乘积均值，然后仅优化 ΔB 来拟合客户端乘积的平均。
    返回 CPU 版的完整 state_dict（便于后续传输/保存）。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) 客户端内配对，再汇总
    groups = _pair_by_layer(weights)

    # 2) 逐层统计: \hat A, \hat B, \overline{P}
    hat_A, hat_B, avg_prod = {}, {}, {}
    for prefix, it in groups.items():
        pairs = it["pairs"]
        if not pairs:
            continue
        # 均值 A/B
        A_mean = torch.stack([A for A, _ in pairs], 0).mean(0)   # (r,l)
        B_mean = torch.stack([B for _, B in pairs], 0).mean(0)   # (d,r)
        # 均值 ΔW
        prod_mean = torch.stack([_safe_product(B, A) for A, B in pairs], 0).mean(0)  # (d,l)
        hat_A[prefix] = A_mean.to(device)
        hat_B[prefix] = B_mean.to(device)
        avg_prod[prefix] = prod_mean.to(device)

    # 3) 优化 ΔB：min 1-cos(avg_prod, (hat_B+ΔB)hat_A) + λ||ΔB||^2
    delta_B = {}
    opt_params = []
    for prefix in hat_B.keys():
        #dB = torch.empty_like(hat_B[prefix], device=device)
        #torch.nn.init.xavier_uniform_(dB)
        dB = torch.zeros_like(hat_B[prefix], device=device)  # ← 全零
        dB.requires_grad_(True)
        #dB.requires_grad_(True)
        delta_B[prefix] = dB
        opt_params.append(dB)

    if opt_params:
        optimizer = torch.optim.SGD(opt_params, lr=lr)
        for _ in range(iters):
            optimizer.zero_grad()
            total_loss = 0.0
            for prefix in hat_B.keys():
                Bcorr = hat_B[prefix] + delta_B[prefix]                # (d,r)
                recon = _safe_product(Bcorr, hat_A[prefix])            # (d,l)
                # 余弦相似度：按全部元素展平
                x = avg_prod[prefix].reshape(-1)
                y = recon.reshape(-1)
                # 避免除零：加极小值
                cos = torch.dot(x, y) / (x.norm(p=2).clamp_min(eps) * y.norm(p=2).clamp_min(eps))
                loss1 = 1.0 - cos
                #loss1 =  cos
                loss2 = lambda_reg * (delta_B[prefix].pow(2).sum())
                total_loss = total_loss + loss1 + loss2
            total_loss.backward()
            optimizer.step()
    # 4) 组装新的 LoRA state，并加载到模型
    lora_only = {}
    for prefix, it in groups.items():
        A_key, B_key = it["A_key"], it["B_key"]
        A_proto, B_proto = it["A_proto"], it["B_proto"]
        if A_key and prefix in hat_A:
            lora_only[A_key] = hat_A[prefix].to(dtype=A_proto.dtype, device=A_proto.device)
        if B_key and prefix in hat_B:
            B_corr = (hat_B[prefix] + delta_B.get(prefix, 0.)).to(dtype=B_proto.dtype, device=B_proto.device)
            lora_only[B_key] = B_corr
    model.load_state_dict(lora_only,strict=False)
    return {k: v.cpu() for k, v in model.state_dict().items()}


def apply_weights_lora_fair_CV(
    num_workers,
    weights,                 # List[Dict[str, Tensor]] 各客户端(绝对)LoRA参数
    model,
    *,
    selection=1.0,           # 参与比例；这里做“均值”不需要额外缩放
    iters=200,               # 优化 ΔB 的迭代步数
    lr=1e-2,                 # 学习率
    lambda_reg=1.0,          # 正则系数 λ
    eps=1e-12
):
    """
    LoRA-FAIR 聚合：对各层先求 A/B 均值与乘积均值，然后仅优化 ΔB 来拟合客户端乘积的平均。
    返回 CPU 版的完整 state_dict（便于后续传输/保存）。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) 客户端内配对，再汇总
    groups = _pair_by_layer(weights)

    # 2) 逐层统计: \hat A, \hat B, \overline{P}
    hat_A, hat_B, avg_prod = {}, {}, {}
    for prefix, it in groups.items():
        pairs = it["pairs"]
        if not pairs:
            continue
        # 均值 A/B
        A_mean = torch.stack([A for A, _ in pairs], 0).mean(0)   # (r,l)
        B_mean = torch.stack([B for _, B in pairs], 0).mean(0)   # (d,r)
        # 均值 ΔW
        prod_mean = torch.stack([_safe_product(B, A) for A, B in pairs], 0).mean(0)  # (d,l)
        hat_A[prefix] = A_mean.to(device)
        hat_B[prefix] = B_mean.to(device)
        avg_prod[prefix] = prod_mean.to(device)

    # 3) 优化 ΔB：min 1-cos(avg_prod, (hat_B+ΔB)hat_A) + λ||ΔB||^2
    delta_B = {}
    opt_params = []
    for prefix in hat_B.keys():
        #dB = torch.empty_like(hat_B[prefix], device=device)
        #torch.nn.init.xavier_uniform_(dB)
        dB = torch.zeros_like(hat_B[prefix], device=device)  # ← 全零
        dB.requires_grad_(True)
        #dB.requires_grad_(True)
        delta_B[prefix] = dB
        opt_params.append(dB)

    if opt_params:
        optimizer = torch.optim.SGD(opt_params, lr=lr)
        for _ in range(iters):
            optimizer.zero_grad()
            total_loss = 0.0
            for prefix in hat_B.keys():
                Bcorr = hat_B[prefix] + delta_B[prefix]                # (d,r)
                recon = _safe_product(Bcorr, hat_A[prefix])            # (d,l)
                # 余弦相似度：按全部元素展平
                x = avg_prod[prefix].reshape(-1)
                y = recon.reshape(-1)
                # 避免除零：加极小值
                cos = torch.dot(x, y) / (x.norm(p=2).clamp_min(eps) * y.norm(p=2).clamp_min(eps))
                loss1 = 1.0 - cos
                #loss1 =  cos
                loss2 = lambda_reg * (delta_B[prefix].pow(2).sum())
                total_loss = total_loss + loss1 + loss2
            total_loss.backward()
            optimizer.step()
    # 4) 组装新的 LoRA state，并加载到模型
    lora_only = {}
    for prefix, it in groups.items():
        A_key, B_key = it["A_key"], it["B_key"]
        A_proto, B_proto = it["A_proto"], it["B_proto"]
        if A_key and prefix in hat_A:
            lora_only[A_key] = hat_A[prefix].to(dtype=A_proto.dtype, device=A_proto.device)
        if B_key and prefix in hat_B:
            B_corr = (hat_B[prefix] + delta_B.get(prefix, 0.)).to(dtype=B_proto.dtype, device=B_proto.device)
            lora_only[B_key] = B_corr

    #lora_only = {k: v for k, v in new_lora_state.items() if "lora" in k}
    scale = 1.0 / (num_workers * selection)
    # 聚合 delta_wi
    for weight in weights:
        for k, v in weight.items():
            if ('classifier' in k) or ('head' in k):
                if k not in lora_only.keys():
                    lora_only[k] = torch.zeros_like(v, device='cpu')
                lora_only[k].add_(v, alpha=scale)  # inplace 加法
    model.load_state_dict(lora_only,strict=False)
    #model.load_state_dict(lora_only,strict=False)
    return {k: v.cpu() for k, v in model.state_dict().items()}