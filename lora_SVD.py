import torch
from collections import defaultdict

@torch.no_grad()
def aggregate_AB_then_SVD(
    weights,                 # List[Dict[str, Tensor]]                      # 目标秩
    num_workers,        # 客户端数量（用于缩放）
    r=16,
    selection: float = 1.0,  # 采样率（例如客户端子采样时用）
    eps: float = 1e-12       # 数值稳定项
):
    """
    逐层：
      1) 对每客户端计算 ΔW_k = B_k @ A_k
      2) 求平均得到 ΔW_global = (1 / (num_workers * selection)) * sum_k ΔW_k
      3) 对 ΔW_global 做截断 SVD: U Σ V^T
      4) 用“平衡分解”：B_new = U_r @ sqrt(Σ_r), A_new = sqrt(Σ_r) @ V_r^T
         使得 B_new @ A_new ≈ ΔW_global，且 A/B 的尺度较均衡。
    返回仅含 lora_A/lora_B，可用于 model.load_state_dict(..., strict=False)
    """
    # 1) 按层归组
    groups = defaultdict(lambda: {"A": [], "B": [], "A_key": None, "B_key": None, "A_proto": None, "B_proto": None})
    for w in weights:
        for k, v in w.items():
            if not isinstance(v, torch.Tensor):
                continue
            if "lora_A" in k:
                #print(v.shape)
                prefix = k.split("lora_A")[0].rstrip(".")
                #print(prefix )
                groups[prefix]["A"].append(v.detach().to(torch.float32))
                groups[prefix]["A_key"] = k
                groups[prefix]["A_proto"] = v
            elif "lora_B" in k:
                prefix = k.split("lora_B")[0].rstrip(".")
                groups[prefix]["B"].append(v.detach().to(torch.float32))
                groups[prefix]["B_key"] = k
                groups[prefix]["B_proto"] = v

    new_lora_state = {}

    # 2) 逐层聚合 -> SVD -> 回写
    scale = num_workers * max(selection, eps)

    for layer_prefix, it in groups.items():
        As, Bs = it["A"], it["B"]
        if len(As) == 0 or len(Bs) == 0:
            continue

        # 对齐 A/B 数量
        m = min(len(As), len(Bs))
        #print(m)
        As, Bs = As[:m], Bs[:m]

        # (a) 累加 ΔW_k
        deltaW_sum = None
        for A, B in zip(As, Bs):
            # 约定：B ∈ R^{d×r}，A ∈ R^{r×d}，则 ΔW_k = B @ A ∈ R^{d×l}
            #print(B.shape,A.shape)
            if B.ndim != 2 or A.ndim != 2:
                raise ValueError(f"{layer_prefix}: A/B 必须是二维矩阵，得到 A{A.shape}, B{B.shape}")
            if B.shape[1] == A.shape[0]:
                deltaW_k = B @ A
            elif B.shape[0] == A.shape[1]:
                # 兼容 B,A 方向颠倒的实现（例如把 A 做成 (l×r)）
                deltaW_k = (B @ A.t())
            elif B.shape[1] == A.shape[1]:
                # 另一种常见误差：A 也是 (r×l) 但被转置了；用 B @ A^T 试配
                deltaW_k = B @ A.t()
            else:
                raise ValueError(f"{layer_prefix}: 无法匹配乘法维度，A{A.shape}, B{B.shape}")

            if deltaW_sum is None:
                deltaW_sum = deltaW_k
            else:
                # 若各客户端在不同 device，这里都已转到 float32（CPU/GPU 不强制），可以直接相加
                deltaW_sum = deltaW_sum + deltaW_k

        if deltaW_sum is None:
            continue

        # (b) 求平均（带 selection 缩放）
        deltaW = deltaW_sum / scale

        # (c) 截断 SVD
        d, l = deltaW.shape
        r_eff = int(min(r, d, l))
        if r_eff == 0:
            continue
        r_eff=r
        # torch.linalg.svd 返回 U(d,k), S(k,), Vh(k,l), 其中 k = min(d,l)
        U, S, Vh = torch.linalg.svd(deltaW.to('cuda'), full_matrices=False)
        #U, S, Vh = torch.linalg.svd(deltaW, full_matrices=False)
        U_r  = U[:, :r_eff]                  # (d, r)
        S_r  = S[:r_eff]      # (r,)
        Vh_r = Vh[:r_eff, :]                 # (r, d)
        sqrt_S = torch.sqrt(S_r)
        B_new = U_r * S_r.unsqueeze(0)   # (d, r)
        A_new =Vh_r   # (r, d)
        #Q, R = torch.linalg.qr(deltaW, mode='reduced')  # Q:(d,k), R:(k,l)
        #B_new = Q[:, :r_eff]  # (d, r)
        #A_new = R[:r_eff, :]  # (r, d)

        # (e) 回到原 dtype/device 并写 key
        A_proto, B_proto = it["A_proto"], it["B_proto"]
        if B_proto is not None and it["B_key"] is not None:
            new_lora_state[it["B_key"]] = B_new.to(dtype=B_proto.dtype, device=B_proto.device)
        if A_proto is not None and it["A_key"] is not None:
            new_lora_state[it["A_key"]] = A_new.to(dtype=A_proto.dtype, device=A_proto.device)

    return new_lora_state



def aggregate_FRLORA(
    weights,                 # List[Dict[str, Tensor]]                      # 目标秩
    num_workers,        # 客户端数量（用于缩放）
    r=16,
    selection: float = 1.0,  # 采样率（例如客户端子采样时用）
    eps: float = 1e-12       # 数值稳定项
):
    """
    逐层：
      1) 对每客户端计算 ΔW_k = B_k @ A_k
      2) 求平均得到 ΔW_global = (1 / (num_workers * selection)) * sum_k ΔW_k
      3) 对 ΔW_global 做截断 SVD: U Σ V^T
      4) 用“平衡分解”：B_new = U_r @ sqrt(Σ_r), A_new = sqrt(Σ_r) @ V_r^T
         使得 B_new @ A_new ≈ ΔW_global，且 A/B 的尺度较均衡。
    返回仅含 lora_A/lora_B，可用于 model.load_state_dict(..., strict=False)
    """
    # 1) 按层归组
    groups = defaultdict(lambda: {"A": [], "B": [], "A_key": None, "B_key": None, "A_proto": None, "B_proto": None})
    for w in weights:
        for k, v in w.items():
            if not isinstance(v, torch.Tensor):
                continue
            if "lora_A" in k:
                #print(v.shape)
                prefix = k.split("lora_A")[0].rstrip(".")
                #print(prefix )
                groups[prefix]["A"].append(v.detach().to(torch.float32))
                groups[prefix]["A_key"] = k
                groups[prefix]["A_proto"] = v
            elif "lora_B" in k:
                prefix = k.split("lora_B")[0].rstrip(".")
                groups[prefix]["B"].append(v.detach().to(torch.float32))
                groups[prefix]["B_key"] = k
                groups[prefix]["B_proto"] = v

    new_lora_state = {}

    # 2) 逐层聚合 -> SVD -> 回写
    scale = num_workers * max(selection, eps)

    for layer_prefix, it in groups.items():
        As, Bs = it["A"], it["B"]
        if len(As) == 0 or len(Bs) == 0:
            continue

        # 对齐 A/B 数量
        m = min(len(As), len(Bs))
        #print(m)
        As, Bs = As[:m], Bs[:m]

        # (a) 累加 ΔW_k
        deltaW_sum = None
        for A, B in zip(As, Bs):
            # 约定：B ∈ R^{d×r}，A ∈ R^{r×d}，则 ΔW_k = B @ A ∈ R^{d×l}
            #print(B.shape,A.shape)
            if B.ndim != 2 or A.ndim != 2:
                raise ValueError(f"{layer_prefix}: A/B 必须是二维矩阵，得到 A{A.shape}, B{B.shape}")
            if B.shape[1] == A.shape[0]:
                deltaW_k = B @ A
            elif B.shape[0] == A.shape[1]:
                # 兼容 B,A 方向颠倒的实现（例如把 A 做成 (l×r)）
                deltaW_k = (B @ A.t())
            elif B.shape[1] == A.shape[1]:
                # 另一种常见误差：A 也是 (r×l) 但被转置了；用 B @ A^T 试配
                deltaW_k = B @ A.t()
            else:
                raise ValueError(f"{layer_prefix}: 无法匹配乘法维度，A{A.shape}, B{B.shape}")

            if deltaW_sum is None:
                deltaW_sum = deltaW_k
            else:
                # 若各客户端在不同 device，这里都已转到 float32（CPU/GPU 不强制），可以直接相加
                deltaW_sum = deltaW_sum + deltaW_k

        if deltaW_sum is None:
            continue

        # (b) 求平均（带 selection 缩放）
        deltaW = deltaW_sum / scale

        # (c) 截断 SVD
        d, l = deltaW.shape
        r_eff = int(min(r, d, l))
        if r_eff == 0:
            continue
        r_eff=r
        # torch.linalg.svd 返回 U(d,k), S(k,), Vh(k,l), 其中 k = min(d,l)
        U, S, Vh = torch.linalg.svd(deltaW.to('cuda'), full_matrices=False)

        #U, S, Vh = torch.linalg.svd(deltaW, full_matrices=False)
        U_r  = U[:, :r_eff]                  # (d, r)
        S_r  = S[:r_eff]      # (r,)
        Vh_r = Vh[:r_eff, :]                 # (r, d)
        sqrt_S = torch.sqrt(S_r)
        B_new = U_r * sqrt_S.unsqueeze(0)   # (d, r)
        A_new =sqrt_S.unsqueeze(1)*Vh_r   # (r, d)

        # (e) 回到原 dtype/device 并写 key
        A_proto, B_proto = it["A_proto"], it["B_proto"]
        if B_proto is not None and it["B_key"] is not None:
            new_lora_state[it["B_key"]] = B_new.to(dtype=B_proto.dtype, device=B_proto.device)
        if A_proto is not None and it["A_key"] is not None:
            new_lora_state[it["A_key"]] = A_new.to(dtype=A_proto.dtype, device=A_proto.device)

    return new_lora_state


def aggregate_AB_then_QR(
    weights,                 # List[Dict[str, Tensor]]                      # 目标秩
    num_workers,        # 客户端数量（用于缩放）
    r=16,
    selection: float = 1.0,  # 采样率（例如客户端子采样时用）
    eps: float = 1e-12       # 数值稳定项
):
    """
    逐层：
      1) 对每客户端计算 ΔW_k = B_k @ A_k
      2) 求平均得到 ΔW_global = (1 / (num_workers * selection)) * sum_k ΔW_k
      3) 对 ΔW_global 做截断 SVD: U Σ V^T
      4) 用“平衡分解”：B_new = U_r @ sqrt(Σ_r), A_new = sqrt(Σ_r) @ V_r^T
         使得 B_new @ A_new ≈ ΔW_global，且 A/B 的尺度较均衡。
    返回仅含 lora_A/lora_B，可用于 model.load_state_dict(..., strict=False)
    """
    # 1) 按层归组
    groups = defaultdict(lambda: {"A": [], "B": [], "A_key": None, "B_key": None, "A_proto": None, "B_proto": None})
    for w in weights:
        for k, v in w.items():
            if not isinstance(v, torch.Tensor):
                continue
            if "lora_A" in k:
                #print(v.shape)
                prefix = k.split("lora_A")[0].rstrip(".")
                #print(prefix )
                groups[prefix]["A"].append(v.detach().to(torch.float32))
                groups[prefix]["A_key"] = k
                groups[prefix]["A_proto"] = v
            elif "lora_B" in k:
                prefix = k.split("lora_B")[0].rstrip(".")
                groups[prefix]["B"].append(v.detach().to(torch.float32))
                groups[prefix]["B_key"] = k
                groups[prefix]["B_proto"] = v

    new_lora_state = {}

    # 2) 逐层聚合 -> SVD -> 回写
    scale = num_workers * max(selection, eps)

    for layer_prefix, it in groups.items():
        As, Bs = it["A"], it["B"]
        if len(As) == 0 or len(Bs) == 0:
            continue

        # 对齐 A/B 数量
        m = min(len(As), len(Bs))
        #print(m)
        As, Bs = As[:m], Bs[:m]

        # (a) 累加 ΔW_k
        deltaW_sum = None
        for A, B in zip(As, Bs):
            # 约定：B ∈ R^{d×r}，A ∈ R^{r×d}，则 ΔW_k = B @ A ∈ R^{d×l}
            #print(B.shape,A.shape)
            if B.ndim != 2 or A.ndim != 2:
                raise ValueError(f"{layer_prefix}: A/B 必须是二维矩阵，得到 A{A.shape}, B{B.shape}")
            if B.shape[1] == A.shape[0]:
                deltaW_k = B @ A
            elif B.shape[0] == A.shape[1]:
                # 兼容 B,A 方向颠倒的实现（例如把 A 做成 (l×r)）
                deltaW_k = (B @ A.t())
            elif B.shape[1] == A.shape[1]:
                # 另一种常见误差：A 也是 (r×l) 但被转置了；用 B @ A^T 试配
                deltaW_k = B @ A.t()
            else:
                raise ValueError(f"{layer_prefix}: 无法匹配乘法维度，A{A.shape}, B{B.shape}")

            if deltaW_sum is None:
                deltaW_sum = deltaW_k
            else:
                # 若各客户端在不同 device，这里都已转到 float32（CPU/GPU 不强制），可以直接相加
                deltaW_sum = deltaW_sum + deltaW_k

        if deltaW_sum is None:
            continue

        # (b) 求平均（带 selection 缩放）
        deltaW = deltaW_sum / scale

        # (c) 截断 SVD
        d, l = deltaW.shape
        r_eff = int(min(r, d, l))
        if r_eff == 0:
            continue
        r_eff=16
        # torch.linalg.svd 返回 U(d,k), S(k,), Vh(k,l), 其中 k = min(d,l)
        #U, S, Vh = torch.linalg.svd(deltaW, full_matrices=False)
        #U_r  = U[:, :r_eff]                  # (d, r)
        #S_r  = S[:r_eff]      # (r,)
        #Vh_r = Vh[:r_eff, :]                 # (r, d)
        #sqrt_S = torch.sqrt(S_r)
        #B_new = U_r * S_r.unsqueeze(0)   # (d, r)
        #A_new =Vh_r   # (r, d)
        Q, R = torch.linalg.qr(deltaW, mode='reduced')  # Q:(d,k), R:(k,l)
        B_new = Q[:, :r_eff]  # (d, r)
        A_new = R[:r_eff, :]  # (r, d)

        # (e) 回到原 dtype/device 并写 key
        A_proto, B_proto = it["A_proto"], it["B_proto"]
        if B_proto is not None and it["B_key"] is not None:
            new_lora_state[it["B_key"]] = B_new.to(dtype=B_proto.dtype, device=B_proto.device)
        if A_proto is not None and it["A_key"] is not None:
            new_lora_state[it["A_key"]] = A_new.to(dtype=A_proto.dtype, device=A_proto.device)

    return new_lora_state

