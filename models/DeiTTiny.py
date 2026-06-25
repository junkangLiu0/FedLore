import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, _cfg
from functools import partial

class ViTForCIFAR(VisionTransformer):
    def __init__(self, img_size=32, patch_size=4, num_classes=100, depth=6, **kwargs):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=192,
            #depth=12,
            depth=depth,
            num_heads=3,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


class DeiTTinyForCIFAR_256(VisionTransformer):
    def __init__(self, img_size=32, patch_size=4, num_classes=100, **kwargs):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=384,
            #depth=12,
            depth=6,
            num_heads=3,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


class Vit(VisionTransformer):
    def __init__(self, img_size=32, patch_size=4, num_classes=100, **kwargs):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=384,
            #depth=12,
            depth=6,
            num_heads=3,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


def ViTForCIFAR(pretrained=False,num_classes=100, img_size=32, depth=6, **kwargs):
    model = ViTForCIFAR(num_classes=num_classes, img_size=img_size, depth=depth,**kwargs)
    return model

def ViTForCIFAR(pretrained=False,num_classes=100, img_size=32, depth=6, **kwargs):
    model = ViTForCIFAR(num_classes=num_classes, img_size=img_size, depth=depth,**kwargs)
    return model

def deit_tiny_256(pretrained=False,num_classes=100, img_size=32, **kwargs):
    model = DeiTTinyForCIFAR_256(num_classes=num_classes, img_size=img_size,**kwargs)
    return model

def deit_tiny(pretrained=False,num_classes=100, img_size=32, depth=6, **kwargs):
    model = ViTForCIFAR(num_classes=num_classes, img_size=img_size, depth=depth,**kwargs)
    return model
