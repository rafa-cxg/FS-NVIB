import math
from functools import partial

import torch
import torch.nn as nn

from .nvib_layer import Nvib
from .utils import trunc_normal_


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
    
class DenoisingAttention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qkv_bias = qkv_bias


    def forward(self, x, nvib_tuple=None, set_mata_training_mode=False):
        
        # NVIB: In training the key is z and is sampled
        if nvib_tuple is not None:
            z, pi, mu, logvar, alpha, mask = nvib_tuple
            Nl = z.shape[1] # Length including prior

        B, N, C = x.shape
        # Breakdown the qkv layer
        w_q, w_k, w_v = self.qkv.weight.split(C, dim=0)
        if self.qkv_bias:
            b_q, b_k, b_v = self.qkv.bias.split(C, dim=0)

            q = x @ w_q.T + b_q
            if set_mata_training_mode:
                k = (z @ w_k.T + b_k) * self.scale
                v = z @ w_v.T + b_v
            else:
                biased_var = torch.exp(logvar) + math.sqrt(self.head_dim)
                k = (mu / biased_var) @ w_k.T + b_k
                # v = (mu) @ w_v.T + b_v #当mu是从identity获得的时候，和常规 k v值一样
                v = ((math.sqrt(self.head_dim) / biased_var) * mu) @ w_v.T + b_v
        else:
            q = x @ w_q.T
            if set_mata_training_mode:
                k = (z @ w_k.T) * self.scale
                v = z @ w_v.T
            else:
                biased_var = torch.exp(logvar) + math.sqrt(self.head_dim)
                k = (mu / biased_var) @ w_k.T
                # v = (mu) @ w_v.T
                v = ((math.sqrt(self.head_dim) / biased_var) * mu) @ w_v.T
    
        q = q.reshape(B, N, self.num_heads,self.head_dim).permute(0,2,1,3)
        k = k.reshape(B, Nl, self.num_heads,self.head_dim).permute(0,2,1,3) 
        v = v.reshape(B, Nl, self.num_heads,self.head_dim).permute(0,2,1,3)


        attn = (q @ k.transpose(-2, -1)) 

        pi = torch.clamp(pi.clone(), min=torch.finfo(pi.dtype).tiny)  # [B, S, 1]

        if set_mata_training_mode:

            # L2 norm term
            l2_norm = (1 / (2 * math.sqrt(C // self.num_heads))) * (#为什么要乘2?
                (torch.norm(z, dim=-1, keepdim=True)) ** 2
            )  # [B, S, 1]
            # Include bias terms, copied over heads, broadcasted over T
            attn += (torch.log(pi) - l2_norm).unsqueeze(1).permute(0, 1, 3, 2)

        else:
            biased_var = torch.exp(logvar) + math.sqrt(C // self.num_heads)

            # L2 norm term
            l2_norm = 0.5 * (
                (torch.norm((mu / torch.sqrt(biased_var)), dim=-1, keepdim=True)) ** 2
            )  # [B, S, 1]

            # Variance penalty term NOTE: in V2 we do not have the var penalty
            # var_penalty = torch.sum(
            #     0.5
            #     * (torch.log(biased_var)),
            #     dim=-1,
            #     keepdim=True,
            # )  # [B, S, 1]
            # Include bias terms, copied over heads, broadcasted over T
            attn += (
                (torch.log(alpha) - l2_norm
                 # - var_penalty
                 ).unsqueeze(1).permute(0, 1, 3, 2)
            )


        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if set_mata_training_mode:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        else:
            # NOTE: in V2 we do not have the interpolation
            ###### NVIB INTERPOLATION ######
            # # Interpolate the pre-projected value "mu" and post-projected query "projected_u"
            # # Create projected_u
            # w_k = w_k.view(self.num_heads, self.head_dim, -1)  # [nh, hs, C]
            # w_v = w_v.view(self.num_heads, self.head_dim, -1)  # [nh, hs, C]
            # if self.qkv_bias is not None:
            #     b_v = b_v.view(self.num_heads, self.head_dim, 1)  # [nh, hs, C]
            # else:
            #     b_v = torch.zeros(self.num_heads, self.head_dim, 1, dtype=q.dtype)
            # projected_u = torch.einsum("bhme, hep -> bhmp", q, w_k)
            # # out = torch.einsum("bhmp, bnp -> bhmn", projected_u, (mu/biased_var)) # Same as before
            # # Calculate the interpolation
            # output = (
            #     torch.einsum("bhmn, bnp -> bhmp", attn, (torch.exp(logvar) / biased_var))
            #     * projected_u
            # ) + torch.einsum("bhmn, bnp -> bhmp", attn, ((math.sqrt(self.head_dim) / biased_var) * mu))
            # # Project into the correct space and add the bias. The bias is theoretically
            # # multiplied by the attention_probs, but it noramlises so we can just add it.
            # x = torch.einsum("bhmp, hep -> bhme", output, w_v) + b_v.unsqueeze(0).permute(
            #     0, 1, 3, 2
            # )
            # x = x.transpose(1, 2).reshape(B, N, C)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        **kwargs
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # NVIB layers is a list of true falses
        if kwargs["nvib_layers"].pop(0):
            # NVIB
            rng_state = torch.get_rng_state()
            # Nvib layer
            self.nvib_layer = Nvib(
                size_in=dim,
                size_out=dim,
                prior_mu= None,
                prior_var= None,
                prior_log_alpha=(None),
                prior_log_alpha_stdev=(None),
                delta=kwargs.get("delta", 1),
                nheads=num_heads,
                alpha_tau=kwargs.get("alpha_tau", 10),
                mu_tau=1,
                stdev_tau=kwargs.get("stdev_tau", 0),
            )
            torch.set_rng_state(rng_state)

            self.attn = DenoisingAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

    def forward(self, x, return_attention=False, set_mata_training_mode=False):
        if hasattr(self, "nvib_layer"):

            x_norm = self.norm1(x) # Very similar
            nvib_tuple = self.nvib_layer(x_norm, set_mata_training_mode=set_mata_training_mode)

            # KL divergence loss
            # Calculate KL divergences
            kl_gaussian = self.nvib_layer.kl_gaussian(
                mu=nvib_tuple[2],
                logvar=nvib_tuple[3],
                alpha=nvib_tuple[4],
                mask=nvib_tuple[5],
                set_mata_training_mode=set_mata_training_mode
            )
            kl_dirichlet = self.nvib_layer.kl_dirichlet(
                alpha=nvib_tuple[4],
                mask=nvib_tuple[5],
                set_mata_training_mode=set_mata_training_mode
            )
            y, attn = self.attn(x_norm, nvib_tuple=nvib_tuple,set_mata_training_mode=set_mata_training_mode)
            if return_attention:
                return attn
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, (kl_gaussian, kl_dirichlet)
        else:
            x_norm = self.norm1(x)
            y, attn = self.attn(x_norm)
            if return_attention:
                return attn
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class NvibVisionTransformer(nn.Module):
    """Vision Transformer with NVIB"""

    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        **kwargs
    ):
        super().__init__()

        # Make a boolean list of which layers are NVIB layers
        kwargs["nvib_layers"] = [
            i in kwargs["nvib_layers"] for i in range(depth)
        ]
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    **kwargs
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        # self.apply(self._init_weights)
        # Reinitialise ALL the NVIB parameters
        for i in range(0, len(self.blocks)):
            # if layer has attribute nvib_layer, then init_parameters
            block = self.blocks[i]
            if hasattr(block, "nvib_layer"):
                print("Layer " + str(i) + " has nvib_layer")
                block.nvib_layer.init_parameters()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
            align_corners=False,
            recompute_scale_factor=False,
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x, ada_token=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        if ada_token is not None:
            ada_tokens = ada_token.expand(B, -1, -1)  # B, p, d
            x = torch.cat((x, ada_tokens), dim=1)

        return self.pos_drop(x)

    def forward(self, x, ada_token=None, use_patches=False, set_mata_training_mode=False):
        x = self.prepare_tokens(x, ada_token)
        all_kl_gaussian = []
        all_kl_dirichlet = []
        for blk in self.blocks:
            if hasattr(blk, "nvib_layer"):
                x, kls = blk(x, set_mata_training_mode=set_mata_training_mode)
                all_kl_gaussian.append(kls[0])
                all_kl_dirichlet.append(kls[1])
            else:
                x = blk(x)
        x = self.norm(x)
        if use_patches:
            return x[:, 1:]
        else:
            # Using the CLS I assume
            return x[:, 0], (all_kl_gaussian, all_kl_dirichlet)

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny(patch_size=16, **kwargs):
    model = NvibVisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_small(patch_size=16, **kwargs):
    model = NvibVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_base(patch_size=16, **kwargs):
    model = NvibVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
