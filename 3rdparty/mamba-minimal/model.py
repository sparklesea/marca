"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
from __future__ import annotations
import math
import json
from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum

from generation import GenerationMixin
from collections import namedtuple

import os
ROOT_DIR = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijinhao-240108540148/research_huangshan/marca/3rdparty/mamba-minimal/profile_result/'

@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    dtype: torch.dtype
    device: torch.device
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    debug: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)

class Mamba(nn.Module, GenerationMixin):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        
        self.args = args
        factory_kwargs = {"device": args.device, "dtype": args.dtype}
        super().__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.d_model, **factory_kwargs)
        self.layers = nn.ModuleList([ResidualBlock(args, layer_id) for layer_id in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model, **factory_kwargs)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False, **factory_kwargs)
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.
                                                     # See "Weight Tying" paper


    def forward(self, input_ids, task):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x, task)
            
        x = self.norm_f(x).to(self.norm_f.weight.dtype)
        logits = self.lm_head(x)

        return logits

    
    @staticmethod
    def from_pretrained(pretrained_model_name: str, debug=False, dtype=torch.float32, device=torch.device('cpu')):
        """Load pretrained weights from HuggingFace into model.
    
        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            model: Mamba model with weights loaded
    
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size'],
            debug=debug,
            dtype=dtype,
            device=device,
        )
        model = Mamba(args)
        
        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)
        
        return model


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args, layer_id)
        self.norm = RMSNorm(args.d_model, device=args.device, dtype=args.dtype)
        
    def forward(self, x, task):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        output = self.mixer(self.norm(x.to(self.norm.weight.dtype))) + x.to(torch.float32)

        return output
            

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args
        factory_kwargs = {"device": args.device, "dtype": args.dtype}
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
            **factory_kwargs,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False, **factory_kwargs)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True, **factory_kwargs)

        A = repeat(torch.arange(1, args.d_state + 1, dtype=torch.float32, device=args.device), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner, dtype=torch.float32, device=args.device))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias, **factory_kwargs)

        self.layer_id = layer_id

        self.deltaA = 0
        self.deltaB = 0

    def forward(self, x, task):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x, task)
        
        y = y * F.silu(res)

        y = y.to(dtype=self.args.dtype)
        
        output = self.out_proj(y)

        del x, y, res, x_and_res
        torch.cuda.empty_cache()

        return output

    
    def ssm(self, x, task):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)

        delta = F.softplus(self.dt_proj(delta).float())  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D, task)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        del delta, A, B, C, x_dbl
        torch.cuda.empty_cache()
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D, task):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
    
        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        B = B.float()
        C = C.float()
        u = u.float()

        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        # deltaB = einsum(delta, B, 'b l d_in, b l n -> b l d_in n')
        # deltaB_u = einsum(deltaB, u, 'b l d_in n, b l d_in -> b l d_in n')

        # if self.args.debug:
        #     with torch.no_grad():
        #         deltaB_compressed = einsum(delta.mean(dim=1).squeeze(1), B.mean(dim=1).squeeze(1), 'b d_in, b n -> b d_in n')
        #         # deltaB_compressed = deltaB.mean(dim=1).squeeze(0)
        #         deltaA_compressed = deltaA.mean(dim=1).squeeze(0) # (d_in n)
        #         # deltaB_compressed = deltaB # (d_in n)

        #         dA_path = ROOT_DIR + f'pt/deltaA/{task}/'
        #         dB_path = ROOT_DIR + f'pt/deltaB/{task}/'
        #         if not os.path.exists(dA_path):
        #             os.makedirs(dA_path)
        #         if not os.path.exists(dB_path):
        #             os.makedirs(dB_path)

        #         dA_path = dA_path + f'{self.layer_id}.pt'
        #         dB_path = dB_path + f'{self.layer_id}.pt'

        #         if os.path.exists(dA_path):
        #             temp_dA = torch.load(dA_path)
        #             dA_to_save = deltaA_compressed + temp_dA
        #             torch.save(dA_to_save, dA_path)
        #             del temp_dA, dA_to_save
        #         else:
        #             torch.save(deltaA_compressed, dA_path)
        #         if os.path.exists(dB_path):
        #             temp_dB = torch.load(dB_path)
        #             dB_to_save = deltaB_compressed + temp_dB
        #             torch.save(dB_to_save, dB_path)
        #             del temp_dB, dB_to_save
        #         else:
        #             torch.save(deltaB_compressed, dB_path)

        #         # release memory
        #         del deltaA_compressed, deltaB_compressed
        #         torch.cuda.empty_cache()

        if self.args.debug:
            with torch.no_grad():
                self.deltaB = self.deltaB + einsum(delta.mean(dim=1).squeeze(1), B.mean(dim=1).squeeze(1), 'b d_in, b n -> b d_in n').squeeze(0)
                self.deltaA = self.deltaA + deltaA.mean(dim=1).squeeze(0) # (d_in n)
        
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        del deltaA, deltaB_u
        torch.cuda.empty_cache()
    
        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 dtype = None,
                 device = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
        
