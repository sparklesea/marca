# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        marca_args=None,
        # debug=False,
        # sparsedB=False,
        # sparsehs=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        self.debug = marca_args.debug
        
        # sparse
        self.sparseB = marca_args.sparsedB
        self.sparseA = False
        self.sparse_hs = marca_args.sparsehs
        self.constant = False
        
        #approx
        self.fastexp = marca_args.fastexp
        self.silu = marca_args.silu
        
        if self.debug:
            self.sparseB, self.sparseA, self.sparse_hs, self.constant = False, False, False, False
        self.deltaB = None
        self.deltaA = None
        self.hidden_states = None
        self.count = []
        self.chunk_size = 8
        
        # self.profile_layers = [0, 4, 16, 64, 256, 1024]
        # self.num_profile_layers = len(self.profile_layers)
        # self.count_hidden_states = [0] * self.num_profile_layers
        # if self.debug:
        #     self.hidden_states = torch.zeros(self.d_inner, self.d_state, self.num_profile_layers, device=device) #(d_inner, d_state)
        
        self.count_hs = []
        
        if self.sparse_hs:
            self.mask_hs = torch.load('/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijinhao-240108540148/research_huangshan/marca/profile_result/pt/hidden_states_mask/all_L.pt', map_location=device) 
        if self.sparseB:
            # self.mask = torch.load('/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijinhao-240108540148/research_huangshan/marca/profile_result/pt/deltaB_mask/all.pt', map_location=device)
            self.mask: Tensor = torch.load('/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijinhao-240108540148/research_huangshan/marca/profile_result/pt/deltaB_mask/all_L.pt', map_location=device)
        if self.sparseA:    
            self.maskA: Tensor = torch.load('/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijinhao-240108540148/research_huangshan/marca/profile_result/pt/deltaA_mask/all_L.pt', map_location=device)
        elif self.constant:
            self.constant_dB = torch.load('/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijinhao-240108540148/research_huangshan/marca/profile_result/pt/deltaB_mask/constant.pt', map_location=device)
    
    @staticmethod
    def average_seqlen_chunk(input, chunk_size=256):
        (d, seq_len, n) = input.shape
        # 计算完整的 256 块数和剩余的元素数
        full_blocks = seq_len // chunk_size
        remainder = seq_len % chunk_size
        
        # 初始化结果张量
        result_list = []
        
        # 对完整的 256 块进行平均
        if full_blocks > 0:
            full_blocks_tensor = input[:, :full_blocks * chunk_size, :]
            reshaped_full_blocks = full_blocks_tensor.view(d, full_blocks, chunk_size, n)
            averaged_full_blocks = reshaped_full_blocks.mean(dim=2)  # 在 256 维度上平均
            result_list.append(averaged_full_blocks)
        
        # 对剩余的部分进行平均（如果有）
        if remainder > 0:
            remainder_tensor = input[:, full_blocks * chunk_size:, :]
            averaged_remainder = remainder_tensor.mean(dim=1, keepdim=True)  # 在 l 维度上平均，保持维度
            # 调整形状以匹配 (d, 1, n) 以便后续拼接
            result_list.append(averaged_remainder)
        
        # 拼接结果
        if len(result_list) > 1:
            # 如果既有完整块又有剩余部分，拼接它们
            result_tensor = torch.cat(result_list, dim=1)
        else:
            # 如果只有完整块或只有剩余部分，直接使用该结果
            result_tensor = result_list[0]
        
        return result_tensor, full_blocks + (1 if remainder > 0 else 0)
    
    @staticmethod
    def pad_and_add(tensor1: Tensor, tensor2: Tensor):
        # 获取两个 tensor 的维度
        d1, l1, n1 = tensor1.shape
        d2, l2, n2 = tensor2.shape

        # 确保其他维度相同
        assert d1 == d2 and n1 == n2, "其他维度必须相同"

        # 确定最大的 l 维度
        max_l = max(l1, l2)
        # 补齐 tensor1 的 l 维度（如果需要）
        if l1 < max_l:
            # 在 l 维度上补齐 (左边补 0，右边补 max_l - l1)
            tensor1 = F.pad(tensor1, (0, 0, 0, max_l - l1))
        # 补齐 tensor2 的 l 维度（如果需要）
        if l2 < max_l:
            # 在 l 维度上补齐 (左边补 0，右边补 max_l - l2)
            tensor2 = F.pad(tensor2, (0, 0, 0, max_l - l2))
        # 相加
        result = tensor1 + tensor2
        return result
    
    def process_profile_data(self, deltaA: Tensor, deltaB: Tensor):
        batch = deltaA.shape[0]
        deltaA = deltaA.sum(dim=0) # (d, l, n)
        
        deltaA, covered_nums = self.average_seqlen_chunk(deltaA, chunk_size=self.chunk_size)
        
        if covered_nums > len(self.count):
            self.count.extend([0] * (covered_nums - len(self.count)))
        self.count[:covered_nums] = [x + batch for x in self.count[:covered_nums]]
        
        if self.deltaA is None:
            self.deltaA = deltaA
        elif isinstance(self.deltaA, torch.Tensor):
            self.deltaA = self.pad_and_add(self.deltaA, deltaA)
        else:
            raise NotImplementedError("self.deltaA is not float or torch.Tensor")
        
        deltaB = deltaB.sum(dim=0)
        deltaB, covered_nums = self.average_seqlen_chunk(deltaB, chunk_size=self.chunk_size)
        if self.deltaB is None:
            self.deltaB = deltaB
        elif isinstance(self.deltaB, torch.Tensor):
            self.deltaB = self.pad_and_add(self.deltaB, deltaB)
        else:
            raise NotImplementedError("self.deltaB is not float or torch.Tensor")

    
    def selective_scan_ref(self, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                        return_last_state=False):
        """
        u: r(B D L)
        delta: r(B D L)
        A: c(D N) or r(D N)
        B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
        C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
        D: r(D)
        z: r(B D L)
        delta_bias: r(D), fp32

        out: r(B D L)
        last_state (optional): r(B D dstate) or c(B D dstate)
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        
        if self.fastexp:
            import ctypes
            import numpy as np
            import struct
            from fast_exp_cuda import fast_exp

            # def fast_exp_np(y: np.ndarray) -> np.ndarray:
            #     # 确保输入为 float32 类型
            #     y = y.astype(np.float32, copy=False)
            #     # 计算缩放系数（向量化）
            #     scale = y * 1.4426950409 + 126.94201519
            #     # 通过视图转换直接操作二进制位（无需逐元素循环）
            #     scale_uint32 = scale.view(np.uint32)
            #     # 左移 23 位并保留 32 位范围（等价于原 CUDA 的位操作）
            #     shifted_uint32 = (scale_uint32 << 23) & 0xFFFFFFFF
            #     # 转换回 float32 并添加修正项
            #     result = shifted_uint32.view(np.float32) + 0.0285784
            #     return result.astype(y.dtype, copy=False)  # 保持原始数据类型
            # deltaA = torch.from_numpy(fast_exp_np(torch.einsum('bdl,dn->bdln', delta, A).cpu().numpy())).to(u.device)
            
            deltaA = fast_exp(torch.einsum('bdl,dn->bdln', delta, A / torch.log(torch.tensor(2.0, device=A.device))).contiguous().view(-1)).reshape(batch, dim, u.shape[-1], dstate)
        else:
            deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
            
        else:
            if B.dim() == 3:
                if not self.sparseA and not self.sparseB and not self.constant:
                    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
                
                if self.debug:
                    deltaB = torch.einsum('bdl,bnl->bdln', delta, B)
                    self.process_profile_data(deltaA, deltaB)
                    
                    # deltaB = torch.einsum('bdl,bnl->bdln', delta, B).sum(dim=0) / 61197
                    # self.deltaB = self.deltaB + torch.einsum('bdl,bnl->bdln', delta, B).mean(dim=2).sum(dim=0) / 61197
                    # self.deltaA = self.deltaA + deltaA.mean(dim=2).sum(dim=0) / 61197 # (d_in n) 
                
                if self.sparseB:
                    deltaB = torch.einsum('bdl,bnl->bdln', delta, B)
                    mask = self.mask[self.layer_idx]
                    
                    # maskB with chunks of seq length
                    if mask.dim() == 2: # (l/chunk_size, n)
                        mask = mask.repeat_interleave(repeats=self.chunk_size, dim=0)[:delta.shape[-1], :] #(l, n)
                        if mask.dtype == torch.bool: # bool mask
                            deltaB_mask = deltaB * (mask == False)
                        else: # constant mask without dim
                            condition = (mask != torch.inf)
                            deltaB_mask = torch.where(condition, mask, deltaB)
                    elif mask.dim() == 3: # constant mask with dim
                        mask = mask.unsqueeze(0) #(b,d,l/chunk_size,n)
                        condition = (mask != 0)
                        deltaB_mask = torch.where(condition, mask, deltaB)
                    else:
                        raise ValueError("mask dim must be 2 or 3!")
                    
                    deltaB_u = torch.einsum('bdln,bdl->bdln', deltaB_mask, u)
                
                if self.sparseA:
                    mask = self.maskA[self.layer_idx]
                    if mask.dim() == 2:
                        mask = mask.repeat_interleave(repeats=self.chunk_size, dim=0)[:delta.shape[-1], :]
                        if mask.dtype == torch.bool:
                            deltaA = torch.where(mask, 0.98, deltaA)
                        else:
                            condition = (mask != torch.inf)
                            deltaA = torch.where(condition, mask, deltaA)
                    # still need to compute deltaB_u
                    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
                elif self.constant:
                    deltaB_constant = self.constant_dB[self.layer_idx].unsqueeze(0).unsqueeze(2).repeat(delta.shape[0], 1, delta.shape[-1], 1)
                    
                    # deltaB = torch.einsum('bdl,bnl->bdln', delta, B)
                    # constant_dB = self.constant_dB[self.layer_idx].unsqueeze(0).unsqueeze(2)
                    # constant_dB_dropout = nn.functional.dropout(constant_dB, p = 0.9999999)
                    # condition = constant_dB_dropout != 0
                    # deltaB_constant = torch.where(condition, constant_dB_dropout, deltaB)
                    deltaB_u = torch.einsum('bdln,bdl->bdln', deltaB_constant, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        
        temp_hidden_states = None
        for i in range(u.shape[2]): # for each L
            if self.sparse_hs:
                temp_hs_mask = self.mask_hs[self.layer_idx, i // self.chunk_size]
                x = torch.where(temp_hs_mask, 0, x)
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i] #(b, d, n)
            
            # if self.sparse_hiddens_state:
                ## dynamic sparse
                # x_mean = x.mean(0).mean(0)
                # x_std = x.mean(0).std(dim=0)
                # mask = (x_mean.abs() < 0.001) & (x_std < 0.001)
                # print(f"layer: {self.layer_idx}, sparsity: {mask.sum() / mask.numel()}")
                # x = torch.where(mask, 0, x)
            
            # for hidden states profile
            if self.debug:
                if temp_hidden_states is None:
                    temp_hidden_states = x.sum(0).unsqueeze(1)
                else:
                    temp_hidden_states = torch.cat([temp_hidden_states, x.sum(0).unsqueeze(1)], dim=1) #(d, l, n)
            
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)

        if self.debug:
            temp_hidden_states, chunk_num = self.average_seqlen_chunk(temp_hidden_states, chunk_size=self.chunk_size)
            
            if chunk_num > len(self.count_hs):
                self.count_hs.extend([0] * (chunk_num - len(self.count_hs)))
            self.count_hs[:chunk_num] = [num + batch for num in self.count_hs[:chunk_num]]
            
            if self.hidden_states is None:
                self.hidden_states = temp_hidden_states
            elif isinstance(self.hidden_states, torch.Tensor):
                self.hidden_states = self.pad_and_add(self.hidden_states, temp_hidden_states)
            else:
                raise NotImplementedError("self.hidden_states is torch.Tensor")
        
        y = torch.stack(ys, dim=2) # (batch dim L)
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            if self.silu:
                def fast_silu(tensor):
                    result = torch.empty_like(tensor,device=tensor.device)
                    # x < -5
                    mask1 = tensor < -5
                    result[mask1] = -0.0135
                    # -5 <= x < -1.5
                    mask2 = (tensor >= -5) & (tensor < -1.5)
                    result[mask2] = -0.06244 * tensor[mask2] - 0.3457
                    # -1.5 <= x <= 0.75
                    mask3 = (tensor >= -1.5) & (tensor <= 0.75)
                    result[mask3] = 0.232 * (tensor[mask3] + 1.181) ** 2 - 0.275
                    # x > 0.75
                    mask4 = tensor > 0.75
                    result[mask4] = 1.05 * tensor[mask4] - 0.2781
                    return result
                out = out * fast_silu(z)
            else:
                out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        return out if not return_last_state else (out, last_state)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if False:
        # if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
            # if causal_conv1d_fn is not None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            # dtB (b d dstate l) * x (b d l)
            # 
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            # y = selective_scan_fn(
            #     x,
            #     dt,
            #     A,
            #     B,
            #     C,
            #     self.D.float(),
            #     z=z,
            #     delta_bias=self.dt_proj.bias.float(),
            #     delta_softplus=True,
            #     return_last_state=ssm_state is not None,
            # )
            y = self.selective_scan_ref(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out
    
    def compute_proportion(self, input, low_condition, high_condition, mode='outside'):
        # # 计算总元素数量
        total_elements = input.numel()
        if mode == 'outside':
            # 条件1：小于0.001
            condition1 = input < low_condition
            count1 = torch.sum(condition1).item()
            percentage1 = (count1 / total_elements) * 100

            # 条件2：位于0.95到1之间（包括0.95但不包括1，即 [0.95, 1)）
            condition2 = (input >= high_condition) & (input <= 1)
            count2 = torch.sum(condition2).item()
            percentage2 = (count2 / total_elements) * 100
            with open(file=f'/root/huangshan/research/marca/exp/log/proportion_{low_condition}_{high_condition}.csv', mode='a') as f:
                print(f"{self.layer_idx}, {percentage1:.2f}%, {percentage2:.2f}%", file=f)
        elif mode == 'inside':
            condition = (input >= low_condition) & (input <= high_condition)
            count = torch.sum(condition).item()
            percentage = (count / total_elements) * 100
            with open(file=f'/root/huangshan/research/marca/exp/log/proportion_{low_condition}_{high_condition}.csv', mode='a') as f:
                print(f"{self.layer_idx}, {percentage:.2f}%", file=f)


    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        # if causal_conv1d_update is None:
        if causal_conv1d_update is not None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        # if selective_state_update is None:
        if selective_state_update is not None:
            # Discretize A and B
            import matplotlib.pyplot as plt
            # plt.hist((dt + self.dt_proj.bias.to(dtype=dt.dtype)).flatten().cpu().numpy(), bins=100, density=True)
            # plt.savefig(f'/home/huangshan/huangshan/research/mamba/fig/sfp_input/{self.layer_idx}_sfp_input')
            # plt.close()
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dtA = torch.einsum("bd,dn->bdn", dt, A)

            with open(file=f'/root/huangshan/research/marca/exp/log/dt_A.log', mode='a') as f:
                print(f"max: {dtA.max()}, min: {dtA.min()}", file=f)

            # plt.hist(dtA.flatten().cpu().numpy(), bins=100, density=True)
            # plt.savefig(f'/root/huangshan/research/marca/exp/fig/exp_input/{self.layer_idx}_exp_input')
            # plt.close()
            dA = torch.exp(dtA)

            self.compute_proportion(dA, 0.01, 0.99, 'outside')
 
            dB = torch.einsum("bd,bn->bdn", dt, B)

            self.compute_proportion(dB, -0.05, 0.05, 'inside')

            # self.compute_proportion(x, -0.02, 0.02, 'inside')
            # plt.hist(B.flatten().cpu().numpy(), bins=100, density=True)
            # plt.savefig(f'/root/huangshan/research/marca/exp/fig/B/{self.layer_idx}_B')
            # plt.close()

            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        # if not self.fused_add_norm:
        if True:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            # print("block residual: ", residual.shape)
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            # print("block norm: ", hidden_states.shape)
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
