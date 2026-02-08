from dataclasses import dataclass
from typing import Optional, List
from torch import nn
from torch import Tensor
from torch.nn.init import xavier_uniform_


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: float = 8.0
    bias: bool = False
    learnable_alpha: bool = False
    apply_q: bool = True
    apply_k: bool = False
    apply_v: bool = True
    apply_out: bool = False


def xavier_uniform_nullable(module):
    if module is not None:
        xavier_uniform_(module)


def init_normal_nullable(module, std=0.02):
    if module is not None:
        nn.init.normal_(module, 0.0, std)


def init_zero_nullable(module):
    if module is not None:
        nn.init.zeros_(module)


def lora_in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_lora_proj_weight_a: Optional[Tensor] = None,
    k_lora_proj_weight_a: Optional[Tensor] = None,
    v_lora_proj_weight_a: Optional[Tensor] = None,
    q_lora_proj_weight_b: Optional[Tensor] = None,
    k_lora_proj_weight_b: Optional[Tensor] = None,
    v_lora_proj_weight_b: Optional[Tensor] = None,
    q_lora_proj_bias: Optional[Tensor] = None,
    k_lora_proj_bias: Optional[Tensor] = None,
    v_lora_proj_bias: Optional[Tensor] = None,
) -> List[Tensor]:
    result = []
    if q_lora_proj_weight_a is not None:
        if q_lora_proj_bias is not None:
            result.append(
                q @ q_lora_proj_weight_a @ q_lora_proj_weight_b + q_lora_proj_bias
            )
        else:
            result.append(q @ q_lora_proj_weight_a @ q_lora_proj_weight_b)
    else:
        result.append(0)

    if k_lora_proj_weight_a is not None:
        if k_lora_proj_bias is not None:
            result.append(
                k @ k_lora_proj_weight_a @ k_lora_proj_weight_b + k_lora_proj_bias
            )
        else:
            result.append(k @ k_lora_proj_weight_a @ k_lora_proj_weight_b)
    else:
        result.append(0)

    if v_lora_proj_weight_a is not None:
        if v_lora_proj_bias is not None:
            result.append(
                v @ v_lora_proj_weight_a @ v_lora_proj_weight_b + v_lora_proj_bias
            )
        else:
            result.append(v @ v_lora_proj_weight_a @ v_lora_proj_weight_b)
    else:
        result.append(0)

    return result
