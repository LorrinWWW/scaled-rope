import math
import warnings
from functools import partial
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaModel
from transformers.models.llama.modeling_llama import LlamaAttention

from flash_attn.flash_attn_interface import (
    flash_attn_func, 
    flash_attn_kvpacked_func, 
    flash_attn_qkvpacked_func,
    flash_attn_varlen_kvpacked_func, 
)


def compute_flash_attention(flash_attn, q, k, v, attention_mask=None, head_mask=None):

    attn_outputs = flash_attn_func(
        q, k, v, dropout_p=0.0, softmax_scale=None, causal=True, return_attn_probs=False)

    return attn_outputs


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def llama_forward_with_flash_attn(
    self: LlamaAttention,
    flash_attn: nn.Module,  # flash_attn.modules.mha.FlashSelfAttention
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # FAv2 handles this
    # repeat k/v heads if n_kv_heads < n_heads
    # key_states = repeat_kv(key_states, self.num_key_value_groups)
    # value_states = repeat_kv(value_states, self.num_key_value_groups)

    flash_attn.train(self.training)
    # out_dtype = value_states.dtype
    q, k, v = (
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
    )
    attn_output = compute_flash_attention(flash_attn, q, k, v, attention_mask)
    # attn_output = attn_output.to(out_dtype)

    attn_output = attn_output.view(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def add_dropout(module: nn.Module, patched_fwd: Callable, p_dropout: float = 0.1):
    dropout = nn.Dropout(p=p_dropout)
    module.old_forward = module.forward
    module.forward = partial(patched_fwd, dropout, module)


def add_flash_attn(module: nn.Module, causal: bool = True):
    """
    Replaces the standard attention implementation with Flash Attention [1].
    Limitations:
      - Only works for fp16 or bf16 inputs
      - Requires inputs to be on CUDA
      - `output_attentions=True` does not work after patching, attention weights will be None
      - Non-contiguous attention masks are not supported (e.g. [1, 1, 0, 1, 1, 0, 0] will just become [1, 1, 1, 1, 1, 0, 0]).

    [1] https://github.com/HazyResearch/flash-attention
    """

    flash_attn = FlashSelfAttention(causal=causal)
    if isinstance(module, LlamaAttention):
        module.old_forward = module.forward
        module.forward = partial(llama_forward_with_flash_attn, module, flash_attn)


def _patched_mlp_forward(post_module: nn.Module, module: nn.Module, *args, **kwargs):
    post_module.train(module.training)
    out = module.old_forward(*args, **kwargs)
    out = post_module(out)
    return out


def _patched_attn_forward(post_module: nn.Module, module: nn.Module, *args, **kwargs):
    post_module.train(module.training)
    out = module.old_forward(*args, **kwargs)
    hiddens = post_module(out[0])
    return (hiddens,) + out[1:]


def patch_model(
    model: nn.Module,
    resid_pdrop: Optional[float] = 0.1,
    flash_attention: bool = True,
    patch_unsupported: bool = False,
    residual_dropout_lima: bool = False,
):
    """
    Helper function for patching HF language models.
    Currently supports: GPTNeoX-based models

    Limitations:
      - Flash attention requires CUDA and fp16/bf16 training. It also requires contiguous attention masks.
      - Residual dropout does not support multi-GPU training without DeepDpeed.
    """
    global FlashSelfAttention
    if flash_attention:
        try:
            from flash_attn.modules.mha import \
                FlashSelfAttention  # pyright: reportMissingImports=false
        except ModuleNotFoundError:
            warnings.warn(
                """\nmodule flash_attn not found - either install:
  pip3 install flash_attn
or run with:
  --use_flash_attention=false """
            )
            exit(1)

    if isinstance(model, LlamaForCausalLM):
        model = model.model

    if model.__class__.__name__ == "RWForCausalLM":
        model = model.base_model

    attention_key_lookup = {
        LlamaModel: "self_attn",
    }
    mlp_key_lookup = {
        LlamaModel: "mlp",
    }
    if model.__class__.__name__ == "RWModel":
        layers = model.h
        attention_key = "self_attention"
        mlp_key = "mlp"
    else:
        layers = model.layers
        attention_key = attention_key_lookup.get(model.__class__, "attention")
        mlp_key = mlp_key_lookup.get(model.__class__, "mlp")
    num_layers = len(layers)
    resid_pdrop_last_layer = resid_pdrop
    for i, layer in enumerate(layers):
        if flash_attention:
            add_flash_attn(getattr(layer, attention_key), causal=True)
        if residual_dropout_lima:
            resid_pdrop = i / (num_layers - 1) * resid_pdrop_last_layer
        if resid_pdrop is not None and resid_pdrop > 0:
            add_dropout(
                getattr(layer, attention_key), _patched_attn_forward, resid_pdrop
            )
            add_dropout(getattr(layer, mlp_key), _patched_mlp_forward, resid_pdrop)
