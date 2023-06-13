# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
from torch.nn import Parameter
from fairseq.modules import LayerNorm


@with_incremental_state
class JointMultiheadAttentionWeightedSum(nn.Module):
    """
    Joint Multi-headed attention on both contexts and images.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        
        # weighting bias
        self.memory_bias = Parameter(torch.zeros(self.num_heads))

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.beam_size = 1

        self.reset_parameters()

        self.onnx_trace = False
        self.skip_embed_dim_check = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        position_encoding: Optional[Tensor] = None,
        long_context_retrieval: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel X Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        num_heads = self.num_heads
        head_dim = self.head_dim
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()

        assert key is not None and value is not None
        if long_context_retrieval is not None:
            return self.joint_multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                tuple((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias,)) if self.q_proj.bias else None,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout_module.p,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training or self.dropout_module.apply_during_inference,
                key_padding_mask,
                need_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
                position_encoding=position_encoding,
                long_context_retrieval=long_context_retrieval,
            )
            # import time
            # start = time.time()
            # tgt_len, bsz, embed_dim = query.shape
            # src_len, _, _ = key.shape

            # q = self.q_proj(query)
            # k = self.k_proj(query)
            # v = self.v_proj(query)
            
            # retrieval_k, retrieval_v = long_context_retrieval['k'].to(q.device).type(q.dtype), long_context_retrieval['v'].to(q.device).type(q.dtype)

            # # perform normalization
            # # q = torch.nn.functional.normalize(q, dim=-1, p=2)
            # # k = torch.nn.functional.normalize(k, dim=-1, p=2)
            # # retrieval_k = torch.nn.functional.normalize(retrieval_k, dim=-1, p=2)

            # # prep attention mask
            # if attn_mask is not None:
            #     attn_mask = attn_mask.unsqueeze(0)
            
            # # reshape q, k, v for multihead attention and make em batch first
            # q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
            # k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
            # v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
            
            # _, _, num_k, _ = retrieval_k.shape
            # assert retrieval_k.shape[0] == bsz * num_heads
            
            # # retrieval_k = retrieval_k.transpose(0, 1).transpose(1, 2).contiguous().view(tgt_len, num_k, bsz * num_heads, head_dim).transpose(1, 2).transpose(0, 1)
            # # retrieval_v = retrieval_v.transpose(0, 1).transpose(1, 2).contiguous().view(tgt_len, num_k, bsz * num_heads, head_dim).transpose(1, 2).transpose(0, 1)
            # src_len += num_k

            # # (deep breath) calculate attention and out projection
            # # q = q / math.sqrt(self.head_dim)
            # q *= self.scaling
            # attn_text = torch.bmm(q, k.transpose(-2, -1))
            # attn_retrieval = torch.matmul(q.unsqueeze(-2), retrieval_k.transpose(-2, -1)).squeeze(-2)
            # # attn = torch.cat((attn_text, attn_retrieval), dim=-1)
            # # position_encoding = torch.cat((position_encoding.view(bsz * num_heads, 1, tgt_len), torch.zeros((bsz * num_heads, 1, num_k), device=q.device, dtype=q.dtype)), dim=-1)
            # attn_text = attn_text + position_encoding.view(bsz * num_heads, 1, tgt_len) # add alibi tensor here
            
            # if attn_mask is not None:
            #     attn_text += attn_mask
            # attn_text_probs = torch.softmax(attn_text, dim=-1)
            # attn_retrieval_probs = torch.softmax(attn_retrieval, dim=-1)
            # # print(attn_retrieval_probs)
            # attn_text = self.dropout_module(attn_text_probs)
            # attn_output = torch.bmm(attn_text_probs, v) * (1 - torch.sigmoid(self.memory_bias.reshape(self.num_heads, 1, 1).repeat(bsz, 1, 1))) + torch.matmul(attn_retrieval_probs.unsqueeze(-2), retrieval_v).squeeze(-2) * torch.sigmoid(self.memory_bias.reshape(num_heads, 1, 1).repeat(bsz, 1, 1))
            # # attn_output = attn_output_text + attn_output_retrieval
            # attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
            
            # attn_output = self.out_proj(attn_output)
            # print(time.time() - start)
            # if need_weights:
            #     # average attention weights over heads
            #     attn_output_weights = attn.view(bsz, num_heads, tgt_len, src_len)
            #     attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
            #     return attn_output, attn_output_weights
            # else:
            #     return attn_output, None
        else:
            tgt_len, bsz, embed_dim = query.size()
            src_len = tgt_len
            
            if key is not None:
                src_len, key_bsz, _ = key.size()
                if not torch.jit.is_scripting():
                    assert key_bsz == bsz
                    assert value is not None
                    assert src_len, bsz == value.shape[:2]

            if incremental_state is not None:
                saved_state = self._get_input_buffer(incremental_state)
                if saved_state is not None and "prev_key" in saved_state:
                    # previous time steps are cached - no need to recompute
                    # key and value if they are static
                    if static_kv:
                        assert self.encoder_decoder_attention and not self.self_attention
                        key = value = None
            else:
                saved_state = None

            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
            
            q *= self.scaling

            q = (
                q.contiguous()
                .view(tgt_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
            if k is not None:
                k = (
                    k.contiguous()
                    .view(-1, bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
                )
            if v is not None:
                v = (
                    v.contiguous()
                    .view(-1, bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
                )

            if saved_state is not None:
                # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
                if "prev_key" in saved_state:
                    _prev_key = saved_state["prev_key"]
                    assert _prev_key is not None
                    prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                    if static_kv:
                        k = prev_key
                    else:
                        assert k is not None
                        k = torch.cat([prev_key, k], dim=1)
                    src_len = k.size(1)
                if "prev_value" in saved_state:
                    _prev_value = saved_state["prev_value"]
                    assert _prev_value is not None
                    prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                    if static_kv:
                        v = prev_value
                    else:
                        assert v is not None
                        v = torch.cat([prev_value, v], dim=1)
                prev_key_padding_mask: Optional[Tensor] = None
                if "prev_key_padding_mask" in saved_state:
                    prev_key_padding_mask = saved_state["prev_key_padding_mask"]
                assert k is not None and v is not None
                key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                    key_padding_mask=key_padding_mask,
                    prev_key_padding_mask=prev_key_padding_mask,
                    batch_size=bsz,
                    src_len=k.size(1),
                    static_kv=static_kv,
                )

                saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
                saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
                saved_state["prev_key_padding_mask"] = key_padding_mask
                # In this branch incremental_state is never None
                assert incremental_state is not None
                incremental_state = self._set_input_buffer(incremental_state, saved_state)
            assert k is not None

            # This is part of a workaround to get around fork/join parallelism
            # not supporting Optional types.
            if key_padding_mask is not None and key_padding_mask.dim() == 0:
                key_padding_mask = None

            if key_padding_mask is not None:
                if key_val is not None and key_padding_mask.size(1) == tgt_len:
                    key_padding_mask = torch.cat([torch.zeros(bsz, src_len-tgt_len).type_as(key_padding_mask), key_padding_mask], dim=1)
                assert key_padding_mask.size(0) == bsz
                assert key_padding_mask.size(1) == src_len

            if self.add_zero_attn:
                assert v is not None
                src_len += 1
                k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
                v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
                if attn_mask is not None:
                    attn_mask = torch.cat(
                        [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                    )
                if key_padding_mask is not None:
                    key_padding_mask = torch.cat(
                        [
                            key_padding_mask,
                            torch.zeros(key_padding_mask.size(0), 1).type_as(
                                key_padding_mask
                            ),
                        ],
                        dim=1,
                    )

            attn_weights = torch.bmm(q, k.transpose(1, 2))
            attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

            assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

            position_encoding = position_encoding.view(bsz * self.num_heads, 1, src_len)
            attn_weights = attn_weights + position_encoding # add alibi tensor here

            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(0)
                if self.onnx_trace:
                    attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
                # if key_val is not None:
                #     attn_mask = torch.cat([attn_mask[:,:,0:1], torch.zeros(1, tgt_len, src_len-tgt_len).type_as(attn_mask), attn_mask[:,:,1:]], dim=2)
                #     attn_mask[:,0,1:] = attn_mask[0,0,-1]
                attn_weights += attn_mask

            if key_padding_mask is not None:
                # don't attend to padding symbols
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                if not is_tpu:
                    attn_weights = attn_weights.masked_fill(
                        key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                        float("-inf"),
                    )
                else:
                    attn_weights = attn_weights.transpose(0, 2)
                    attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                    attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
                
            if before_softmax:
                return attn_weights, v

            attn_probs = utils.softmax(
                    attn_weights, dim=-1, onnx_trace=self.onnx_trace
                )
            
            attn_probs = attn_probs.type_as(k)
            attn_probs = self.dropout_module(attn_probs)

            assert v is not None
            attn = torch.bmm(attn_probs, v)
            
            assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
            if self.onnx_trace and attn.size(1) == 1:
                # when ONNX tracing a single decoder step (sequence length == 1)
                # the transpose is a no-op copy before view, thus unnecessary
                attn = attn.contiguous().view(tgt_len, bsz, self.embed_dim)
            else:
                attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
            
            attn = self.out_proj(attn)
            attn_weights: Optional[Tensor] = None
            if need_weights:
                attn_weights = attn_weights_float.view(
                    bsz, self.num_heads, tgt_len, src_len
                ).transpose(1, 0)
                if not need_head_weights:
                    # average attention weights over heads
                    attn_weights = attn_weights.mean(dim=0)

            return attn, attn_weights
    
    # @staticmethod
    def joint_multi_head_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Optional[Tensor],
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        retrieval_k_proj_weight: Optional[Tensor] = None,
        retrieval_v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
        position_encoding: Optional[Tensor] = None,
        long_context_retrieval: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        '''
        Inputs:
        long_context_retrieval: (batch_size, seq_len, k, embedding_size)
        '''

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        assert embed_dim == embed_dim_to_check, \
            f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        assert use_separate_proj_weight == True

        # compute in-projection
        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight, in_proj_bias[0])
            k = F.linear(key, k_proj_weight, in_proj_bias[1])
            v = F.linear(value, v_proj_weight, in_proj_bias[2])
        else:
            q = F.linear(query, q_proj_weight, in_proj_bias)
            k = F.linear(key, k_proj_weight, in_proj_bias)
            v = F.linear(value, v_proj_weight, in_proj_bias)
        
        retrieval_k, retrieval_v = long_context_retrieval['k'].to(q.device).type(q.dtype), long_context_retrieval['v'].to(q.device).type(q.dtype)

        # perform normalization
        # q = torch.nn.functional.normalize(q, dim=-1, p=2)
        # k = torch.nn.functional.normalize(k, dim=-1, p=2)
        # retrieval_k = torch.nn.functional.normalize(retrieval_k, dim=-1, p=2)

        # prep attention mask
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
        
        # reshape q, k, v for multihead attention and make em batch first
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        
        _, _, num_k, _ = retrieval_k.shape
        assert retrieval_k.shape[0] == bsz * num_heads
        
        src_len += num_k

        # (deep breath) calculate attention and out projection
        # q = q / math.sqrt(self.head_dim)
        q *= self.scaling
        attn_text = torch.bmm(q, k.transpose(-2, -1))
        attn_retrieval = torch.matmul(q.unsqueeze(-2), retrieval_k.transpose(-2, -1)).squeeze(-2)
        # attn = torch.cat((attn_text, attn_retrieval), dim=-1)
        # position_encoding = torch.cat((position_encoding.view(bsz * num_heads, 1, tgt_len), torch.zeros((bsz * num_heads, 1, num_k), device=q.device, dtype=q.dtype)), dim=-1)
        attn_text = attn_text + position_encoding.view(bsz * num_heads, 1, tgt_len) # add alibi tensor here
        
        if attn_mask is not None:
            attn_text += attn_mask
        attn_text_probs = torch.softmax(attn_text, dim=-1)
        attn_retrieval_probs = torch.softmax(attn_retrieval, dim=-1)
        attn_text_probs = self.dropout_module(attn_text_probs)

        attn_output = torch.bmm(attn_text_probs, v) * (1 - torch.sigmoid(self.memory_bias.reshape(self.num_heads, 1, 1).repeat(bsz, 1, 1))) + torch.matmul(attn_retrieval_probs.unsqueeze(-2), retrieval_v).squeeze(-2) * torch.sigmoid(self.memory_bias.reshape(self.num_heads, 1, 1).repeat(bsz, 1, 1))
        # print(torch.sigmoid(self.memory_bias))

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        
        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_text.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
            return attn_output, attn_output_weights
        else:
            return attn_output, None

    @staticmethod
    def simplified_joint_multi_head_attention_forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Optional[Tensor],
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        retrieval_k_proj_weight: Optional[Tensor] = None,
        retrieval_v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
        long_context_retrieval: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        assert embed_dim == embed_dim_to_check, \
            f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        assert use_separate_proj_weight == True
        
        # compute in-projection
        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight, in_proj_bias[0])
            k = F.linear(key, k_proj_weight, in_proj_bias[1])
            v = F.linear(value, v_proj_weight, in_proj_bias[2])
            retrieval_k = F.linear(long_context_retrieval, retrieval_k_proj_weight, in_proj_bias[3])
            retrieval_v = F.linear(long_context_retrieval, retrieval_v_proj_weight, in_proj_bias[4])
        else:
            q = F.linear(query, q_proj_weight, in_proj_bias)
            k = F.linear(key, k_proj_weight, in_proj_bias)
            v = F.linear(value, v_proj_weight, in_proj_bias)
            retrieval_k = F.linear(long_context_retrieval, retrieval_k_proj_weight, in_proj_bias)
            retrieval_v = F.linear(long_context_retrieval, retrieval_v_proj_weight, in_proj_bias)
        
        q = q.contiguous().view(tgt_len, bsz, num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(k.shape[0], bsz, num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(v.shape[0], bsz, num_heads, head_dim).transpose(0, 1)

        _, _, num_k, _ = retrieval_k.shape
        # retrieval_k, v: bsz, tgt_len, num_k, embd
        retrieval_k = retrieval_k.contiguous().view(bsz, tgt_len, num_k, num_heads, head_dim).transpose(2, 3)
        retrieval_v = retrieval_v.contiguous().view(bsz, tgt_len, num_k, num_heads, head_dim).transpose(2, 3)
        src_len += num_k

        # (deep breath) calculate attention and out projection
        q = q / math.sqrt(embed_dim)
        attn_text = torch.einsum("...qhd,...khd->...hqk", q, k)
        attn_retrieval = torch.einsum("...qhd, ...qhid->...hqi", q, retrieval_k)
        attn = torch.cat((attn_text, attn_retrieval), dim=-1)
        if attn_mask is not None:
            attn_mask = torch.cat((attn_mask, torch.zeros((tgt_len, num_k), device=q.device)), dim=-1)
            attn += attn_mask
        attn = torch.softmax(attn, dim=-1)
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)
        attn_text = attn[:, :, :, :tgt_len]
        attn_retrieval = attn[:, :, :, tgt_len:]

        attn_output = torch.einsum("...hqk,...khd->...qhd", attn_text, v) + torch.einsum("...hqi,...qhid->...qhd", attn_retrieval, retrieval_v)
        attn_output = attn_output.contiguous().view(bsz, tgt_len, embed_dim).transpose(0, 1)
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        
        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn.sum(dim=1) / num_heads
            return attn_output, attn_output_weights
        else:
            return attn_output, None

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - prev_key_padding_mask.size(1)),
                    device=prev_key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [prev_key_padding_mask.float(), filler.float()], dim=1
                )
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - key_padding_mask.size(1)),
                    device=key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [filler.float(), key_padding_mask.float()], dim=1
                )
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(
                        0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value