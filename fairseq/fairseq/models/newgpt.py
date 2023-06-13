# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from typing import Dict, List, Optional

from .hf_newgpt import NewGPTConfig, NewGPTForCausalLM

import torch
import torch.nn as nn
from fairseq import checkpoint_utils
from fairseq.models import (
    FairseqIncrementalDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)


logger = logging.getLogger(__name__)


@register_model("newgpt")
class NewGPTLanguageModel(FairseqLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--gpt-model-path', default="", help='checkpoint path')
        parser.add_argument('--embed-dim', type=int, metavar='N',
                            help='embedding dimension')
        parser.add_argument('--num-attention-heads', type=int, metavar='N',
                            help='num attention heads')
        parser.add_argument('--num-layers', type=int, metavar='N',
                            help='num layers')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability for all fully connected layers '
                                 'in the embeddings, encoder, and pooler')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--newgpt-mode', type=str, metavar='N',
                            help='change the model structure')
        parser.add_argument('--result-path', type=str, metavar='N',
                            help='path to save results',
                            default="./output/debug.json")
        parser.add_argument('--retrieval-layer-index', type=int, metavar='N',
                            help='The layer index for retrieval',
                            default="17")
        parser.add_argument('--use-external-memory', action="store_true",
                            help='use external memory or not',
                            default=False)
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        
        default_architecture(args)
        model = cls(NewGPTDecoder(args, task))
        if args.gpt_model_path != "":
            state = checkpoint_utils.load_checkpoint_to_cpu(args.gpt_model_path)
            model.load_state_dict(state["model"], strict=True, args=args)

        return model


class NewGPTDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, task):
        super().__init__(task.target_dictionary)

        config = NewGPTConfig(
            vocab_size=len(task.target_dictionary),
            # n_ctx=args.max_target_positions,
            n_embd=args.embed_dim,
            mode=args.newgpt_mode,
            max_position_embeddings=args.tokens_per_sample,
            n_layer=args.num_layers,
            n_head=args.num_attention_heads,
            resid_pdrop=args.dropout,
            embd_pdrop=args.dropout,
            attn_pdrop=args.attention_dropout,
            layer_norm_epsilon=1e-6,
            retrieval_layer_index=args.retrieval_layer_index,
            use_external_memory=getattr(args, "use_external_memory", False),
        )
        self.model = NewGPTForCausalLM(config)

        # set zero embedding for padding symbol
        self.pad_idx = task.target_dictionary.pad()
        self.model.transformer.wte.weight.data[self.pad_idx].zero_()

    def forward(
        self,
        prev_output_tokens,
        src_lengths=None,
        incremental_state: Optional[Dict[str, List[torch.Tensor]]] = None,
        encoder_out=None,
        features_only: Optional[bool] = False,
        return_all_hiddens: Optional[bool] = False,
        disable_add_index: Optional[bool] = False,
    ):
        features, all_hidden_states, kv_dict, position_encoding = self.extract_features(prev_output_tokens, incremental_state, disable_add_index=disable_add_index)

        lm_logits = self.model.lm_head(features)
        if return_all_hiddens:
            return lm_logits, all_hidden_states, kv_dict, position_encoding
        if features_only:
            return features
        return (lm_logits, None)

    def extract_features(
        self,
        prev_output_tokens,
        incremental_state: Optional[Dict[str, List[torch.Tensor]]] = None,
        disable_add_index: Optional[bool] = False,
    ):
        if incremental_state:
            past = self.get_incremental_state("past")
        else:
            past = None

        # don't attend to padding symbols
        attention_mask = prev_output_tokens.ne(self.pad_idx).int()

        outputs, kv_dict, position_encoding = self.model.transformer(
            input_ids=prev_output_tokens,
            past_key_values=past,
            attention_mask=attention_mask,
            output_hidden_states=True,
            external_memory=self.external_memory,
            disable_add_index=disable_add_index,
        )
        last_hidden_state = outputs.last_hidden_state
        all_hidden_states = outputs.hidden_states

        if incremental_state:
            self.set_incremental_state(incremental_state, "past", outputs[1])

        return last_hidden_state, all_hidden_states, kv_dict, position_encoding


@register_model_architecture("newgpt", "newgpt-small")
def default_architecture(args):
    args.embed_dim = getattr(args, "embed_dim", 768)
    args.num_attention_heads = getattr(args, "num_attention_heads", 12)
    args.num_layers = getattr(args, "num_layers", 12)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.tokens_per_sample = getattr(args, "tokens_per_sample", 2048)
    args.newgpt_window = getattr(args, "newgpt_window", args.tokens_per_sample)
    args.retrieval_layer_index = getattr(args, "retrieval_layer_index", 10)

@register_model_architecture("newgpt", "newgpt-mini")
def newgpt_medium(args):
    args.embed_dim = getattr(args, "embed_dim", 128)
    args.num_attention_heads = getattr(args, "num_attention_heads", 4)
    args.num_layers = getattr(args, "num_layers", 8)
    default_architecture(args)

@register_model_architecture("newgpt", "newgpt-medium")
def newgpt_medium(args):
    args.embed_dim = getattr(args, "embed_dim", 1024)
    args.num_attention_heads = getattr(args, "num_attention_heads", 16)
    args.num_layers = getattr(args, "num_layers", 24)
    default_architecture(args)


@register_model_architecture("newgpt", "newgpt-large")
def newgpt_large(args):
    args.embed_dim = getattr(args, "embed_dim", 1280)
    args.num_attention_heads = getattr(args, "num_attention_heads", 20)
    args.num_layers = getattr(args, "num_layers", 36)
    default_architecture(args)


@register_model_architecture("newgpt", "newgpt-xl")
def newgpt_xl(args):
    args.embed_dim = getattr(args, "embed_dim", 1600)
    args.num_attention_heads = getattr(args, "num_attention_heads", 25)
    args.num_layers = getattr(args, "num_layers", 48)
    default_architecture(args)
