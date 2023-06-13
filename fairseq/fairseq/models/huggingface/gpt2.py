# import torch
# import warnings

# from torch.nn import CrossEntropyLoss
# from fairseq import utils
# from fairseq.models import (
#     FairseqIncrementalDecoder,
#     FairseqLanguageModel,
#     register_model,
#     register_model_architecture,
#     BaseFairseqModel,
# )

# try:
#     from transformers import GPT2Config, GPT2LMHeadModel
# except ImportError:
#     raise ImportError(
#         "\n\nPlease install huggingface/transformers with:"
#         "\n\n  pip install transformers"
#     )

# # Copyright (c) Facebook, Inc. and its affiliates.
# #
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

# import logging
# import os
# import sys
# from typing import Dict, List, Optional

# import torch
# from fairseq.models import (
#     FairseqIncrementalDecoder,
#     FairseqLanguageModel,
#     register_model,
#     register_model_architecture,
# )


# logger = logging.getLogger(__name__)


# DEFAULT_MAX_TARGET_POSITIONS = 1024


# @register_model("gpt2_pretrained")
# class HuggingFaceGPT2LanguageModel(FairseqLanguageModel):
#     def __init__(self, decoder):
#         super().__init__(decoder)

#     @staticmethod
#     def add_args(parser):
#         """Add model-specific arguments to the parser."""
#         # fmt: off
#         parser.add_argument('--embed-dim', type=int, metavar='N',
#                             help='embedding dimension')
#         parser.add_argument('--num-attention-heads', type=int, metavar='N',
#                             help='num attention heads')
#         parser.add_argument('--num-layers', type=int, metavar='N',
#                             help='num layers')
#         parser.add_argument('--dropout', type=float, metavar='D',
#                             help='dropout probability for all fully connected layers '
#                                  'in the embeddings, encoder, and pooler')
#         parser.add_argument('--attention-dropout', type=float, metavar='D',
#                             help='dropout probability for attention weights')
#         # fmt: on

#     @classmethod
#     def build_model(cls, args, task):
#         """Build a new model instance."""
#         # default_architecture(args)
        
#         return cls(HuggingFaceGPT2Decoder(args, task))


# class HuggingFaceGPT2Decoder(FairseqIncrementalDecoder):
#     def __init__(self, args, task):
#         try:
#             from transformers import GPT2Config, GPT2LMHeadModel
#         except ImportError:
#             raise ImportError(
#                 "\n\nPlease install huggingface/transformers with:"
#                 "\n\n  pip install transformers"
#             )

#         super().__init__(task.target_dictionary)

#         hf_config = GPT2Config.from_pretrained(args.hf_config)
#         hf_model = GPT2LMHeadModel.from_pretrained(args.hf_config)
#         self.model = GPT2LMHeadModel(config)

#         # set zero embedding for padding symbol
#         self.pad_idx = task.target_dictionary.pad()
#         self.model.transformer.wte.weight.data[self.pad_idx].zero_()
#         self.model.transformer.wpe.weight.data[0].zero_()

#     def forward(
#         self,
#         prev_output_tokens,
#         src_lengths=None,
#         incremental_state: Optional[Dict[str, List[torch.Tensor]]] = None,
#         encoder_out=None,
#     ):
#         features = self.extract_features(prev_output_tokens, incremental_state)
#         lm_logits = self.model.lm_head(features)
#         return (lm_logits,)

#     def extract_features(
#         self,
#         prev_output_tokens,
#         incremental_state: Optional[Dict[str, List[torch.Tensor]]] = None,
#     ):
#         if incremental_state:
#             past = self.get_incremental_state("past")
#         else:
#             past = None

#         # don't attend to padding symbols
#         attention_mask = prev_output_tokens.ne(self.pad_idx).int()

#         # set position ids to exclude padding symbols
#         position_ids = attention_mask * (
#             torch.arange(1, 1 + prev_output_tokens.size(1))
#             .to(prev_output_tokens)
#             .repeat(prev_output_tokens.size(0), 1)
#         )

#         outputs = self.model.transformer(
#             input_ids=prev_output_tokens,
#             past=past,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#         )
#         last_hidden_states = outputs[0]

#         if incremental_state:
#             self.set_incremental_state(incremental_state, "past", outputs[1])

#         return last_hidden_states

#     def max_positions(self):
#         return self.model.config.n_positions - 1


# @register_model_architecture("hf_gpt2", "hf_gpt2")
# def default_architecture(args):
#     if getattr(args, "max_target_positions", None) is None:
#         args.max_target_positions = getattr(
#             args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
#         )
#     args.embed_dim = getattr(args, "embed_dim", 768)
#     args.num_attention_heads = getattr(args, "num_attention_heads", 12)
#     args.num_layers = getattr(args, "num_layers", 12)
#     args.dropout = getattr(args, "dropout", 0.1)
#     args.attention_dropout = getattr(args, "attention_dropout", 0.1)


# @register_model_architecture("hf_gpt2", "hf_gpt2_medium")
# def hf_gpt2_medium(args):
#     args.embed_dim = getattr(args, "embed_dim", 1024)
#     args.num_attention_heads = getattr(args, "num_attention_heads", 16)
#     args.num_layers = getattr(args, "num_layers", 24)
#     default_architecture(args)


# @register_model_architecture("hf_gpt2", "hf_gpt2_large")
# def hf_gpt2_large(args):
#     args.embed_dim = getattr(args, "embed_dim", 1280)
#     args.num_attention_heads = getattr(args, "num_attention_heads", 20)
#     args.num_layers = getattr(args, "num_layers", 36)
#     default_architecture(args)


# @register_model_architecture("hf_gpt2", "hf_gpt2_xl")
# def hf_gpt2_xl(args):
#     args.embed_dim = getattr(args, "embed_dim", 1600)
#     args.num_attention_heads = getattr(args, "num_attention_heads", 25)
#     args.num_layers = getattr(args, "num_layers", 48)
#     default_architecture(args)





# def get_dep_trg_net_input(src_tokens, src_attention_mask):
#   dep_tokens = src_tokens[:, :1].clone()
#   dep_attention_mask = src_attention_mask[:, :1].clone()
#   return dep_tokens, dep_attention_mask


# @register_model('gpt2')
# class GPT2HfWrapper(FairseqLanguageModel):

#   def __init__(self, args, hf_config, hf_model):
#     super().__init__()
#     self.args = args
#     self.hf_config = hf_config
#     self.hf_model = hf_model
  
#   @staticmethod
#   def add_args(parser):
#     # parser.add_argument("--hf_config", type=str, help="path to model config file")
#     pass
  
#   @classmethod
#   def build_model(cls, args, task):
#     hf_config = MT5Config.from_pretrained(args.hf_config)
#     hf_model = T5ForConditionalGeneration(hf_config)
#     return cls(args, hf_config, hf_model)
  
#   def get_dep_trg_net_input(self, src_tokens, src_attention_mask):
#     dep_tokens = src_tokens[:, :1].clone()
#     dep_attention_mask = src_attention_mask[:, :1].clone()
#     return dep_tokens, dep_attention_mask
  
#   def forward_encoding(self, src_net_input, **kwargs):
#     src_tokens, src_attention_mask = self.parse_net_input(src_net_input)
#     trg_tokens, trg_attention_mask = self.get_dep_trg_net_input(src_tokens, src_attention_mask)
#     outputs = self.hf_model(input_ids=src_tokens, labels=trg_tokens, attention_mask=src_attention_mask, decoder_attention_mask=trg_attention_mask, **kwargs)
#     return outputs

#   def forward(self, src_net_input, trg_net_input):
#     src_tokens, src_attention_mask = self.parse_net_input(src_net_input)
#     trg_tokens, trg_attention_mask = self.parse_net_input(trg_net_input)
#     outputs = self.hf_model(input_ids=src_tokens, labels=trg_tokens, attention_mask=src_attention_mask, decoder_attention_mask=trg_attention_mask)
#     return outputs
  
#   def seq2seq_outputs_to_encoder_outputs(self, s2s_outputs):
#     encoder_last_hidden_state=s2s_outputs.encoder_last_hidden_state
#     encoder_hidden_states=s2s_outputs.encoder_hidden_states
#     encoder_attentions=s2s_outputs.encoder_attentions
#     return (encoder_last_hidden_state, encoder_hidden_states, encoder_attentions)
  
#   def forward_decoding(self, encoder_outputs, trg_net_input):
#     trg_tokens, trg_attention_mask = self.parse_net_input(trg_net_input)
#     outputs = self.hf_model(labels=trg_tokens, encoder_outputs=encoder_outputs, decoder_attention_mask=trg_attention_mask)
#     return outputs

#   def constrained_generate_with_fixed_vocab(self, src_net_input, vocab_id_list, **kwargs):

#     def _prefix_allowed_tokens_fn(batch_id, sent):
#       return vocab_id_list
    
#     # return self.constrained_generate(src_net_input, _prefix_allowed_tokens_fn)
#     return self.generate(src_net_input, prefix_allowed_tokens_fn=_prefix_allowed_tokens_fn, **kwargs)
  
#   def generate(self, src_net_input, **kwargs):
#     src_tokens, src_attention_mask = self.parse_net_input(src_net_input)
#     outputs = self.hf_model.generate(input_ids=src_tokens, attention_mask=src_attention_mask, **kwargs)
#     return outputs

#   def parse_net_input(self, net_input):
#     tokens = net_input["src_tokens"]
#     lengths = net_input["src_lengths"]
#     _, max_len = tokens.size()
#     device = tokens.device
#     attention_mask = (torch.arange(max_len)[None, :].to(device) < lengths[:, None]).float()
#     return tokens, attention_mask


# @register_model_architecture('t5', 't5_mt5_small')
# def t5_small_architecture(args):
#   args.hf_config = getattr(args, "hf_config", "google/mt5-small")



# # Warning messafe for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
# __HEAD_MASK_WARNING_MSG = """
# The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
# `decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
# If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
# num_heads)`.
# """

# class T5ForConditionalGenerationDDP(T5ForConditionalGeneration):

#   def forward(
#     self,
#     input_ids=None,
#     attention_mask=None,
#     decoder_input_ids=None,
#     decoder_attention_mask=None,
#     head_mask=None,
#     decoder_head_mask=None,
#     encoder_outputs=None,
#     past_key_values=None,
#     inputs_embeds=None,
#     decoder_inputs_embeds=None,
#     labels=None,
#     use_cache=None,
#     output_attentions=None,
#     output_hidden_states=None,
#     return_dict=None,):

#     use_cache = use_cache if use_cache is not None else self.config.use_cache
#     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#     # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
#     if head_mask is not None and decoder_head_mask is None:
#       if self.config.num_layers == self.config.num_decoder_layers:
#         warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
#         decoder_head_mask = head_mask

#     # Encode if needed (training, first prediction pass)
#     if encoder_outputs is None:
#       # Convert encoder inputs in embeddings if needed
#       encoder_outputs = self.encoder(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         inputs_embeds=inputs_embeds,
#         head_mask=head_mask,
#         output_attentions=output_attentions,
#         output_hidden_states=output_hidden_states,
#         return_dict=return_dict,
#       )
#     elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
#       encoder_outputs = BaseModelOutput(
#         last_hidden_state=encoder_outputs[0],
#         hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
#         attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
#       )

#     hidden_states = encoder_outputs[0]

#     if self.model_parallel:
#       torch.cuda.set_device(self.decoder.first_device)

#     if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
#       # get decoder inputs from shifting lm labels to the right
#       decoder_input_ids = self._shift_right(labels)

#     # If decoding with past key value states, only the last tokens
#     # should be given as an input
#     if past_key_values is not None:
#       assert labels is None, "Decoder should not use cached key value states when training."
#       if decoder_input_ids is not None:
#         decoder_input_ids = decoder_input_ids[:, -1:]
#       if decoder_inputs_embeds is not None:
#         decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

#     # Set device for model parallelism
#     if self.model_parallel:
#       torch.cuda.set_device(self.decoder.first_device)
#       hidden_states = hidden_states.to(self.decoder.first_device)
#       if decoder_input_ids is not None:
#         decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
#       if attention_mask is not None:
#         attention_mask = attention_mask.to(self.decoder.first_device)
#       if decoder_attention_mask is not None:
#         decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

#     # Decode
#     decoder_outputs = self.decoder(
#       input_ids=decoder_input_ids,
#       attention_mask=decoder_attention_mask,
#       inputs_embeds=decoder_inputs_embeds,
#       past_key_values=past_key_values,
#       encoder_hidden_states=hidden_states,
#       encoder_attention_mask=attention_mask,
#       head_mask=decoder_head_mask,
#       encoder_head_mask=head_mask,
#       use_cache=use_cache,
#       output_attentions=output_attentions,
#       output_hidden_states=output_hidden_states,
#       return_dict=return_dict,
#     )

#     sequence_output = decoder_outputs[0]

#     # Set device for model parallelism
#     if self.model_parallel:
#       torch.cuda.set_device(self.encoder.first_device)
#       self.lm_head = self.lm_head.to(self.encoder.first_device)
#       sequence_output = sequence_output.to(self.lm_head.weight.device)

#     if self.config.tie_word_embeddings:
#       # Rescale output before projecting on vocab
#       # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
#       sequence_output = sequence_output * (self.model_dim ** -0.5)

#     lm_logits = self.lm_head(sequence_output)

#     loss = None
#     sample_size = None
#     if labels is not None:
#       loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='sum')
#       loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
#       sample_size = len(labels.view(-1))
#       # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

#     if not return_dict:
#       raise NotImplementedError
#       output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
#       return ((loss,) + output) if loss is not None else output

#     ret = Seq2SeqLMOutput(
#       loss=loss,
#       logits=lm_logits,
#       past_key_values=decoder_outputs.past_key_values,
#       decoder_hidden_states=decoder_outputs.hidden_states,
#       decoder_attentions=decoder_outputs.attentions,
#       cross_attentions=decoder_outputs.cross_attentions,
#       encoder_last_hidden_state=encoder_outputs.last_hidden_state,
#       encoder_hidden_states=encoder_outputs.hidden_states,
#       encoder_attentions=encoder_outputs.attentions,
#     )

#     ret.sample_size = sample_size
#     return ret


# @register_model('t5_ddp')
# class T5HfWrapperDDP(T5HfWrapper):

#   @classmethod
#   def build_model(cls, args, task):
#     hf_config = MT5Config.from_pretrained(args.hf_config)
#     hf_model = T5ForConditionalGenerationDDP(hf_config)
#     return cls(args, hf_config, hf_model)


# @register_model_architecture('t5_ddp', 't5_ddp_mt5_small')
# def t5_ddp_small_architecture(args):
#   args.hf_config = getattr(args, "hf_config", "google/mt5-small")