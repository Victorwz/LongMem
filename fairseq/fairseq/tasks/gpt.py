# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, field
from typing import Optional
import logging
import os
from argparse import Namespace
import json
from omegaconf import MISSING, II, OmegaConf
from typing import Any

import numpy as np
from fairseq import utils
from fairseq.data import Dictionary
from fairseq.tasks import FairseqTask, register_task
from fairseq.data.lm_loader import LMLoader
from fairseq.data.open_web_loader import OpenWebLMLoader
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from fairseq.data.encoders.gpt2_bpe import GPT2BPE

logger = logging.getLogger(__name__)

SAMPLE_BREAK_MODE_CHOICES = ChoiceEnum(["none", "complete", "complete_doc", "eos"])
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])

DEFAULT_ENCODER_JSON = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json"
DEFAULT_VOCAB_BPE = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe"


@dataclass
class GPTPretrainingkConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=2048,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    output_dictionary_size: int = field(
        default=-1, metadata={"help": "limit the size of output dictionary"}
    )
    self_target: bool = field(default=False, metadata={"help": "include self target"})
    future_target: bool = field(
        default=False, metadata={"help": "include future target"}
    )
    past_target: bool = field(default=False, metadata={"help": "include past target"})
    add_bos_token: bool = field(
        default=False, metadata={"help": "prepend beginning of sentence token (<s>)"}
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    pad_to_fixed_length: Optional[bool] = field(
        default=False,
        metadata={"help": "pad to fixed length"},
    )
    pad_to_fixed_bsz: Optional[bool] = field(
        default=False,
        metadata={"help": "boolean to pad to fixed batch size"},
    )

    gpt2_encoder_json: str = field(
        default=DEFAULT_ENCODER_JSON, metadata={"help": "path to encoder.json"}
    )
    gpt2_vocab_bpe: str = field(
        default=DEFAULT_VOCAB_BPE, metadata={"help": "path to vocab.bpe"}
    )

    required_batch_size_multiple: int = II("dataset.required_batch_size_multiple")

    batch_read_ahead: int = field(
        default=10000,
        metadata={"help": "batch read ahead size for infinibatch"},
    )

    mlm_cut_length: int = field(
        default=128,
        metadata={"help": "max input length for mlm"},
    )

    mlm_tokens_proportion: float = field(
        default=0.23,
        metadata={
            "help": "proportion of tokens produced by mlm"
        },
    )

    gpt_dict_path: str = field(
        default="",
        metadata={"help": "mlm checkpoint path"},
    )

    # TODO common vars below add to parent
    seed: int = II("common.seed")
    batch_size: Optional[int] = II("dataset.batch_size")
    batch_size_valid: Optional[int] = II("dataset.batch_size_valid")
    # dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
    #     "dataset.dataset_impl"
    # )
    data_buffer_size: int = II("dataset.data_buffer_size")
    tpu: bool = II("common.tpu")
    use_plasma_view: bool = II("common.use_plasma_view")
    plasma_path: str = II("common.plasma_path")


@register_task("gpt_pretraining", dataclass=GPTPretrainingkConfig)
class GPTTask(FairseqTask):

    def __init__(self, cfg, dictionary, tokenizer):
        super().__init__(cfg)
        self.cfg = cfg
        self.dictionary = dictionary
        self.tokenizer = tokenizer
        self.seed = cfg.seed

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0

        if "tnlg" in cfg.data:
            dictionary = Dictionary.load(cfg.gpt_dict_path)
        else:
            dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))

        dictionary.pad_to_multiple_(cfg.required_batch_size_multiple)
        logger.info("dictionary: {} types".format(len(dictionary)))

        tokenizer = GPT2BPE(Namespace(
            gpt2_vocab_bpe=cfg.gpt2_vocab_bpe,
            gpt2_encoder_json=cfg.gpt2_encoder_json))
        return cls(cfg, dictionary, tokenizer)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        if "tnlg" in self.cfg.data:
            self.datasets[split] = {
                'data': json.load(open(f'{self.cfg.data}/json/{split}-nogithub-noarvix-nopubmed.json')) if split == 'train' else json.load(open(f'{self.cfg.data}/json/{split}.json')),
                'data_dir': self.cfg.data,
                'shuffle': True if split == 'train' else False,
            }
        else:
            self.datasets[split] = {
                'data': json.load(open(f'{self.cfg.data}/json/{split}.json')),
                'data_dir': self.cfg.data,
                'shuffle': True if split == 'train' else False,
            }
        self.datasets[split] = Namespace(**self.datasets[split])
    
    def dataset(self, split):
        if split not in self.datasets:
            raise KeyError("Dataset not loaded: " + split)
        
        return self.datasets[split]

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False
    ):
        if "tnlg" in self.cfg.data:
            return LMLoader(
                    self.cfg,
                    dataset,
                    self.dictionary,
                    self.tokenizer,
                    max_tokens=max_tokens,
                    max_sentences=max_sentences,
                    max_positions=max_positions,
                    ignore_invalid_inputs=ignore_invalid_inputs,
                    required_batch_size_multiple=required_batch_size_multiple,
                    seed=seed,
                    epoch=epoch,
                    num_shards=num_shards,
                    shard_id=shard_id,
            )
        else:
            return OpenWebLMLoader(
                    self.cfg,
                    dataset,
                    self.dictionary,
                    self.tokenizer,
                    max_tokens=max_tokens,
                    max_sentences=max_sentences,
                    max_positions=max_positions,
                    ignore_invalid_inputs=ignore_invalid_inputs,
                    required_batch_size_multiple=required_batch_size_multiple,
                    seed=seed,
                    epoch=epoch,
                    num_shards=num_shards,
                    shard_id=shard_id,
            )

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
