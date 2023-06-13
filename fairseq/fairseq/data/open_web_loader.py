import glob
import os
import torch
import numpy as np
import time
import json
import random
import itertools
import hydra
import copy
from omegaconf import DictConfig, OmegaConf

from infinibatch import iterators
from fairseq.data.basic_loader import BaseBatchGen
from fairseq.data.utils import NativeCheckpointableIterator, WeightIterator


class OpenWebLMLoader(BaseBatchGen):

    def __init__(
            self,
            args,
            dataset,
            dictionary,
            tokenizer,
            max_tokens=None,
            max_sentences=None,
            max_positions=None,
            ignore_invalid_inputs=False,
            required_batch_size_multiple=1,
            seed=1,
            epoch=1,
            num_shards=1,
            shard_id=0,
    ):
        super().__init__()
        self.args = args
        self.data = dataset.data
        self.data_dir = dataset.data_dir
        self.shuffle = dataset.shuffle
        self.dictionary = dictionary
        self.tokenizer = tokenizer

        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.max_positions = max_positions
        self.tokens_per_sample = args.tokens_per_sample
        self.mlm_cut_length = args.mlm_cut_length
        self.mlm_tokens_proportion = args.mlm_tokens_proportion
        self.ignore_invalid_inputs = ignore_invalid_inputs
        self.required_batch_size_multiple = required_batch_size_multiple
        self.seed = str(seed)
        self.epoch = epoch
        self.num_shards = num_shards
        self.shard_id = shard_id

        self.batch_read_ahead = args.batch_read_ahead

        self._build_iter()
    
    def _build_iter(self):
        tokenized_lines = self._tokenize()
        self.padded_batches = self._batchify(tokenized_lines)
        
        prefetch_batches = iterators.PrefetchIterator(
            self.padded_batches, 
            buffer_size=10000, 
            buffer_in_main_process=True, 
            log_empty_buffer_warning=True and self.shard_id == 0,
        )

        prefetch_batches = iterators.MapIterator(
            prefetch_batches, self._move_to_tensor
        )

        self._iter = prefetch_batches

    def _tokenize(self):
        '''
        data:
        {
            'source': list[Path],
        }
        '''
        dataset = list(zip(self.data['source']))

        if self.shuffle:
            chunk_files = \
                iterators.InfinitePermutationSourceIterator(
                    dataset,
                    seed=self.seed, 
                    shuffle=self.shuffle, 
                    num_instances=self.num_shards, 
                    instance_rank=self.shard_id,
                )
        else:
            chunk_files = \
                iterators.ChunkedSourceIterator(
                    dataset,
                    num_instances=self.num_shards, 
                    instance_rank=self.shard_id,
                )
        
        tokenized_lines = iterators.SelectManyIterator(chunk_files, lambda files: self._read_from_files(*files))
        tokenized_lines = iterators.SamplingRandomMapIterator(tokenized_lines, self._prepare, self.seed)
        
        return tokenized_lines

    def getstate(self):
        state = super().getstate()
        state["epoch"] = self.epoch
        state["iterations_in_epoch"] = None
        return state

    def _batchify(self, lines):
        
        if self.max_sentences is not None:
            if self.batch_read_ahead > 0:
                lines = iterators.BlockwiseShuffleIterator(lines, self.batch_read_ahead, self.seed)
            batches = iterators.FixedBatchIterator(lines, self.max_sentences)
        else:
            # -
            def dynamic_batch_size(sample):
                lengths = [len(x) for x in sample]
                batch_size = self.max_tokens // max(lengths) // self.required_batch_size_multiple * self.required_batch_size_multiple
                return max(1, batch_size)
            
            batches = iterators.BucketedReadaheadBatchIterator(
                    lines,
                    read_ahead=self.batch_read_ahead, 
                    key=(lambda x: max(len(x[0]), len(x[1]))) if self.shuffle else None, 
                    batch_size=dynamic_batch_size, 
                    shuffle=self.shuffle,
                    seed=self.seed,
            )

        def collate(batch):
            batch = torch.LongTensor(batch)
            ret_batch = {
                'net_input': {
                    'src_tokens': batch[:, :-1]
                },
                'target': batch[:, 1:],
                'nsentences': len(batch),
                'ntokens': sum([len(x) - 1 for x in batch])
            }
            return ret_batch

        padded_batches = iterators.MapIterator(
            batches, collate
        )

        return padded_batches

    def _prepare(self, _random, doc):
        return doc

    def _read_from_files(self, source_file):
        file_path = os.path.join(self.data_dir, source_file)
        
        if not os.path.exists(file_path):
            print('| file {} not exists'.format(file_path), flush=True)
            return iter([]) # skip bad file

        with open(file_path, 'r', encoding='utf8') as f:
            lines = f.read().strip().split('\n')

        gpt_format_text = []
        for line in lines:
            gpt_format_text.extend(list(filter(None, json.loads(line)["text"].split("\n"))))
            gpt_format_text.append('')

        tokens_per_sample = self.tokens_per_sample + 1
        tokenized_lines = [self.tokenizer.encode(line) for line in gpt_format_text]
        tokenized_ids = [[self.dictionary.bos()] + self.dictionary.encode_line(line, add_if_not_exist=False).tolist() for line in tokenized_lines]
        flatten_tokenized_ids = list(itertools.chain(*tokenized_ids))
        total_length = (len(flatten_tokenized_ids) // tokens_per_sample) * tokens_per_sample
        data = [flatten_tokenized_ids[i : i + tokens_per_sample] for i in range(0, total_length, tokens_per_sample)]
        return data