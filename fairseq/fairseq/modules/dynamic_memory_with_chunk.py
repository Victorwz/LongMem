#!/usr/bin/env/python

import torch
import faiss
import numpy as np
import os

import time
import math
import faiss.contrib.torch_utils

import torch.nn.functional as F


class External_Memory:

    def __init__(self, cfg):
        self.dimension = cfg.decoder.embed_dim
        self.use_gpu_to_search = cfg.use_gpu_to_search
        self.k = cfg.k
        self.reduce_fac = cfg.layer_reduction_factor
        
        self.memory_size = cfg.memory_size
        self.num_heads = cfg.decoder.attention_heads
        self.head_dim = int(self.dimension / self.num_heads)
        self.chunk_size = getattr(cfg, "chunk_size", 4)
        print("chunk size", self.chunk_size)

        if self.use_gpu_to_search:
            self.index_list = []
            print('put index from cpu to gpu {}'.format(torch.cuda.current_device()))

            self.res = faiss.StandardGpuResources()
            for i in range(self.num_heads):
                gpu_index = faiss.IndexFlatIP(self.head_dim)
                gpu_index = faiss.index_cpu_to_gpu(self.res, torch.cuda.current_device(), gpu_index)
                self.index_list.append(gpu_index)
            print("put done")
            self.keys = [torch.zeros((self.memory_size//self.chunk_size), self.chunk_size, self.head_dim, dtype=torch.float16, device=torch.cuda.current_device()) for i in range(self.num_heads)]
            self.vals = [torch.zeros((self.memory_size//self.chunk_size), self.chunk_size, self.head_dim, dtype=torch.float16, device=torch.cuda.current_device()) for i in range(self.num_heads)]
        else:
            self.index_list = [faiss.IndexFlatIP(self.head_dim) for i in range(self.num_heads)]
            # self.index = faiss.IndexHNSWFlat(self.dimension, 15, faiss.METRIC_INNER_PRODUCT)
            self.keys = [torch.zeros(self.memory_size, self.head_dim) for i in range(self.num_heads)]
            self.vals = [torch.zeros(self.memory_size, self.head_dim) for i in range(self.num_heads)]
        
        self.time_for_retrieve = 0.
        self.retrieve_count = 0.
        self.time_for_setup_prob = 0.

        self.dstore_idx = 0

    def setup_faiss(self, args):
        try:
            start_gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES").split(",")[0])
        except AttributeError:
            start_gpu_id = 0
        # import pynvml
        # pynvml.nvmlInit()
        # handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device()+start_gpu_id)
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print("CUDA:{} has {:.2f}GB free GPU memory.".format(torch.cuda.current_device(), meminfo.free/2**30))
        index = faiss.index_cpu_to_gpu(res, torch.cuda.current_device(), index, co)
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print("CUDA:{} has {:.2f}GB free GPU memory after loading knn index into GPU.".format(torch.cuda.current_device(), meminfo.free/2**30))
        if faiss.get_num_gpus() == 1:
            co = faiss.GpuClonerOptions()
            # co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(res, 0, index, co)
        elif faiss.get_num_gpus() > 1:
            ngpu = faiss.get_num_gpus()
            print('running on %d GPUs' % ngpu)
            co = faiss.GpuMultipleClonerOptions()
            # co.useFloat16 = True
            index = faiss.index_cpu_to_all_gpus(index, co=co, ngpu=ngpu)
        else:
            raise RuntimeError("No available GPU for faiss index")
    
    def reset(self):
        self.dstore_idx = 0
        for index in self.index_list:
            index.reset()
        # self.index = faiss.IndexHNSWFlat(self.dimension, 15, faiss.METRIC_INNER_PRODUCT)
        
        self.time_for_retrieve = 0.
        self.retrieve_count = 0.
        self.time_for_setup_prob = 0.

        if self.use_gpu_to_search:
            self.keys = [torch.zeros((self.memory_size//self.chunk_size), self.chunk_size, self.head_dim, dtype=torch.float16, device=torch.cuda.current_device()) for i in range(self.num_heads)]
            self.vals = [torch.zeros((self.memory_size//self.chunk_size), self.chunk_size, self.head_dim, dtype=torch.float16, device=torch.cuda.current_device()) for i in range(self.num_heads)]

    def add_index(self, qkv_val, retrieval_layer_index=None, padding_mask=None):
        keys, vals = qkv_val['k'], qkv_val['v']
        bsz, seq_len = keys.shape[:2]
        
        if self.dstore_idx + (bsz*seq_len//self.chunk_size) >= (self.memory_size//self.chunk_size):
            # update_size = int(self.memory_size / 2)
            update_size = (2*bsz*seq_len)//self.chunk_size
            self.dstore_idx = self.dstore_idx - update_size
            if self.use_gpu_to_search:
                for i, index in enumerate(self.index_list):
                    
                    temp = faiss.index_gpu_to_cpu(index)
                    remove_n_total = temp.remove_ids(np.arange(update_size))
                    # print("Removing {} keys from index".format(remove_n_total))
                    
                    new_gpu_index = faiss.index_cpu_to_gpu(self.res, torch.cuda.current_device(), temp)
                    self.index_list[i] = new_gpu_index
                
            else:
                for index in self.index_list:
                    index.remove_ids(np.arange(update_size))
                
            self.keys = [torch.cat((self.keys[i][update_size:, ...], torch.zeros(update_size, self.chunk_size, self.head_dim, dtype=torch.float16, device=torch.cuda.current_device()))) for i in range(self.num_heads)]
            self.vals = [torch.cat((self.vals[i][update_size:, ...], torch.zeros(update_size, self.chunk_size, self.head_dim, dtype=torch.float16, device=torch.cuda.current_device()))) for i in range(self.num_heads)]

        keys = keys.view(bsz*seq_len, self.num_heads, self.head_dim)
        # features = features[padding_mask]
        vals = vals.view(bsz*seq_len, self.num_heads, self.head_dim)

        keep_dim = (bsz*seq_len)//self.chunk_size*self.chunk_size
        keys_with_chunk = keys[:keep_dim, ...].contiguous().view(keep_dim//self.chunk_size, self.chunk_size, self.num_heads, self.head_dim)
        vals_with_chunk = vals[:keep_dim, ...].contiguous().view(keep_dim//self.chunk_size, self.chunk_size, self.num_heads, self.head_dim)

        # print(keys_with_chunk.shape)
        for i, index in enumerate(self.index_list):
            index.add(keys_with_chunk[:, :, i, :].mean(dim=-2).type(torch.float32).contiguous())
            self.keys[i][self.dstore_idx:keys_with_chunk.shape[0]+self.dstore_idx, ...] = keys_with_chunk[:, :, i, :]
            self.vals[i][self.dstore_idx:keys_with_chunk.shape[0]+self.dstore_idx, ...] = vals_with_chunk[:, :, i, :]

        self.dstore_idx += keys_with_chunk.shape[0]

    def retrieve(self, queries):

        seq_len, bsz, hid_size = queries.shape
        queries = queries.view(seq_len*bsz, self.num_heads, self.head_dim).type(torch.float32)

        indexs = [self.index_list[i].search(queries[:, i, :].contiguous(), (self.k)//self.chunk_size)[1] for i in range(self.num_heads)]

        keys_tgt_index = [self.keys[i][indexs[i]].view(seq_len*bsz, self.k, self.head_dim) for i in range(self.num_heads)]
        vals_tgt_index = [self.vals[i][indexs[i]].view(seq_len*bsz, self.k, self.head_dim) for i in range(self.num_heads)] 

        keys_tgt_index = torch.stack(keys_tgt_index, dim=1).view(seq_len, bsz*self.num_heads, self.k, self.head_dim).transpose(0, 1)
        vals_tgt_index = torch.stack(vals_tgt_index, dim=1).view(seq_len, bsz*self.num_heads, self.k, self.head_dim).transpose(0, 1)

        return {'knn_index': indexs, 'tgt_index': {"k": keys_tgt_index, "v": vals_tgt_index}}