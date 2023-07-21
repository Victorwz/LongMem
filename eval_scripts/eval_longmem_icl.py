import torch
from fairseq.checkpoint_utils import load_model_ensemble
from fairseq.data import Dictionary
import os, sys
import argparse
import json
import faiss
from tqdm import tqdm
import random
from argparse import Namespace
from fairseq.data.encoders.gpt2_bpe import GPT2BPE
import numpy as np
import itertools
from math import *


DEFAULT_ENCODER_JSON = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json"
DEFAULT_VOCAB_BPE = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe"


def load_data(task):
    data = {}
    path = os.path.join("data/icl/original/", task)
    if task == "SST-2":
        for split in ["train", "validation",]:
            data[split] = []
            file_split = "dev" if split == "validation" else split
            
            with open(os.path.join(path, "{}.tsv".format(file_split)), "r") as f:
                for line in tqdm(f.readlines()):
                    review, label = line.strip("\n").split("\t")
                    if label == "label":
                        continue
                    label = "positive" if int(label) == 1 else "negative"
                    data[split].append([review, label])
    
    if task == "sst-5":
        mapping = {0: "terrible", 1: "bad", 2: "okay", 3: "good", 4: "great"}
        for split in ["train", "validation", "test"]:
            file_split = "dev" if split == "validation" else split
            if os.path.exists(os.path.join(path, "stsa.fine.{}".format(file_split))):
                data[split] = []
                with open(os.path.join(path, "stsa.fine.{}".format(file_split)), "r") as f:
                    for line in tqdm(f.readlines()):
                        review, label = line[2:], line[0]                        
                        label = mapping[int(label)]
                        data[split].append([review.strip("\n"), label])

    elif task in ["mr", "mpqa"]:
        for split in ["train", "validation", "test"]:
            if os.path.exists(os.path.join(path, "{}.csv".format(split))):
                data[split] = []
                with open(os.path.join(path, "{}.csv".format(split)), "r") as f:
                    for line in tqdm(f.readlines()):
                        review, label = line[2:], line[0]
                        label = "positive" if int(label) == 1 else "negative"
                        data[split].append([review.strip("\n"), label])
        print(data['test'][0])

    elif task in ["subj"]:
        for split in ["train", "test"]:
            if os.path.exists(os.path.join(path, "{}.csv".format(split))):
                data[split] = []
                with open(os.path.join(path, "{}.csv".format(split)), "r") as f:
                    for line in tqdm(f.readlines()):
                        review, label = line[2:], line[0]
                        label = "subjective" if int(label) == 0 else "objective"
                        data[split].append([review.strip("\n").strip('"'), label])
        print(data['test'][0])
    
    return data


def main(args):
    if "longmem_gpt2_medium" in args.path:
        override_args = {"pretrained_model_path": args.pretrained_model_path, "gpt_encoder_path": args.gpt_encoder_path, "data": "gpt2_bpe", "chunk_size": 2}
    else:
        override_args = {"gpt2_vocab_bpe": os.path.join(args.gpt_encoder_path, "vocab.bpe"), "gpt2_encoder_json": os.path.join(args.gpt_encoder_path, "encoder.json"), "gpt_dict_path": os.path.join(args.gpt_encoder_path, "dict.txt")}


    task_template_dict = {"SST-2": "Review: {} Sentiment: {}. ",
                          "sst-5": "Review: {} Sentiment: {}. ",
                            "mr": "Review: {} Sentiment: {}. ",
                          "mpqa": "Review: {} Sentiment: {}. ",
                          "subj": "Input: {} Type: {}. "}

    dictionary = Dictionary.load(os.path.join(args.gpt_encoder_path, "dict.txt"))
    tokenizer =  GPT2BPE(Namespace(gpt2_vocab_bpe=DEFAULT_VOCAB_BPE, gpt2_encoder_json=DEFAULT_ENCODER_JSON))

    data = load_data(args.task)
    if args.task in ['SST-2', "sst-5"]:
        args.subset = "validation"

    task_template = task_template_dict[args.task]

    context_length = 1024
    model_list_acc = []

    model, _ = load_model_ensemble([args.path], arg_overrides=override_args, task=None)
    model = model[0]
    model = model.eval()
    model = model.cuda()
    
    for seed in [1,2,3,4,5,6]:
        random.seed(seed)
        original_demon_train_subset = random.sample(data['train'], args.k)
        original_demon_train_subset = [task_template.format(s[0], s[1]) for s in original_demon_train_subset]
        demonstration = "".join(original_demon_train_subset)

        if "longmem_gpt2_medium" in args.path:
            print("Load {} examples into memory".format(args.cache_k))
            memory_set = [task_template.format(s[0], s[1]) for idx, s in enumerate(random.sample(data['train'], args.cache_k))]

            tokenized_lines = [tokenizer.encode(line) for line in memory_set]
            tokenized_ids = [[dictionary.bos()] + dictionary.encode_line(line, add_if_not_exist=False).tolist() for line in tokenized_lines]
            article_tokens = list(itertools.chain(*tokenized_ids))
            article_list = [article_tokens[i*context_length:(i+1)*context_length] for i in range(ceil(len(article_tokens)//context_length))]
            for t in article_list:
                model(torch.LongTensor([t]).cuda())
            print(model.decoder.external_memory.index_list[0].ntotal)
        
        total_cnt = 0
        acc_cnt = 0
        
        for item in data[args.subset]:
            total_cnt += 1
            
            test_subset = original_demon_train_subset + [task_template[:-5].format(item[0])]
            tokenized_lines = [tokenizer.encode(line) for line in test_subset]
            tokenized_ids = [[dictionary.bos()] + dictionary.encode_line(line, add_if_not_exist=False).tolist() for line in tokenized_lines]
            tokens = list(itertools.chain(*tokenized_ids))

            tokens = torch.LongTensor([tokens[:-1]]).cuda()

            if "longmem_gpt2_medium" in args.path:
                prediction = model(tokens, features_only=False, disable_add_index=True)
            else:
                prediction = model(tokens, features_only=False)
            
            prediction = prediction[0][0, -1, :].softmax(dim=-1)
            
            prediction = tokenizer.decode(dictionary.string([prediction.argmax(-1).item()]))
            acc_cnt += (item[1].startswith(prediction.strip()) and prediction.strip() != "")
        
        model_list_acc.append(acc_cnt / total_cnt)

        try:
            if model.decoder.external_memory:
                model.decoder.external_memory.reset()
        except AttributeError:
            pass

        print("Acc for random seed {}: {}".format(seed, acc_cnt / total_cnt))
    model_list_acc = [np.mean(model_list_acc), np.std(model_list_acc)] 

    print("Mean acc across 6 seeds: {:.4f}, std: {:.4f}".format(model_list_acc[0], model_list_acc[1]))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Arguments for evaluating GPT2 LongMem Model")
    parser.add_argument("--path", type=str, default="checkpoints/longmem_gpt2_medium/checkpoint_last.pt", help="The path to the model checkpoint")
    parser.add_argument("--pretrained-model-path", type=str, default="checkpoints/gpt2_medium/checkpoint_last.pt", help="The path to the data")
    parser.add_argument("--task", type=str, default="SST-2", help="The evaluated task for in-context learning")
    parser.add_argument("--gpt-encoder-path", type=str, default="gpt2_bpe", help="The path to the gpt2 encoder and dictionary")
    parser.add_argument("--k", type=int, default=20, help="number of demonstration examples in in-context learning")
    parser.add_argument("--cache-k", type=int, default=2000, help="number of cached examples in LongMem's memory")
    parser.add_argument("--subset", type=str, default="test", help="normally test set. But for SST-2, there is no testset, we use validation set instead")
    args = parser.parse_args()
    main(args)