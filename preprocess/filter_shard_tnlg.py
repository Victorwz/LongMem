import json
import sys, os, re
import numpy as np
import random
from tqdm import tqdm
random.seed(2)

"""
Please modify the following paths to your local paths.
"""
project_path="./LongMem"
source_path="./data/pile/"
target_path="./data/longmem"

def json_write(dataset, sample_json_list):
    with open(target_path + "/train.txt", "a") as write_file:
        for i in tqdm(sample_json_list):
            
            # print(shard)
            with open("{}/{}/{}".format(source_path, dataset, shard), 'r') as json_file:
                lines = json_file.read().strip().split('\n')

            for json_str in lines:
                text = list(filter(None, json.loads(json_str)["text"].split("\n")))
                text.append("")
                write_file.write("\n".join(text) + "\n")

"""
The Pile dataset contains different sub-dataset. Each sub-dataset has been splited and constructed in jsonline split files.
This dictionary is generated from a weight matrix. The value refers to number of jsonline split files of this subset to be included in the large training set. 
"""
dataset = {"Gutenberg_PG-19_ftfy_cleaned_id_cleaned": 27290, "BookCorpus2_ftfy_cleaned_id": 1876, "Books3_ftfy_cleaned_id": 19401, "CC-2020-50_id_cleaned": 79296, "CC-2021-04_id_cleaned": 96130, "NIH_ExPorter_ftfy_id": 755, "OpenWebText2_ftfy_cleaned_id": 15674, "Pile-CC_id_cleaned": 50000, "rn_dedup_shuf_cleaned_0.7_cleaned": 28773, "stories_dedup0.7_shuf_cleaned":6702, "Wikipedia_en_ftfy_id": 5861}

total_jsonls = sum(dataset.values())
print(total_jsonls)
"""
fairseq will load all preprocessed data into memory. As we only iterates on 26B tokens, 50000 jsonline split files are enough to be used as the memory-augmented training set.
The sampled memory-augmented training set cannot be iterated for even 1 epoch.
"""
sampling_jsonls = 50000

sampled_shards_tnlg = []

for data, data_total_jsonls in dataset.items():
    print(data)
    num_jsonls_this_data = int(data_total_jsonls / total_jsonls * sampling_jsonls)
    print(num_jsonls_this_data)
    sample_json_list = random.sample(list(range(1, data_total_jsonls+1)), num_jsonls_this_data)
    for shard_index in sample_json_list:
        shard="split{}.jsonl".format(str(shard_index).zfill(8))
        shard_file_path = "{}/{}/{}".format(source_path, data, shard)
        sampled_shards_tnlg.append(shard_file_path)

random.shuffle(sampled_shards_tnlg)

with open(target_path + "/train.txt", "a") as write_file:
    for file_name in tqdm(sampled_shards_tnlg):
        
        # print(shard)
        with open(file_name, 'r') as json_file:
            lines = json_file.read().strip().split('\n')

        for json_str in lines:
            text = list(filter(None, json.loads(json_str)["text"].split("\n")))
            text.append("")
            write_file.write("\n".join(text) + "\n")

os.system("python fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
    --encoder-json {}/gpt2_bpe/encoder.json \
    --vocab-bpe {}/gpt2_bpe/vocab.bpe \
    --inputs {}/train.txt \
    --outputs {}/train.bpe \
    --keep-empty --workers 64".format(project_path, project_path, target_path, target_path))

# add bos token to each line. fairseq uses newline \n as the eos token.
os.system("sed -i \"s/^/<s> &/g\" {}/train.bpe".format(target_path))

os.system("fairseq-preprocess --only-source --trainpref {}/train.bpe --srcdict {}/gpt2_bpe/dict.txt --destdir {}/data/data-bin/longmem --workers 64".format(target_path, project_path, project_path))