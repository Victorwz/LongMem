# LongMem

Official implementation of our paper "[Augmenting Language Models with Long-Term Memory](https://arxiv.org/abs//2306.07174)".

Please cite our paper if you find this repository interesting or helpful:
```bibtex
@article{LongMem,
  title={Augmenting Language Models with Long-Term Memory},
  author={Wang, Weizhi and Dong, Li and Cheng, Hao and Liu, Xiaodong and Yan, Xifeng and Gao, Jianfeng and Wei, Furu},
  journal={arXiv preprint arXiv:2306.07174},
  year={2023}
}
```

## Environment Setup 
* torch: Please follow [torch official installation guide](https://pytorch.org/get-started/previous-versions/). We recommend torch>=1.8.0. Please select the torch-gpu version which is consistent with your cuda driver version.

* Faiss-GPU: For Nvidia V100 GPUs, simply install via ``pip install faiss-gpu``. For Nvidia A100, A6000 GPUs, please run ``conda install faiss-gpu cudatoolkit=11.0 -c pytorch``. The A100 GPU is not officially supported by faiss-gpu, sometimes it will lead to errors, you can refer to this git [issue](https://github.com/facebookresearch/faiss/issues/2064) of faiss for help.

* fairseq: ``pip install --editable ./fairseq`` Then the revised `fairseq` and dependency packages will be installed. We strongly recommend you to use python 3.8 for stability.

* other packages: ``pip install -r requirements.txt``

## Project Structure
* Pre-trained LLM Class (L24, E1024, Alibi positional embedding): [`fairseq/fairseq/models/newgpt.py`](fairseq/fairseq/models/newgpt.py)

* Transformer Decoder with SideNetwork (L12, E1024): [`fairseq/fairseq/models/sidenet/transformer_decoder_sidenet.py`](fairseq/fairseq/models/sidenet/transformer_decoder_sidenet.py)

* Transformer Language Model with SideNetwork Class: [`fairseq/fairseq/models/transformer_lm_sidenet.py`](fairseq/fairseq/models/transformer_lm_sidenet.py)

* Memory Bank and Retrieval: [`fairseq/fairseq/modules/dynamic_memory_with_chunk.py`](fairseq/fairseq/modules/dynamic_memory_with_chunk.py)

* Joint Attention for Memory Fusion: [`fairseq/fairseq/modules/joint_multihead_attention_sum.py`](fairseq/fairseq/modules/joint_multihead_attention_sum.py)

## Memory-Augmented Adaptation Training
### Data collection and Preprocessing
Please download the Pile from [official release](https://pile.eleuther.ai/). Each sub-dataset in the Pile is organized as various jsonline splits. You can refer to [`preprocess/filter_shard_tnlg.py`](preprocess/filter_shard_tnlg.py) fpr how we sample the training set and binalize following standard fairseq preprocessing process.

Memory-Augmented Adaptation Training:
```
bash train_scripts/train_longmem.sh
```

## Evaluation
Please firstly download the checkpoints for pre-trained [GPT2-medium model and LongMem model](https://drive.google.com/file/d/1ZTNN8r5X2dkQMRuckW6J08sVDOMdzbzA/view?usp=sharing) to ``checkpoints/``.

### Memory-Augmented In-Context Learning
```
# Evaluate gpt2 baseline
python eval_scripts/eval_longmem_icl.py --path /path/to/gpt2_pretrained_model
# Evaluate LongMem model
python eval_scripts/eval_longmem_icl.py --path /path/to/longmem_model --pretrained-model-path /path/to/gpt2_pretrained_model
```

## Credits
LongMem is developed based on [fairseq](https://github.com/facebookresearch/fairseq). Thanks to the team from eleuther.ai who constructed the largest high-quality corpora, the Pile.
