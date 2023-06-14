# LongMem

Official implementation of our paper "[Augmenting Language Models with Long-Term Memory](https://arxiv.org/abs//2306.07174)". 

## Environment Setup 
* torch: Please follow [torch official installation guide](https://pytorch.org/get-started/previous-versions/). We recommend torch>=1.8.0. Please select the torch-gpu version which is consistent with your cuda driver version.

* Faiss-GPU: For Nvidia V100 GPUs, simply install via ``pip install faiss-gpu``. For Nvidia A100 GPUs, please run ``conda install faiss-gpu cudatoolkit=11.0 -c pytorch``. The A100 GPU is not officially supported by faiss-gpu, sometimes it will lead to errors, you can refer to this git [issue](https://github.com/facebookresearch/faiss/issues/2064) of faiss for help.

* fairseq: ``pip install --editable ./fairseq`` Then the revised `fairseq` and ohter packages will be installed. We strongly recommend you to use python 3.8 for stability.

## Project Strcture
Pre-trained LLM Class (L24, E1024, Alibi POS_ENCODING): ``fairseq/fairseq/models/newgpt.py``

Transformer Decoder with SideNetwork (L12, E1024, Alibi POS_ENCODING): ``fairseq/fairseq/models/sidenet/transformer_decoder_sidenet.py``

Transformer Language Model with SideNetwork Class: ``fairseq/fairseq/models/transformer_lm_sidenet.py``

Memory Bank and Retrieval: ``fairseq/fairseq/modules/dynamic_memory_with_chunk.py``

Joint Attention for Memory Fusion: ``fairseq/fairseq/modules/joint_multihead_attention_sum.py``

## Citation
Please cite our paper if you find this repository helpful in your research:
```
@article{LongMem,
  title={Augmenting Language Models with Long-Term Memory},
  author={Wang, Weizhi and Dong, Li and Cheng, Hao and Liu, Xiaodong and Yan, Xifeng and Gao, Jianfeng and Wei, Furu},
  journal={arXiv preprint arXiv:2306.07174},
  year={2023}
}
```
