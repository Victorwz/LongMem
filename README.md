# NeurIPS 2023 Submission Language Models Augmented with Decoupled Memory

## Project Strcture
Pre-trained LLM Class (L24, E1024, Alibi POS_ENCODING): ``fairseq/fairseq/models/newgpt.py``

Transformer Decoder with SideNetwork (L12, E1024, Alibi POS_ENCODING): ``fairseq/fairseq/models/sidenet/transformer_decoder_sidenet.py``

Transformer Language Model with SideNetwork Class: ``fairseq/fairseq/models/transformer_lm_sidenet.py``

Memory Bank and Retrieval: ``fairseq/fairseq/modules/dynamic_memory_with_chunk.py``

Joint Attention for Memory Fusion: ``fairseq/fairseq/modules/joint_multihead_attention_sum.py``