task=pile
DATA_DIR=/path/to/pile_preprocessed_binary
CKPT_DIR=models/longmem_gpt2_sidenet
PTM_PATH=/path/to/pretrained_gpt2_model

fairseq-train ${DATA_DIR}  \
    --save-dir ${CKPT_DIR} \
    --task language_modeling --arch transformer_lm_sidenet_gpt2_small \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 \
    --lr 2e-4 --lr-scheduler polynomial_decay \
    --weight-decay 0.01 \
    --save-interval-updates 10000 --sample-break-mode none \
    --tokens-per-sample 1024 \
    --batch-size 8 --total-num-update 100000 --seed 42 \
    --pretrained-model-path bigscience/bloom-1b7 \
    --layer-reduction-factor 2 \
    --disable-validation \
    --use-external-memory --memory-size 65536 \
    --k 64 --chunk-size 4 \
    --fp16 \
    --use-gpu-to-search \
    --no-token-positional-embeddings \
    --data-no-shuffle \
    --retrieval-layer-index 17 \
    --reload-ptm-layer