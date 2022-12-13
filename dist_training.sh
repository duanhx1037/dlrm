python -m torch.distributed.launch --nproc_per_node=8 dlrm_s_pytorch.py \
    --arch-embedding-size="80000-80000-80000-80000-80000-80000-80000-80000" \
    --arch-sparse-feature-size=64 \
    --arch-mlp-bot="128-128-128-64" \
    --arch-mlp-top="512-512-512-256-1" \
    --max-ind-range=40000000 \
    --data-generation=random \
    --loss-function=bce \
    --round-targets=True \
    --learning-rate=1.0 \
    --mini-batch-size=2048 \
    --print-freq=2 \
    --print-time \
    --test-freq=2 \
    --test-mini-batch-size=2048 \
    --memory-map \
    --use-gpu \
    --num-batches=2 \
    --dist-backend=nccl \

rm -rf torchelastic_*