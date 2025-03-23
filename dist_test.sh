CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 val.py \
    --config configs/racformer_r50_nuimg_704x256_f8.py --weights checkpoints/racformer_r50_f8.pth