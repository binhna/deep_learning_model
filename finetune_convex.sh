CUDA_VISIBLE_DEVICES=5 python finetune_convex.py \
    --model-path model/convex2/checkpoint-15741 \
    --batch-size 64 \
    --train-path data/merge_vi_entity_vinbdi/train.txt \
    --valid-path data/merge_vi_entity_vinbdi/valid.txt \
    --finetuning \
    --epoch 10 \
    --do-eval \
    --do-train \
    --output-dir model/convex2_tuning_entity_merge