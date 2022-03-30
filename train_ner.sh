CUDA_VISIBLE_DEVICES=0 python train_ner.py \
    --model-path ../../shared_data/roberta-2L-vi \
    --format flatten \
    --train-path data/company_stock/train.src \
    --valid-path data/company_stock/valid.src \
    --freeze_layer_count 0 \
    --epoch 10 \
    --do-eval \
    --do-train \
    --output-dir company_stock_model