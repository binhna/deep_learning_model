CUDA_VISIBLE_DEVICES=0 python train_ner.py \
    --model-path ../../shared_data/roberta-2L-vi \
    --format conll \
    --train-path data/entity_dialogue/entity_dialogue_train.txt \
    --valid-path data/entity_dialogue/entity_dialogue_valid.txt \
    --freeze_layer_count 0 \
    --epoch 10 \
    --do-eval \
    --do-train \
    --output-dir company_stock_model