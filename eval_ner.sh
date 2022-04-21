CUDA_VISIBLE_DEVICES=1 python train_ner.py \
    --batch-size 96 \
    --model-path model/mSystemEntity \
    --format conll \
    --train-path data/mSystemEntity/valid \
    --valid-path data/mSystemEntity/valid/vi.txt \
    --freeze_layer_count 0 \
    --epoch 5 \
    --do-eval
