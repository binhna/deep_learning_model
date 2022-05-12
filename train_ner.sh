CUDA_VISIBLE_DEVICES=0 python train_ner.py \
    --model-path ../../shared_data/bdi_roberta_4L_oscarwiki_envi \
    --batch-size 16 \
    --train-path data/text_corpus/training_data_convex.txt \
    --valid-path data/text_corpus/validation_data_convex.txt \
    --freeze_layer_count 0 \
    --epoch 10 \
    --do-eval \
    --do-train \
    --output-dir model/convex