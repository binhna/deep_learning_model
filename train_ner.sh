CUDA_VISIBLE_DEVICES=0 python train_ner.py \
    --format csv \
    --batch-size 128 \
    --train-path data/sentiment_for_financial_news/train.csv \
    --valid-path data/sentiment_for_financial_news/valid.csv \
    --epoch 100 \
    --do-eval \
    --do-train \
    --task text_classification \
    --metric_for_best f1 \
    --output-dir model/visemhash