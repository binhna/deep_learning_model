for VARIABLE in 1 2 3 4 5
do
	CUDA_VISIBLE_DEVICES=7 python train_ner.py \
    --model-path ../shared_data/BDIRoBerta \
    --batch-size 32 \
    --format conll \
    --train-path data/entity_dialogue/entity_dialogue.txt \
    --valid-path data/entity_dialogue/entity_dialogue_valid.txt \
    --epoch 3 \
    --do-eval \
    --do-train \
    --output-dir "model/entity_dialogue_$VARIABLE"
done