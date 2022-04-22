import os
import numpy as np
from seqeval.metrics import f1_score, classification_report
from argparse import ArgumentParser
from transformers import AutoConfig, AutoTokenizer, TrainingArguments

from src.data_utils import NERDataset
from src.model import RobertaNER
from src.trainer_utils import CustomTrainer

parser = ArgumentParser()

parser.add_argument(
    "--model-path", required=True, type=str, help="model pretrained path"
)
parser.add_argument(
    "--train-path", required=True, type=str, help="path of training data"
)
parser.add_argument("--freeze_layer_count", type=int, default=-1, help="freeze layer")
parser.add_argument("--valid-path", required=True, type=str, help="path of valid data")
parser.add_argument("--test-path", type=str, default=None, help="path of test data")
parser.add_argument("--batch-size", type=int, default=32, help="batch size")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--use-crf", action="store_true", help="whether to use crf layer")
parser.add_argument("--use-adapter", action="store_true", help="whether to use adapter")
parser.add_argument("--epoch", type=int, default=5, help="number of epoch")
parser.add_argument("--do-train", action="store_true")
parser.add_argument("--do-eval", action="store_true")
parser.add_argument(
    "--from-checkpoint", type=str, default="", help="model path for evaluation"
)
parser.add_argument(
    "--output-dir", type=str, default="model", help="output dir for the model"
)
parser.add_argument(
    "--format",
    type=str,
    default="conll",
    choices=["conll", "flatten"],
    help="format of the training data",
)
parser.add_argument(
    "--augment_lower",
    action="store_true",
    help="whether to augment data with lowercase",
)
parser.add_argument(
    "--task-name",
    type=str,
    default="ner",
    help="task name for the adapter",
)
parser.add_argument("--lower", action="store_true", help="lowercase the training data")


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    preds = preds.argmax(-1)

    mapping = lambda i: id2label[i]
    v_func = np.vectorize(mapping)
    labels = v_func(labels).tolist()
    preds = v_func(preds).tolist()
    for i, (g_tags, p_tags) in enumerate(zip(labels, preds)):

        rm_idx = np.where(np.array(g_tags) == "X")
        labels[i] = np.delete(np.array(g_tags), rm_idx).tolist()
        preds[i] = np.delete(np.array(p_tags), rm_idx).tolist()

    # exact match
    em = [np.array_equal(pred_, label_) for pred_, label_ in zip(preds, labels)]
    em = sum(em) / len(em)

    f1 = f1_score(labels, preds, average="macro")
    print(classification_report(labels, preds))
    return {"f1": f1, "em": em}


if __name__ == "__main__":
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, add_prefix_space=True)
    config = AutoConfig.from_pretrained(args.model_path)
    config.update(vars(args))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_train:
        config.id2label = None
        config.label2id = None
        
        train_data = NERDataset(args.train_path, tokenizer=tokenizer, config=config)
        # train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        config.num_labels = train_data.get_num_labels()
        config.id2label = train_data.id2label
        config.label2id = train_data.label2id
    if args.do_eval:
        valid_data = NERDataset(args.valid_path, tokenizer=tokenizer, config=config, split="valid")

    # valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    if args.do_train and args.do_eval:
        assert valid_data.id2label == train_data.id2label
    print(config.id2label)

    step_eval = int(0.2*len(train_data)//config.batch_size)
    # Cứ theo config này thì model sẽ auto lưu lại best model theo điểm f1 (nhớ phải define hàm compute_metrics để output ra f1)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        learning_rate=args.lr,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        eval_steps=step_eval,
        save_steps=step_eval,
        save_strategy="steps",
        metric_for_best_model="f1",
        push_to_hub=False,
        overwrite_output_dir=True,
        save_total_limit=1,
        load_best_model_at_end=True,
    )

    id2label = config.id2label

    model = RobertaNER(config=config)
    model = model.from_pretrained(args.model_path, config=config)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data if args.do_train else None,
        eval_dataset=valid_data if args.do_eval else None,
        compute_metrics=compute_metrics,
    )
    if args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=args.from_checkpoint
            if args.from_checkpoint
            else False
        )
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload
        tokenizer.save_pretrained(args.output_dir)
        trainer.save_metrics("train", metrics)
    if args.do_eval:
        metrics = trainer.evaluate()
        trainer.save_metrics("eval", metrics)
