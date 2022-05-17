import os
import numpy as np
from seqeval.metrics import f1_score, classification_report
from argparse import ArgumentParser
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
from torch.nn.utils.rnn import pad_sequence
import torch

from src.data_utils import ConVExDataset
from src.model import ConVEx
from src.trainer_utils import CustomTrainer

parser = ArgumentParser()

parser.add_argument(
    "--model-path", required=True, type=str, help="model pretrained path"
)
parser.add_argument("--train-path", type=str, help="path of training data")
parser.add_argument("--freeze_layer_count", type=int, default=-1, help="freeze layer")
parser.add_argument("--valid-path", type=str, help="path of valid data")
parser.add_argument("--test-path", type=str, default=None, help="path of test data")
parser.add_argument("--batch-size", type=int, default=32, help="batch size")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
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
    "--augment_lower",
    action="store_true",
    help="whether to augment data with lowercase",
)

parser.add_argument(
    "--finetuning",
    action="store_true",
    help="whether to pretraining or finetuning",
)

parser.add_argument(
    "--task-name",
    type=str,
    default="ner",
    help="task name for the adapter",
)
parser.add_argument("--lower", action="store_true", help="lowercase the training data")
parser.add_argument("--template_fnn_size", type=int, default=128)
parser.add_argument("--input_fnn_size", type=int, default=128)
parser.add_argument("--num_heads", type=int, default=1)


def custom_collator(batch):
    input_ids = [sample["input_ids"] for sample in batch]
    labels = [sample["labels"] for sample in batch]
    input_attention_mask = [sample["input_attention_mask"] for sample in batch]

    template_ids = [sample["template_ids"] for sample in batch]
    template_attention_mask = [sample["template_attention_mask"] for sample in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    input_attention_mask = pad_sequence(
        input_attention_mask, batch_first=True, padding_value=0
    )
    template_ids = pad_sequence(template_ids, batch_first=True, padding_value=0)
    template_attention_mask = pad_sequence(
        template_attention_mask, batch_first=True, padding_value=0
    )

    return {
        "template_ids": template_ids,
        "template_attention_mask": template_attention_mask,
        "input_ids": input_ids,
        "input_attention_mask": input_attention_mask,
        "labels": labels,
    }


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions

    mapping = lambda i: id2label.get(i, "O")
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

    print(classification_report(labels, preds))
    f1 = f1_score(labels, preds, average="micro")
    return {"f1": f1, "em": em}


if __name__ == "__main__":
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, add_prefix_space=True)
    config = AutoConfig.from_pretrained(args.model_path)
    config.update(vars(args))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    config.id2label, config.label2id = {0: "O", 1: "B-PHRASE", 2: "I-PHRASE"}, {
        "O": 0,
        "B-PHRASE": 1,
        "I-PHRASE": 2,
    }
    train_data = ConVExDataset(args.train_path, tokenizer=tokenizer, config=config)
    config.num_labels = train_data.get_num_labels()
    # train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_data = ConVExDataset(args.valid_path, tokenizer=tokenizer, config=config)
    assert (
        train_data.id2label == valid_data.id2label
        and train_data.label2id == valid_data.label2id
    )
    # print(train_data.id2label)
    # valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)

    step_eval = int(0.25 * len(train_data) // config.batch_size)
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
        logging_steps=int(0.05 * len(train_data) // config.batch_size),
        push_to_hub=False,
        overwrite_output_dir=True,
        save_total_limit=1,
        load_best_model_at_end=True,
    )

    id2label = config.id2label

    model = ConVEx(config=config)
    model = model.from_pretrained(args.model_path, config=config)

    optimizer = torch.optim.Adadelta(
        params=model.parameters(), lr=config.lr, weight_decay=0.9
    )
    total_training_step = config.epoch * len(train_data) / config.batch_size
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(total_training_step*0.05),
        num_training_steps=total_training_step,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        compute_metrics=compute_metrics,
        data_collator=custom_collator,
        # optimizers=(optimizer, scheduler),
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
