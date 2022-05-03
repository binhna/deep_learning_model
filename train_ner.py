import os
import numpy as np
from argparse import ArgumentParser
import torch

from src.data_utils import NERDataset
from src.model import NERModel
from src.trainer_utils import train, evaluate
from src.tokenizer import Config, SemHashTokenizer

parser = ArgumentParser()

parser.add_argument(
    "--train-path", required=True, type=str, help="path of training data"
)
parser.add_argument("--valid-path", required=True, type=str, help="path of valid data")
parser.add_argument("--test-path", type=str, default=None, help="path of test data")
parser.add_argument("--batch-size", type=int, default=32, help="batch size")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--use-crf", action="store_true", help="whether to use crf layer")
parser.add_argument("--epoch", type=int, default=5, help="number of epoch")
parser.add_argument("--do-train", action="store_true")
parser.add_argument("--do-eval", action="store_true")
parser.add_argument(
    "--output-dir", type=str, default="model", help="output dir for the model"
)
parser.add_argument(
    "--format",
    type=str,
    default="conll",
    choices=["conll", "flatten", "csv"],
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
parser.add_argument(
    "--tokenizer-config",
    type=str,
    default="./tokenizer_config.json",
    help="task name for the adapter",
)
parser.add_argument(
    "--model-config",
    type=str,
    default="./model_config.json",
    help="task name for the adapter",
)
parser.add_argument(
    "--accu_step",
    type=int,
    default=1,
    help="task name for the adapter",
)
parser.add_argument(
    "--metric_for_best",
    type=str,
    default="f1",
    help="metric for choosing best model"
)
parser.add_argument(
    "--task",
    type=str,
    default="seq_tagging",
    help="task"
)
parser.add_argument("--lower", action="store_true", help="lowercase the training data")


if __name__ == "__main__":
    args = parser.parse_args()

    tokenizer_config = Config.from_pretrained(args.tokenizer_config)
    tokenizer = SemHashTokenizer(tokenizer_config)
    
    config = Config.from_pretrained(args.model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setattr(config, "device", device)
    # print(vars(args), config.pad_word_len)
    config.update(vars(args))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # for key, value in config.__dict__.items():
    #     print(key, type(value))
    # exit()

    config.id2label, config.label2id = None, None
    train_data = NERDataset(args.train_path, tokenizer=tokenizer, config=config)
    config.id2label, config.label2id = train_data.id2label, train_data.label2id
    print(config.id2label)
    config.num_labels = train_data.get_num_labels()
    tokenizer = train_data.tokenizer
    # train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_data = NERDataset(args.valid_path, tokenizer=tokenizer, config=config)
    assert train_data.id2label == valid_data.id2label and train_data.label2id == valid_data.label2id
    # valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)


    model = NERModel(config=config, tokenizer=tokenizer)
    model.to(device)
    # model = model.from_pretrained(args.model_path, config=config)

    if args.do_train:
        train(model, train_data, config, valid_data)
    if args.do_eval:
        evaluate(model, valid_data, config)
