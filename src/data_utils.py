import os
import torch
from tqdm import tqdm
import glob
import re
import random
from multiprocessing import Pool


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, dir_data, tokenizer, config, split="train") -> None:
        assert os.path.exists(dir_data), f"{dir_data} does not exist"
        self.split = split
        self.tokenizer = tokenizer
        self.format_data = config.format
        self.augment_lower = config.augment_lower
        self.lower = config.lower
        self.config = config
        self.max_seq_length = (
            config.max_position_embeddings - tokenizer.pad_token_id - 1
        )
        # check duplicate
        self.hashed_samples = []
        # stores all samples
        self.samples = []
        self.failed_samples = {"mismatch": [], "index_oor": []}

        self.id2label = config.id2label if config.id2label else set()
        self._load_text(dir_data)
        if config.id2label:
            self.id2label = config.id2label
            self.label2id = {label: idx for idx, label in self.id2label.items()}
        random.shuffle(self.samples)
        print(len(self.failed_samples["mismatch"]), len(self.failed_samples["index_oor"]))
        print(split, len(self.samples))

    def __len__(self):
        return len(self.samples)

    def get_num_labels(self):
        return len(self.id2label)

    def add_sample(self, sample_join_char):
        sample = sample_join_char[0]
        join_char = sample_join_char[1]
        special_char = "▁"
        len_char_original_token = [len(token) for token in sample["words"]]
        new_tokens = self.tokenizer.tokenize(join_char.join(sample["words"]))
        if len(new_tokens) > self.max_seq_length * 2:
            return
        new_tags = []
        index_original_token = 0
        next_tag = False
        for token in new_tokens:
            _token = token.replace(special_char, "")
            if index_original_token >= len(len_char_original_token):
                self.failed_samples["index_oor"].append(sample)
                return
            if len(_token) == len_char_original_token[index_original_token]:
                if next_tag:
                    new_tags.append(next_tag)
                else:
                    new_tags.append(sample["tags"][index_original_token])
                    index_original_token += 1
                    next_tag = False
            elif 0 < len(_token) < len_char_original_token[index_original_token]:
                if next_tag:
                    new_tags.append(next_tag)
                else:
                    new_tags.append(sample["tags"][index_original_token])
                    next_tag = (
                        "O"
                        if sample["tags"][index_original_token] == "O"
                        else "I-" + sample["tags"][index_original_token][2:]
                    )
                len_char_original_token[index_original_token] -= len(_token)
                if len_char_original_token[index_original_token] == 0:
                    index_original_token += 1
                    next_tag = False
                elif len_char_original_token[index_original_token] < 0:
                    self.failed_samples["mismatch"].append(sample)
                    return
            else:
                new_tags.append("X")
        assert len(new_tokens) == len(new_tags), [
            new_tokens,
            new_tags,
            join_char.join(sample["words"]),
        ]
        self.samples.append({"text": join_char.join(sample["words"]), "tags": new_tags})


    def load_conll(self, file):
        join_char = " "
        if re.search(r"ja|cn|ko|th", file):
            join_char = ""
        tmp_samples = []
        id2label = set()
        with open(file) as f:
            sample = {"words": [], "tags": []}
            for line in f:
                line = line.strip()
                if line.startswith("-DOCSTART") or not line:
                    if sample["words"]:
                        # self.samples.append(sample)
                        self.add_sample((sample, join_char))
                        # tmp_samples.append((sample, join_char))
                        sample = {"words": [], "tags": []}
                else:
                    parts = line.split()
                    if not re.search(r"^(B-|I-)", parts[-1]) and parts[-1] != "O":
                        break
                    sample["words"].append(parts[0])
                    sample["tags"].append(parts[-1])
                    id2label.add(parts[-1])
            if sample["words"] and sample["tags"]:
                self.add_sample((sample, join_char))
                # self.samples.append(sample)
                # tmp_samples.append((sample, join_char))
        if self.split == "valid":
            random.shuffle(self.samples)
            self.samples = self.samples[:5000] if self.config.do_train else self.samples
        return [self.samples, self.failed_samples, id2label]


    def _load_text(self, dir_data):
        id2label = set()

        if self.format_data == "conll":
            if os.path.isdir(dir_data):
                files = glob.glob(os.path.join(dir_data, "*.txt"))
            else:
                files = [dir_data]
            with Pool(processes=30) as p:
                tmp_samples = list(tqdm(p.imap_unordered(self.load_conll, files),
                                    total=len(files), desc="Loading data"))
                
            for (samples, failed_samples, id2label_) in tmp_samples:
                self.samples.extend(samples)
                self.failed_samples.update(failed_samples)
                id2label.update(id2label_)

        elif self.format_data == "flatten" and dir_data.endswith(".src"):
            with open(dir_data) as f_src, open(
                dir_data.replace(".src", ".tgt")
            ) as f_tgt:
                sentences = f_src.read().split("\n")
                labels = f_tgt.read().split("\n")
                for sent, label in tqdm(
                    zip(sentences, labels),
                    total=len(sentences),
                    bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
                ):
                    if self.lower:
                        sent = sent.lower()
                    tokens = sent.strip().split()
                    tags = label.strip().split()
                    assert len(tokens) == len(tags), [tokens, tags]
                    self.samples.append({"words": tokens, "tags": tags})
                    # self.add_sample({"words": tokens, "tags": tags})
                    self.id2label.update(tags)

                    if self.augment_lower and not self.lower:
                        # lower
                        self.samples.append(
                            {"words": [token.lower() for token in tokens], "tags": tags}
                        )
                        # self.add_sample(
                        #     {"words": [token.lower() for token in tokens], "tags": tags}
                        # )

        if not self.id2label:
            self.id2label = sorted(list(id2label))
            self.id2label = {i: tag for i, tag in enumerate(self.id2label)}
        if not self.config.use_crf:
            self.id2label[-100] = "X"
        self.label2id = {label: idx for idx, label in self.id2label.items()}

    def __getitem__(self, index):
        text = self.samples[index]["text"]
        labels = [self.label2id[l] for l in self.samples[index]["tags"]]
        input_ids = self.tokenizer.encode(
            text,
            truncation=True,
            max_length=self.max_seq_length - 2,
            add_special_tokens=False,
        )
        labels = labels[: self.max_seq_length - 2]
        assert len(input_ids) == len(labels)
        input_ids = (
            [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        )
        labels = [self.label2id["O"]] + labels + [self.label2id["O"]]
        attention_mask = [1] * len(input_ids)

        if len(input_ids) < self.max_seq_length:
            input_ids += [self.tokenizer.pad_token_id] * (
                self.max_seq_length - len(input_ids)
            )
            labels += [self.label2id["X"]] * (len(input_ids) - len(labels))
            attention_mask += [0] * (len(input_ids) - len(attention_mask))

        return {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(labels),
            "attention_mask": torch.LongTensor(attention_mask),
        }


if __name__ == "__main__":
    dataset = NERDataset("data/test.txt")
    print(len(dataset))
