import os
import torch
import numpy as np
import hashlib
from tqdm import tqdm
import pandas as pd


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, dir_data, tokenizer, config) -> None:
        assert os.path.exists(dir_data), f"{dir_data} does not exist"
        self.tokenizer = tokenizer
        self.config = config
        
        self.samples = []

        self._load_text(dir_data)

    def __len__(self):
        return len(self.samples)

    def get_num_labels(self):
        return len(self.id2label)

    def add_sample(self, new_sample):
        text = "".join(new_sample["words"]) + "".join(new_sample["tags"])
        hashtext = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if hashtext not in self.hashed_samples:
            self.hashed_samples.append(hashtext)
            self.samples.append(new_sample)

    def encode_samples(self):
        tmp_samples = []
        for sample in tqdm(self.samples):
            if self.config.task == "seq_tagging":
                words = sample["words"]
                tags = sample["tags"]
                input_ids = []
                labels = []
                for word, tag in zip(words, tags):
                    ids = self.tokenizer.encode(word)
                    # if len(input_ids) + len(ids) > self.max_seq_length - 2:
                    #     break
                    input_ids.extend(ids)
                    if len(ids) == 1:
                        labels.append(self.label2id[tag])
                    else:
                        if tag == "O":
                            labels.extend([self.label2id[tag]] * len(ids))
                        else:
                            if self.config.use_crf:
                                realtag = tag[2:]
                                realtag = f"I-{realtag}"
                            else:
                                realtag = "X"
                            labels.extend(
                                [self.label2id[tag]] + [self.label2id[realtag]] * (len(ids) - 1)
                            )

                assert np.array_equal(
                    np.array(input_ids),
                    np.array(self.tokenizer.encode(" ".join(words))),
                ), [" ".join(words), input_ids, self.tokenizer.encode(" ".join(words))]
            elif self.config.task == "text_classification":
                input_ids = self.tokenizer.encode(sample["text"])
                labels = [self.label2id.get(sample["label"])]
                # print(labels)
            
            tmp_samples.append({
                "input_ids": torch.LongTensor(input_ids),
                "labels": torch.LongTensor(labels),
                "lengths": len(input_ids)
            })
        
        if self.config.task == "seq_tagging":
            self.samples = sorted(tmp_samples, key=lambda x: x["lengths"], reverse=True)
        else:
            self.samples = tmp_samples


    def _load_text(self, dir_data):
        id2label = set()

        if self.config.format == "conll":
            with open(dir_data) as f:
                sample = {"words": [], "tags": []}
                for line in f:
                    line = line.strip()
                    if line.startswith("-DOCSTART") or not line:
                        if sample["words"]:
                            self.samples.append(sample)
                            sample = {"words": [], "tags": []}
                    else:
                        parts = line.split()
                        sample["words"].append(parts[0])
                        sample["tags"].append(parts[-1])
                        id2label.add(parts[-1])
                if sample["words"] and sample["tags"]:
                    self.samples.append(sample)

        elif self.config.format == "flatten" and dir_data.endswith(".src"):
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
                    if self.config.lower:
                        sent = sent.lower()
                    tokens = sent.strip().split()
                    tags = label.strip().split()
                    assert len(tokens) == len(tags), [tokens, tags]
                    self.samples.append({"words": tokens, "tags": tags})
                    id2label.update(tags)

                    if self.config.augment_lower and not self.config.lower:
                        # lower
                        self.samples.append(
                            {"words": [token.lower() for token in tokens], "tags": tags}
                        )
        elif self.config.format == "csv":
            with open(dir_data, encoding = "ISO-8859-1") as f:
                for line in f:
                    parts = line.split(",")
                    label = parts[0].strip()
                    text = ",".join(parts[1:]).strip()
                    if self.config.task == "text_classification":
                        id2label.add(label)
                        self.samples.append({"text": text, "label": label})
                    elif self.config.task == "seq_tagging":
                        id2label.update(label.split())
                        self.samples.append({"words": text.split(), "tags": label.split()})


        if not self.tokenizer.token2id:
            if self.config.task == "seq_tagging":
                self.tokenizer.build_vocab([" ".join(sample["words"]) for sample in self.samples])
            elif self.config.task == "text_classification":
                self.tokenizer.build_vocab([sample["text"] for sample in self.samples])
        # self.samples = sorted(self.samples, key=lambda x: x["tags"], reverse=True)
        # reverse true to make O the first index, 0 index will be used for padding
        id2label = sorted(list(id2label), reverse=True)
        self.id2label = {i: tag for i, tag in enumerate(id2label)} if not self.config.id2label else self.config.id2label
        if self.config.task == "seq_tagging" and not self.config.use_crf:
            self.id2label[-100] = "X"
        self.label2id = {label: idx for idx, label in self.id2label.items()} if not self.config.label2id else self.config.label2id
        self.encode_samples()

    def __getitem__(self, index):
        return self.samples[index]


if __name__ == "__main__":
    dataset = NERDataset("data/test.txt")
    print(len(dataset))
