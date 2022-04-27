import os
import torch
import numpy as np
import hashlib
from tqdm import tqdm


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, dir_data, tokenizer, config) -> None:
        assert os.path.exists(dir_data), f"{dir_data} does not exist"
        self.tokenizer = tokenizer
        self.config = config
        self.max_seq_length = (
            config.max_position_embeddings - tokenizer.pad_token_id - 1
        )
        # stores all samples
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

        id2label = sorted(list(id2label))
        self.id2label = {i: tag for i, tag in enumerate(id2label)} if not self.config.id2label else self.config.id2label
        if not self.config.use_crf:
            self.id2label[-100] = "X"
        self.label2id = {label: idx for idx, label in self.id2label.items()} if not self.config.label2id else self.config.label2id

    def __getitem__(self, index):
        words = self.samples[index]["words"]
        tags = self.samples[index]["tags"]
        input_ids = []
        labels = []
        for word, tag in zip(words, tags):
            ids = self.tokenizer.encode(word, add_special_tokens=False)
            if len(input_ids) + len(ids) > self.max_seq_length - 2:
                break
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

        input_ids = (
            [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        )
        assert np.array_equal(
            np.array(input_ids[1:-1]),
            np.array(self.tokenizer.encode(" ".join(words))[1 : len(input_ids) - 1]),
        ), [" ".join(words), input_ids, self.tokenizer.encode(" ".join(words))]
        labels = [self.label2id["O"]] + labels + [self.label2id["O"]]
        attention_mask = [1] * len(input_ids)

        if len(input_ids) < self.max_seq_length:
            input_ids += [self.tokenizer.pad_token_id] * (
                self.max_seq_length - len(input_ids)
            )
            labels += [self.label2id["O"]] * (len(input_ids) - len(labels))
            attention_mask += [0] * (len(input_ids) - len(attention_mask))

        return {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(labels),
            "attention_mask": torch.LongTensor(attention_mask),
        }


if __name__ == "__main__":
    dataset = NERDataset("data/test.txt")
    print(len(dataset))
