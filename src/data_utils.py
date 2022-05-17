import os

from collections import Counter, defaultdict
import torch
from tqdm import tqdm
import subprocess
import numpy as np
import random


class ConVExDataset(torch.utils.data.Dataset):
    def __init__(self, dir_data, tokenizer, config) -> None:
        assert os.path.exists(dir_data), f"{dir_data} does not exist"
        self.tokenizer = tokenizer
        self.config = config
        self.max_seq_length = (
            config.max_position_embeddings - tokenizer.pad_token_id - 1
        )
        # stores all samples
        self.samples = []
        self.label2id = config.label2id
        self.id2label = config.id2label

        self._load_text(dir_data)

    def __len__(self):
        return len(self.samples)

    def get_num_labels(self):
        return len(self.id2label)

    def _load_text(self, dir_data):
        num_line = subprocess.check_output(["wc", "-l", dir_data]).split()[0]
        num_line = int(num_line)
        with open(dir_data) as f:
            for line in tqdm(
                f,
                total=num_line,
                bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
            ):
                if self.config.lower:
                    line = line.lower()
                template_sent, input_sent, phrase = line.strip().split("\t")
                phrase = "" if phrase == "・" else phrase
                assert (
                    phrase in template_sent and phrase in input_sent
                ) or not phrase, f"{phrase} | {template_sent}: {phrase in template_sent}\n{phrase} | {input_sent}: {phrase in input_sent}"
                self.samples.append(
                    {"template": template_sent, "input": input_sent, "phrase": phrase}
                )

                if self.config.augment_lower and not self.config.lower:
                    # lower
                    self.samples.append(
                        {
                            "template": template_sent.lower(),
                            "input": input_sent.lower(),
                            "phrase": phrase.lower(),
                        }
                    )

    def __getitem__(self, index):
        template_sent = self.samples[index]["template"]
        input_sent = self.samples[index]["input"]
        phrase = self.samples[index]["phrase"]

        if phrase:
            phrase_ids = self.tokenizer.encode(phrase, add_special_tokens=False)
            mask_phrase_ids = [self.tokenizer.mask_token_id] * len(phrase_ids)

            # prepare template ids
            # keep the mask span incased the seq is longer than max_seq_len and the mask span is being trim
            template_ids = []
            template_sent = template_sent.replace(phrase, self.tokenizer.mask_token)
            len_word = []
            index_mask_phrase = 0
            for i, word in enumerate(template_sent.split()):
                if word == self.tokenizer.mask_token:
                    template_ids.append(mask_phrase_ids)
                    len_word.append(len(mask_phrase_ids))
                    index_mask_phrase = i
                    continue
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                template_ids.append(ids)
                len_word.append(len(ids))

            tmp_template_ids = sum(template_ids, [])
            if sum(len_word) > self.max_seq_length - 2:
                tmp_template_ids = template_ids[index_mask_phrase]
                left = index_mask_phrase - 1
                right = index_mask_phrase + 1
                while True:
                    if left >= 0:
                        if (
                            len(template_ids[left]) + len(tmp_template_ids)
                            < self.max_seq_length - 2
                        ):
                            tmp_template_ids = template_ids[left] + tmp_template_ids
                            left -= 1
                        else:
                            break
                    if right < len(template_ids):
                        if (
                            len(template_ids[right]) + len(tmp_template_ids)
                            < self.max_seq_length - 2
                        ):
                            tmp_template_ids.extend(template_ids[right])
                            right += 1
                        else:
                            break
                    if left < 0 and right >= len(template_ids):
                        break

            template_ids = (
                [self.tokenizer.cls_token_id]
                + tmp_template_ids
                + [self.tokenizer.sep_token_id]
            )

            template_attention_mask = [1] * len(template_ids)

            # prepare input ids and labels
            input_ids = []
            labels = []

            input_sent = input_sent.replace(phrase, self.tokenizer.mask_token)
            len_word = []
            index_mask_phrase = 0
            for i, word in enumerate(input_sent.split()):
                if word == self.tokenizer.mask_token:
                    input_ids.append(phrase_ids)
                    labels.append(["B-PHRASE"] + ["I-PHRASE"] * (len(phrase_ids) - 1))
                    len_word.append(len(phrase_ids))
                    index_mask_phrase = i
                    continue
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                input_ids.append(ids)
                labels.append(["O"] * len(ids))
                len_word.append(len(ids))

            tmp_input_ids = sum(input_ids, [])
            tmp_labels = sum(labels, [])
            if sum(len_word) > self.max_seq_length - 2:
                tmp_input_ids = input_ids[index_mask_phrase]
                tmp_labels = labels[index_mask_phrase]
                left = index_mask_phrase - 1
                right = index_mask_phrase + 1
                while True:
                    if left >= 0:
                        if (
                            len(input_ids[left]) + len(tmp_input_ids)
                            < self.max_seq_length - 2
                        ):
                            tmp_input_ids = input_ids[left] + tmp_input_ids
                            tmp_labels = labels[left] + tmp_labels
                            left -= 1
                        else:
                            break
                    if right < len(input_ids):
                        if (
                            len(input_ids[right]) + len(tmp_input_ids)
                            < self.max_seq_length - 2
                        ):
                            tmp_input_ids.extend(input_ids[right])
                            tmp_labels.extend(labels[right])
                            right += 1
                        else:
                            break
                    if left < 0 and right >= len(input_ids):
                        break
            input_ids = (
                [self.tokenizer.cls_token_id]
                + tmp_input_ids
                + [self.tokenizer.sep_token_id]
            )
            labels = ["O"] + tmp_labels + ["O"]
            labels = [self.label2id.get(label) for label in labels]
            assert len(labels) == len(input_ids), [len(labels), len(input_ids)]

            input_attention_mask = [1] * len(input_ids)
        else:
            # negative sample
            template_ids = self.tokenizer.encode(template_sent, add_special_tokens=False)
            template_ids = template_ids[:self.max_seq_length - 1]
            template_attention_mask = [1] * len(template_ids)

            input_ids = self.tokenizer.encode(input_sent, add_special_tokens=False)
            input_ids = input_ids[:self.max_seq_length - 1]
            input_attention_mask = [1] * len(input_ids)
            
            labels = ["O"] * len(input_ids)
            labels = [self.label2id.get(label) for label in labels]

        return {
            "template_ids": torch.LongTensor(template_ids),
            "template_attention_mask": torch.LongTensor(template_attention_mask),
            "input_ids": torch.LongTensor(input_ids),
            "input_attention_mask": torch.LongTensor(input_attention_mask),
            "labels": torch.LongTensor(labels),
        }



class ConVexFTDataset(ConVExDataset):
    def __init__(self, dir_data, tokenizer, config) -> None:
        super().__init__(dir_data, tokenizer, config)
    

    def set_slot_name(self, slot_name):
        self.slot_name = slot_name
        self.id2label = {0: "O", 1: f"I-{slot_name}", 2: f"B-{slot_name}"}
        self.label2id = {label: idx for idx, label in self.id2label.items()}
        
        self.samples_ = []
        num_neg_sample_per_batch = int(self.config.batch_size*0.25)
        num_pos_sample_per_batch = self.config.batch_size - num_neg_sample_per_batch
        
        pos_samples = self.type2samples[slot_name].copy()
        if len(pos_samples) % num_pos_sample_per_batch != 0:
            pos_samples.extend(pos_samples[:num_pos_sample_per_batch-(len(pos_samples)%num_pos_sample_per_batch)])
        neg_samples = []
        for sn in self.slot_names:
            if sn != slot_name:
                neg_samples.extend(self.type2samples[sn])
        if len(neg_samples) % num_neg_sample_per_batch != 0:
            neg_samples.extend(neg_samples[:num_neg_sample_per_batch-(len(neg_samples)%num_neg_sample_per_batch)])

        print(f"# positive of samples {len(pos_samples)}")
        print(f"# negative of samples {len(neg_samples)}")

        while pos_samples and neg_samples:
            batch_ = []
            for _ in range(num_pos_sample_per_batch):
                batch_.append(pos_samples.pop(random.randrange(0, len(pos_samples))))
            for _ in range(num_neg_sample_per_batch):
                batch_.append(neg_samples.pop(random.randrange(0, len(neg_samples))))
            if len(batch_) == self.config.batch_size:
                random.shuffle(batch_)
                self.samples_.extend(batch_)
            else:
                break


    def get_slot_names(self):
        label_analysis = Counter()
        self.type2samples = defaultdict(list)
        for sample in self.samples:
            types = list(set([label[2:] for label in sample["tags"] if len(label) > 2]))
            if not types:
                self.type2samples["O"].append(sample)
            else:
                label_analysis.update(types)
                for type_ in types:
                    self.type2samples[type_].append(sample)
        label_analysis = label_analysis.most_common()
        self.label_counter = dict(label_analysis)
        label_analysis = [label[0] for label in reversed(label_analysis)]
        self.slot_names = label_analysis
        return label_analysis

    
    def __len__(self):
        return len(self.samples_)

    
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

        id2label = list(id2label)
        for label in id2label:
            if label.startswith("B-"):
                label = label.replace("B-", "I-")
                if label not in id2label:
                    id2label.append(label)
        id2label = sorted(id2label, reverse=True)
        self.id2label = {i: tag for i, tag in enumerate(id2label)} if not self.config.id2label else self.config.id2label
        self.label2id = {label: idx for idx, label in self.id2label.items()} if not self.config.label2id else self.config.label2id

    
    def __getitem__(self, index):
        try:
            words = self.samples_[index]["words"]
            tags = self.samples_[index]["tags"]
        except Exception as e:
            print(e)
            print(len(self.samples_), index)
        input_ids = []
        labels = []
        for word, tag in zip(words, tags):
            # each training only handles 1 slot name
            tag = "O" if len(tag) > 3 and tag[2:] != self.slot_name else tag
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
                    realtag = tag[2:]
                    realtag = f"I-{realtag}"
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

        # padding will be moved to batch collator
        # if len(input_ids) < self.max_seq_length:
        #     input_ids += [self.tokenizer.pad_token_id] * (
        #         self.max_seq_length - len(input_ids)
        #     )
        #     labels += [self.label2id["O"]] * (len(input_ids) - len(labels))
        #     attention_mask += [0] * (len(input_ids) - len(attention_mask))

        return {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(labels),
            "input_attention_mask": torch.LongTensor(attention_mask),
        }


if __name__ == "__main__":
    dataset = ConVExDataset("data/test.txt")
    print(len(dataset))
