import os
import torch
from tqdm import tqdm
import subprocess


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


if __name__ == "__main__":
    dataset = ConVExDataset("data/test.txt")
    print(len(dataset))
