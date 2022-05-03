from tqdm import tqdm
import json
from collections import Counter, defaultdict


class Config:
    @classmethod
    def from_pretrained(cls, path):
        with open(path) as f:
            mydict = json.load(f)
            for key, value in mydict.items():
                setattr(cls, key, value)
        return cls
    
    @classmethod
    def update(cls, mydict):
        for key, value in mydict.items():
            setattr(cls, key, value)
    
    @classmethod
    def to_dict(cls):
        mydict = {}
        for key, value in cls.__dict__.items():
            if isinstance(value, (float, str, list, dict, int, bool)) or value is None:
                mydict[key] = value
        return mydict


class SemHashTokenizer:
    def __init__(self, config=None) -> None:
        self.config = config
        self.no_hashing_tokens = [
            self.config.pad_semhash_token,
        ]
        self.token2id = dict()

    def find_ngrams(self, input_list):
        if len(input_list) < self.config.n_gram:
            return zip(*input_list)
        return zip(*[input_list[i:] for i in range(self.config.n_gram)])

    def hashing_token(self, unhashed_token):
        """
        convert token to list of subword
        """
        if unhashed_token in self.no_hashing_tokens:
            return [unhashed_token]
        unhashed_token = "#{}#".format(unhashed_token)
        return [
            "".join(gram)
            for gram in list(self.find_ngrams(list(unhashed_token)))
        ]

    def tokenize(self, text):
        """
        convert sequence to list of subword
        """
        tokens = text.split(" ")
        final_tokens = []
        for unhashed_token in tokens:
            final_tokens += self.hashing_token(unhashed_token)
            # final_tokens += unhashed_token
        return final_tokens

    def encode(self, text_input):
        """
        convert list of sequences to list of list subwords
        """
        words = text_input.split(" ")
        hash_input = []
        # assert len(words) == len(labels)
        for word in words:
            word_hash = self.hashing_token(word)
            # word_hash = word
            word_hash_id = [
                self.token2id.get(token, self.unk_token_id)
                for token in word_hash
            ]
            if self.config.pad_word_len > 0:
                if len(word_hash) < self.config.pad_word_len:
                    word_hash_id.extend(
                        [self.config.pad_semhash_token]
                        * (self.config.pad_word_len - len(word_hash))
                    )
                elif len(word_hash) > self.config.pad_word_len:
                    word_hash_id = word_hash_id[:self.config.pad_word_len]

            hash_input.extend(word_hash_id)
        # return " ".join(map(str, hash_input)), " ".join(map(str, hash_label))
        return hash_input

    # def do_hash_sample_input(self, text_input, ):
    #     """
    #     convert list of sequences to list of list subwords
    #     """
    #     words = text_input.split(" ")
    #     hash_input = []
    #     for word in words:
    #         word_hash = self.hashing_token(word, n_gram)
    #         word_hash_id = [
    #             hash_dict["hash2idx"].get(item, hash_dict["hash2idx"][UNK])
    #             for item in word_hash
    #         ]
    #         if pad_word_len > 0:
    #             if len(word_hash) < pad_word_len:
    #                 word_hash_id.extend(
    #                     [hash_dict["hash2idx"][PAD_SEM_HASH_TOKEN]]
    #                     * (pad_word_len - len(word_hash))
    #                 )
    #             elif len(word_hash) > pad_word_len:
    #                 word_hash_id = word_hash_id[:pad_word_len]

    #         hash_input.extend(word_hash_id)
    #     return hash_input

    def get_ord_map(self):
        ord2char = {ord(" "): " "}
        hash_dict["hash2idx_ascii"] = dict({})
        for key in hash_dict["hash2idx"]:
            key_ascii = ""
            for char in key:
                ord_char = ord(char)
                if ord_char == 34:  # " char
                    ord2char[ord_char] = '"'
                    key_ascii += '"'
                elif ord_char == 92:  # \ char
                    ord2char[ord_char] = "\\"
                    key_ascii += "\\"
                else:
                    ord2char[ord_char] = json.encoder.encode_basestring_ascii(char)[
                        1:-1
                    ]
                    key_ascii += ord2char[ord_char]
            hash_dict["hash2idx_ascii"][key_ascii] = hash_dict["hash2idx"][key]
        hash_dict["ord2char"] = ord2char
        return hash_dict

    def build_vocab_label(self, labels):

        self.label2id = dict()
        self.label2id[self.config.pad_label] = 0
        self.label2id[self.config.pad_semhash_label] = 1

        label_set = set()
        for label in labels:
            label_set.update(label.split(" "))
        label_set = sorted(list(label_set))
        for tag in label_set:
            if tag:
                self.label2id[tag] = len(self.label2id)

        self.id2label = {idx: word for word, idx in self.label2id.items()}

    def build_vocab(self, samples):
        """
        index hash token
        """
        tokenized_samples = [self.tokenize(sample) for sample in samples]
        token2id = Counter()

        for tokenized_sample in tqdm(
            tokenized_samples, total=len(samples), desc="Building vocabulary"
        ):
            token2id.update(tokenized_sample)
        token2id = token2id.most_common()

        self.token2id[self.config.pad_token] = 0
        self.token2id[self.config.unk_token] = 1
        self.token2id[self.config.pad_semhash_token] = 2
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.pad_semhash_token = 2
        for token, _ in token2id:
            self.token2id[token] = len(self.token2id)

        self.id2token = {id: token for token, id in self.token2id.items()}
        print(f"# of vocab: {len(self.token2id)}")

    def save_vocab(self, file_path):
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(
                {"token2id": self.token2id, "id2token": self.id2token},
                file,
                indent=4,
                ensure_ascii=False,
            )
    
    def load_vocab(self, file_path):
        with open(file_path) as f:
            data = json.load(f)
        self.token2id = data["token2id"]
        self.id2token = data["id2token"]
    

    def save_vocab_label(self, file_path):
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(
                {"label2id": self.label2id, "id2label": self.id2label},
                file,
                indent=4,
                ensure_ascii=False,
            )
    
    def load_vocab_label(self, file_path):
        with open(file_path) as f:
            data = json.load(f)
        self.label2id = data["label2id"]
        self.id2label = data["id2label"]


if __name__ == "__main__":
    config = Config.from_pretrained("config.json")
    tokenizer = SemHashTokenizer(config)
    tokenizer.build_vocab(["Nguyễn An Bình"])
    print(tokenizer.encode("Nguyễn Đức Bình"))
    # m_hash_dict = tokenizer.make_hash_dict(["Nguyễn Thái Bình"], ['O O O'], 3)
    # print(m_hash_dict)
    # print(tokenizer.do_hash_sample("Nguyễn Thái Bình", 'O O O', m_hash_dict, 0, 3))
