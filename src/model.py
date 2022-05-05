import torch
from torch import nn

from src.crf_layer import CRF
from typing import List


class NERModel(torch.nn.Module):
    def __init__(self, config, tokenizer):
        super(NERModel, self).__init__()
        self.config = config
        # self.tokenizer = tokenizer
        self.id2label = config.id2label
        self.embedding = nn.Embedding(
            num_embeddings=len(tokenizer.token2id),
            embedding_dim=config.embedding_dim,
            padding_idx=tokenizer.pad_token_id,
        )

        self.cnn_head = torch.jit.script(CNNSubwordLevelBN(config))

        self.linear_hidden = torch.jit.script(
            nn.Linear(
                in_features=config.embedding_dim, out_features=config.tf_embedding_dim
            )
        )

        self.transformer_head = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.tf_embedding_dim,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.ffn_embedding_dim,
                    dropout=config.tf_dropout,
                    # attention_dropout=config.tf_dropout,
                    # activation_dropout=config.tf_dropout,
                    activation=config.activation_fn,
                    batch_first=True,
                )
                for _ in range(config.num_tf_encoder_layers)
            ]
        )

        self.features_transforms = nn.Linear(
            config.tf_embedding_dim, self.config.num_labels
        )
        self.dropout = nn.Dropout(config.dropout)
        # if self.config.use_crf:
        self.crf_layer = torch.jit.script(CRF(self.config.num_labels, batch_first=True))
        # self.pad_word_len = self.config.pad_word_len
        # dictionary = tokenizer.get_ord_map(self.config.dictionary)
        # self.ord2char = dictionary["ord2char"]
        # self.hash2idx = dictionary["hash2idx_ascii"]
        # self.idx2label = dictionary["idx2label"]
        # self.n_gram = config.n_gram
        # self.UNK = tokenizer.UNK
        # self.PAD_SEM_HASH_TOKEN = tokenizer.PAD_SEM_HASH_TOKEN
        # self.padding_idx = config.dictionary["hash2idx"][tokenizer.PAD_TOKEN]

    def forward(self, input_ids):
        # batch x num_subwords
        # batch_size = src_tokens.size(0)
        # num_words = src_tokens.size(1) // self.pad_word_len
        # src_tokens = src_tokens.view(batch_size, num_words, self.pad_word_len)
        # src_tokens = nn.utils.rnn.pack_padded_sequence(input_ids, lengths=lengths, batch_first=True)

        # batch x num_subwords x embedding dim
        src_tokens_embedding = self.embedding(input_ids)

        # batch x num_subwords x embedding dim
        words_embedding = self.cnn_head(src_tokens_embedding)
        words_embedding = self.dropout(words_embedding)
        # words_embedding = src_tokens_embedding

        # n_gram_words_embedding = self.word_transforms(words_embedding)
        # n_gram_words_embedding = self.dropout(n_gram_words_embedding)
        # sentence_embedding = torch.cat(
        #     [words_embedding, n_gram_words_embedding], dim=-1
        # )
        # batch x num_subwords x tf hidden size
        sentence_embedding = torch.tanh(self.linear_hidden(words_embedding))
        # print("sentence_embedding", sentence_embedding.size())
        sentence_embedding = sentence_embedding.squeeze(1)

        # # account for padding while computing the representation
        # padding_mask = input_ids.eq(self.tokenizer.pad_token_id).to(input_ids.device)
        # x = sentence_embedding
        # if padding_mask is not None:
        #     x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # # B x T x C -> T x B x C
        # # x = x.transpose(0, 1)
        # for layer in self.transformer_head:
        #     x = layer(x, src_key_padding_mask=padding_mask)

        # sentence_embedding = x.transpose(0, 1)
        # if self.config.task == "text_classification":
        #     x = torch.max(x, dim=1).values
        # print(x.size())

        # batch x num subwords x num tags
        logits = self.features_transforms(sentence_embedding)

        # if self.config.task == "text_classification":
        # print(logits.size())
        # logits = logits[:, 0, :] # first word

        # print("labels", labels.size())
        # print("logits", logits.size())
        # loss_fct = nn.CrossEntropyLoss()
        # loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return {"logits": logits}

    @torch.jit.export
    def decode(self, src_tokens):
        src_tokens_embedding = self.embedding(src_tokens)
        words_embedding = self.cnn_head(src_tokens_embedding)
        words_embedding = self.dropout(words_embedding)

        sentence_embedding = torch.tanh(self.linear_hidden(words_embedding))
        logits = self.features_transforms(sentence_embedding)

        # if self.config.use_crf:
        return [
            [self.id2label[idx] for idx in item]
            for item in self.crf_layer.decode(logits)
        ]
        # return logits

    @torch.jit.export
    def encode_input(self, char_ord_input):
        # print(list(self.ord2char.keys()))
        # print([item.item() for item in char_ord_input[0]])
        input_str: str = "".join(
            [self.ord2char.get(item.item(), "\\u----") for item in char_ord_input[0]]
        )
        # print([ord(item) for item in input_str])
        words: List[str] = input_str.split()
        ids: List[int] = []
        for word in words:
            # if '\\u----' in word:
            #     print(word)
            word: str = "#{}#".format(word).replace("\\u", " \\u")
            input_list: List[str] = []
            for item in word.split():
                if item.startswith("\\u"):
                    input_list.extend([item[:6]] + list(item[6:]))
                else:
                    input_list.extend(list(item))
            # print([ord(i) for i in input_list])
            if len(input_list) < self.n_gram:
                ngrams = [input_list]
            else:
                # ngrams = zip(*[input_list[i:] for i in range(self.n_gram)])
                ngrams = [
                    input_list[i : i + self.n_gram]
                    for i in range(len(input_list) - self.n_gram + 1)
                ]
            word_ids = [
                self.hash2idx.get("".join(gram), self.hash2idx[self.UNK])
                for gram in list(ngrams)
            ]
            if len(word_ids) < self.pad_word_len:
                word_ids.extend(
                    [self.hash2idx[self.PAD_SEM_HASH_TOKEN]]
                    * (self.pad_word_len - len(word_ids))
                )
            elif len(word_ids) > self.pad_word_len:
                word_ids = word_ids[: self.pad_word_len]
            ids.extend(word_ids)
        return ids


class CNNSubwordLevelBN(torch.nn.Module):
    def __init__(self, config):
        super(CNNSubwordLevelBN, self).__init__()

        # self.config = config
        self._embedding_dim = config.embedding_dim
        self._num_filters = config.embedding_dim // len(config.semhash_window_sizes)
        self.semhash_window_sizes = config.semhash_window_sizes
        self._convolution_layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [
                        nn.Conv1d(
                            in_channels=self._embedding_dim,
                            out_channels=self._num_filters,
                            kernel_size=window_size,
                            padding="same",
                        ),
                        torch.nn.BatchNorm1d(self._num_filters),
                    ]
                )
                for window_size in self.semhash_window_sizes
            ]
        )
        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module("conv_layer_%d" % i, conv_layer)
        self.activation = nn.ReLU()

    def forward(self, features):
        batch_size, num_tokens, embedding_dim = features.size()

        # tokens = features.view(batch_size, num_tokens, embedding_dim)

        # batch x embed size x num subwords
        tokens = torch.transpose(features, 1, 2)

        filter_outputs = []
        for convolution_layer in self._convolution_layers:
            filter_outputs.append(
                self.activation(convolution_layer[1](convolution_layer[0](tokens)))
            )

        # if self.config.task == "text_classification":
        filter_outputs = [
            torch.nn.functional.max_pool1d(
                x_conv, kernel_size=x_conv.size()[-1]
            ).squeeze(-1)
            for x_conv in filter_outputs
        ]

        output = (
            torch.cat(filter_outputs, dim=1)
            if len(filter_outputs) > 1
            else filter_outputs[0]
        )

        return output.view(batch_size, -1, embedding_dim)
