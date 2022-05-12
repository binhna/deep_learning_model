from transformers import RobertaPreTrainedModel, AutoModel
import torch

from .crf_layer import CRF


class ConVEx(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.roberta = AutoModel.from_config(config, add_pooling_layer=False)

        self.activation_fn = torch.nn.GELU()

        # template_fnn_size = 128
        self.template_fnn = torch.nn.Linear(
            config.hidden_size, config.template_fnn_size
        )
        # input_fnn_size = 128
        self.input_fnn = torch.nn.Linear(config.hidden_size, config.input_fnn_size)

        self.core_block = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "self_att": torch.nn.MultiheadAttention(
                            config.input_fnn_size,
                            config.num_heads,
                            dropout=config.hidden_dropout_prob,
                            batch_first=True,
                        ),
                        "layer_norm_1": torch.nn.LayerNorm(config.input_fnn_size),
                        "att": torch.nn.MultiheadAttention(
                            config.input_fnn_size,
                            config.num_heads,
                            dropout=config.hidden_dropout_prob,
                            batch_first=True,
                        ),
                        "layer_norm_2": torch.nn.LayerNorm(config.input_fnn_size),
                        "fnn": torch.nn.Linear(
                            config.input_fnn_size, config.input_fnn_size
                        ),
                        "layer_norm_3": torch.nn.LayerNorm(config.input_fnn_size),
                    }
                )
                for _ in range(2)
            ]
        )

        self.classifier = torch.nn.Linear(config.input_fnn_size, config.num_labels)

        # self.self_att = torch.nn.MultiheadAttention(
        #     config.input_fnn_size,
        #     config.num_heads,
        #     dropout=config.hidden_dropout_prob,
        #     batch_first=True,
        # )
        # self.layer_norm_1 = torch.nn.LayerNorm(config.input_fnn_size)

        # self.att = torch.nn.MultiheadAttention(
        #     config.input_fnn_size,
        #     config.num_heads,
        #     dropout=config.hidden_dropout_prob,
        #     batch_first=True,
        # )

        # self.layer_norm_2 = torch.nn.LayerNorm(config.input_fnn_size)

        # self.classifier = torch.nn.Linear(config.input_fnn_size, self.config.num_labels)

        # self.layer_norm_3 = torch.nn.LayerNorm(config.num_labels)

        self.crf_layer = CRF(num_tags=self.config.num_labels, batch_first=True)

        self.dropout = torch.nn.Dropout(
            config.classifier_dropout
            if config.classifier_dropout
            else config.hidden_dropout_prob
        )
        if config.finetuning:
            self.freeze_encoder()

    def freeze_encoder(self):
        for name, param in self.roberta.named_parameters():
            # print(name, param.requires_grad)
            param.requires_grad = False

    def forward(
        self,
        template_ids=None,
        template_attention_mask=None,
        template_token_type_ids=None,
        template_position_ids=None,
        input_ids=None,
        input_attention_mask=None,
        input_token_type_ids=None,
        input_position_ids=None,
        labels=None,
    ):
        outputs_input = self.roberta(
            input_ids,
            attention_mask=input_attention_mask,
            token_type_ids=input_token_type_ids,
            position_ids=input_position_ids,
        )
        outputs_input = outputs_input[0]
        outputs_input = self.dropout(outputs_input)

        if template_ids is not None:
            # pretraining
            outputs_template = self.roberta(
                template_ids,
                attention_mask=template_attention_mask,
                token_type_ids=template_token_type_ids,
                position_ids=template_position_ids,
            )

            outputs_template = outputs_template[0]

            outputs_template = self.dropout(outputs_template)

            outputs_template = self.activation_fn(self.template_fnn(outputs_template))
        else:
            # finetuning
            outputs_template = self.activation_fn(self.template_fnn(outputs_input))

        # batch x seq x input_fnn_dim
        outputs_input = self.activation_fn(self.input_fnn(outputs_input))
        residual = outputs_input

        for block in self.core_block:
            # batch x seq x input_fnn_dim
            outputs_input, _ = block["self_att"](
                query=outputs_input,
                key=outputs_input,
                value=outputs_input,
                key_padding_mask=input_attention_mask,
            )
            # print("output self_att", outputs_input.size())
            # batch x seq x input_fnn_dim
            outputs_input = block["layer_norm_1"](outputs_input) + residual
            residual = outputs_input
            # print("output add norm 1", outputs_input.size())

            # batch x seq x input_fnn_dim
            outputs_input, _ = block["att"](
                query=outputs_input,
                key=outputs_template,
                value=outputs_template,
                key_padding_mask=template_attention_mask,
            )
            # print("output attention", outputs_input.size())
            # batch x seq x input_fnn_dim
            outputs_input = block["layer_norm_2"](outputs_input) + residual
            residual = outputs_input
            # print("output add norm 2", outputs_input.size())
            # batch x seq x input_fnn_dim
            outputs_input = self.activation_fn(block["fnn"](outputs_input))
            # print("output fnn", outputs_input.size())
            outputs_input = block["layer_norm_3"](outputs_input) + residual
            # print("output add norm 3", outputs_input.size())

        logits = self.classifier(outputs_input)
        # print("input to crf", logits.size())

        # outputs_input, _ = self.self_att(
        #     query=outputs_input,
        #     key=outputs_input,
        #     value=outputs_input,
        #     key_padding_mask=input_attention_mask,
        # )

        # outputs_input = self.layer_norm_1(outputs_input)

        # outputs_input, _ = self.att(
        #     query=outputs_input,
        #     key=outputs_template,
        #     value=outputs_template,
        #     key_padding_mask=template_attention_mask,
        # )

        # outputs_input = self.layer_norm_2(outputs_input)

        # # print(outputs_input.size())
        # logits = self.classifier(outputs_input)

        # logits = self.layer_norm_3(logits)

        # # crf
        # attention_mask = attention_mask.type(torch.ByteTensor).to(attention_mask.device)
        # infer
        batch_size, seq_length = input_ids.size()
        labels = (
            labels
            if labels is not None
            else torch.ones((batch_size, seq_length), dtype=torch.long).to(
                input_ids.device
            )
        )

        loss = -self.crf_layer(emissions=logits, tags=labels)  # , mask=attention_mask)
        tags = self.crf_layer.decode(emissions=logits)  # , mask=attention_mask)
        tags = torch.LongTensor(tags)
        # print(tags)
        return {"tags": tags, "loss": loss}
