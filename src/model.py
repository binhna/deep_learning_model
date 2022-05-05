from transformers import RobertaPreTrainedModel, AutoModel, AdapterConfig
import torch

from .crf_layer import CRF


class RobertaNER(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.roberta = AutoModel.from_config(config, add_pooling_layer=False)
        self.use_adapter = config.use_adapter
        # print("\n\n\n", self.roberta.config.adapters)
        if config.use_crf:
            self.crf_layer = CRF(num_tags=self.config.num_labels, batch_first=True)

        if config.use_adapter:
            print("\n\nadd adapters")
            if config.task_name not in self.roberta.config.adapters:
                task_config = AdapterConfig.load("pfeiffer", reduction_factor=6)
                self.roberta.add_adapter(config.task_name, config=task_config)
            self.roberta.train_adapter([config.task_name])
            self.roberta.set_active_adapters([config.task_name])

        self.dropout = torch.nn.Dropout(
            config.classifier_dropout
            if config.classifier_dropout
            else config.hidden_dropout_prob
        )
        self.classifier = torch.nn.Linear(config.hidden_size, self.config.num_labels)
        self.freeze_layers()
        # print(hasattr(self, "classifier"))

    def freeze_layers(self):
        # for name, param in self.roberta.named_parameters():
        #     print(name, param.requires_grad)
        if self.config.freeze_layer_count and not self.config.use_adapter:
            # We freeze here the embeddings of the model
            for param in self.roberta.embeddings.parameters():
                param.requires_grad = False

            if self.config.freeze_layer_count != -1:
                # if freeze_layer_count == -1, we only freeze the embedding layer
                # otherwise we freeze the first `freeze_layer_count` encoder layers
                for layer in self.roberta.encoder.layer[
                    : self.config.freeze_layer_count
                ]:
                    for param in layer.parameters():
                        param.requires_grad = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]
        # print(sequence_output.size())
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if self.config.use_crf:
            attention_mask = attention_mask.type(torch.ByteTensor).to(attention_mask.device)
            
            # infer
            batch_size, seq_length = input_ids.size()
            labels = labels if labels is not None else torch.ones((batch_size, seq_length), dtype=torch.long).to(input_ids.device)
            
            loss = -self.crf_layer(emissions=logits, tags=labels)#, mask=attention_mask)
            tags = self.crf_layer.decode(emissions=logits)#, mask=attention_mask)
            tags = torch.LongTensor(tags)
            # print(tags)
            return {"tags": tags, "loss": loss}
        # print(logits)
        # exit()
        # return {"logits": logits, "id2label": self.config.id2label}
        return {"logits": logits}
