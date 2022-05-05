import os
import json
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, classification_report
from sklearn.metrics import classification_report as classification_report_sklearn


def compute_loss(model, input_ids, labels, config, return_outputs=False):
    # forward pass
    outputs = model(input_ids)
    if not config.use_crf:
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            logits.view(-1, config.num_labels), labels.view(-1)
        )
    else:
        loss = outputs.get("loss")
    return (loss, outputs) if return_outputs else loss


def collate_function(batch):
    input_ids, labels, lengths = [], [], []
    for sample in batch:
        input_ids.append(sample["input_ids"])
        labels.append(sample["labels"])
        lengths.append(sample["lengths"])
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return {"input_ids": input_ids, "labels": labels, "lengths": lengths}

def train(model, train_dataset, config, eval_dataset=None):
    model.train()

    dataloader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_function)


    param_optimizer = list(model.named_parameters())
    no_decay = ["LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if (not any(nd in n for nd in no_decay))
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if (any(nd in n for nd in no_decay))
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr)

    total_steps = len(dataloader) * config.epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=len(dataloader)//4, num_training_steps=total_steps
    )

    max_score = 0

    for epoch in range(config.epoch):
        model.to(config.device)
        loss_epoch = 0
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(config.device)
            labels = batch["labels"].to(config.device)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(model, input_ids, labels, config)
            loss.backward()
            # if (i + 1) % config.accumulation_step == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            # model.zero_grad()
            
            loss_epoch += loss.item()
        
        print(f"avg loss epoch {epoch+1}:", loss_epoch/(i+1))
        if config.do_eval:
            _ = evaluate(model, train_dataset, config)
            result = evaluate(model, eval_dataset, config)
            if result[config.metric_for_best] > max_score:
                print(f"Saving new best model with {config.metric_for_best} @ {result[config.metric_for_best]}")
                max_score = result[config.metric_for_best]
                torch.save(model.state_dict(), os.path.join(config.output_dir, f'best_model_{config.metric_for_best}_{config.task_name}.pt'))
                
                # torch script model
                model.eval()
                cpu_model = model.cpu()
                torch.jit.save(cpu_model, os.path.join(config.output_dir, f'best_model_{config.metric_for_best}_{config.task_name}_torchscript.pt'))
                
                with open(os.path.join(config.output_dir, f"config.json"), "w") as f:
                    json.dump(config.to_dict(), f, ensure_ascii=False)
            print("===================================")
            model.train()
    print(f"Best model with {config.metric_for_best}: {max_score}")



def evaluate(model, dataset, config):
    # print(config.id2label)
    model.eval()
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, collate_fn=collate_function)

    entity_pred, entity_gold = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(config.device)
            labels = batch["labels"].to(config.device)
            _, outputs = compute_loss(model, input_ids, labels, config, return_outputs=True)
            entity_logits = outputs.get("logits")
            entity_outputs = torch.argmax(entity_logits, dim=-1)
            entity_outputs = entity_outputs.detach().cpu().numpy()
            if config.task == "text_classification":
                entity_labels = torch.squeeze(batch["labels"], -1).detach().cpu().numpy()
                entity_pred.extend(entity_outputs)
                entity_gold.extend(entity_labels)
            elif config.task == "seq_tagging":
                entity_labels = batch["labels"].detach().cpu().numpy()
            
                for pred, gold in zip(entity_outputs, entity_labels):
                    pred = [config.id2label.get(idx) for idx in pred]
                    gold = [config.id2label.get(idx) for idx in gold]
                    entity_pred.append(pred[:len(gold)])
                    entity_gold.append(gold)
    
    if config.task == "seq_tagging":
        # exact match
        em = [np.array_equal(pred_, label_) for pred_, label_ in zip(entity_pred, entity_gold)]
        em = sum(em) / len(em)
        f1 = f1_score(entity_gold, entity_pred, average="macro")
        print(classification_report(entity_gold, entity_pred))
    elif config.task == "text_classification":
        # print(entity_pred[:10], entity_gold[:10])
        # print(entity_pred == entity_gold)
        entity_gold = [config.id2label[idx] for idx in entity_gold]
        entity_pred = [config.id2label[idx] for idx in entity_pred]
        result = classification_report_sklearn(entity_gold, entity_pred, zero_division=0)
        print(result)
        result = result = classification_report_sklearn(entity_gold, entity_pred, output_dict=True, zero_division=0)
        em = sum(np.array(entity_pred) == np.array(entity_gold))/len(entity_pred)
        f1 = result["macro avg"]["f1-score"]
    return {"f1": f1, "em": em}




# class CustomTrainer(Trainer):
    

#     def _save(self, output_dir: Optional[str] = None, state_dict=None):
#         # If we are executing this function, we are the process zero, so we don't check for that.
#         output_dir = output_dir if output_dir is not None else self.args.output_dir
#         os.makedirs(output_dir, exist_ok=True)
#         print(f"Saving model checkpoint to {output_dir}")
#         # Save a trained model and configuration using `save_pretrained()`.
#         # They can then be reloaded using `from_pretrained()`
#         if not isinstance(self.model, PreTrainedModel):
#             if isinstance(unwrap_model(self.model), PreTrainedModel):
#                 if state_dict is None:
#                     state_dict = self.model.state_dict()
#                 unwrap_model(self.model).save_pretrained(
#                     output_dir, state_dict=state_dict
#                 )
#             else:
#                 # logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
#                 if state_dict is None:
#                     state_dict = self.model.state_dict()
#                 torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
#         else:
#             self.model.config.to_json_file(os.path.join(output_dir, "config.json"))
#             if self.model.use_adapter:
#                 self.model.roberta.save_all_adapters(output_dir)
#                 if hasattr(self.model, "classifier"):
#                     torch.save(
#                         self.model.classifier.state_dict(),
#                         os.path.join(output_dir, "classifier.bin"),
#                     )
#             else:
#                 self.model.save_pretrained(output_dir, state_dict=state_dict)

#         if self.tokenizer is not None:
#             self.tokenizer.save_pretrained(output_dir)

#         # Good practice: save your training arguments together with the trained model
#         torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
