import torch
from transformers import AutoTokenizer, AutoConfig
from collections import defaultdict
import os

from .model import RobertaNER


def get_span(ids, tags, subid2id, ignored_ids, tokenizer):
    result = defaultdict(list)
    tmp_ids = []
    current_tag = None
    # print(ignored_ids)
    for i, (id, tag) in enumerate(zip(ids, tags)):
        # print(tmp_ids)
        # if i in ignored_ids:
        #     continue
        if tag.startswith("B"):
            if tmp_ids:
                result[current_tag].append(tmp_ids)
            current_tag = tag[2:]
            tmp_ids = [i]
        elif tag.startswith("I") or tag.startswith("X"):
            tmp_ids.append(i)
            current_tag = tag[2:] if not current_tag else current_tag
        else:
            if tmp_ids:
                result[current_tag].append(tmp_ids)
            current_tag = None
            tmp_ids = []
    if tmp_ids and current_tag:
        result[current_tag].append(tmp_ids)

    print(tags)
    print(result)
    result_span = defaultdict(list)
    for tag, list_ids in result.items():
        for ids_ in list_ids:
            tmp_ids = [ids[id] for id in ids_]
            # for id in ids_:
            #     tmp_ids.extend(ids[subid2id[id]])
            result_span[tag].append(tokenizer.decode(tmp_ids).strip())
    print(dict(result_span))
    return result_span


## Load model
device = "cpu"
model_path = "./model/mSystemEntity"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
except:
    tokenizer = AutoTokenizer.from_pretrained("../shared_data/xlmr_6L", add_prefix_space=True)
config = AutoConfig.from_pretrained(model_path)

# adapter
if config.use_adapter:
    model = RobertaNER.from_pretrained(config.model_path, config=config)
    model.roberta.load_adapter(os.path.join(model_path, "ner"))
    model.classifier.load_state_dict(
        torch.load(os.path.join(model_path, "classifier.bin"))
    )
else:
    model = RobertaNER.from_pretrained(model_path, config=config)
model.to(device)
model.eval()

max_length = config.max_position_embeddings - tokenizer.pad_token_id - 1


