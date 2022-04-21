import re
import time
from collections import defaultdict
import torch

from src.infer import tokenizer, model, get_span, max_length, config


if __name__ == "__main__":
    while True:
        text = input("Enter sentence: ")
        text = re.sub(r"\s+", " ", text)

        start = time.time()
        subid = 0
        subid2id = defaultdict(list)
        ignore_ids = []
        for i, word in enumerate(text.split()):
            subwords = tokenizer.tokenize(word)
            if len(subwords) > 1:
                for j in range(len(subwords)):
                    subid2id[subid].append(subid + j)
                    if j != 0:
                        ignore_ids.append(subid + j)
                subid += len(subwords) - 1
            else:
                subid2id[subid].append(subid)
            subid += 1
        print(subid2id)
        # print("ignore_ids", ignore_ids)

        inputs = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = model(**inputs)
            logits = output.get("logits")
            preds = torch.argmax(logits, dim=-1)[0].numpy()
            index_end = inputs["input_ids"][0].tolist().index(tokenizer.sep_token_id)
            print(preds[1:index_end])
            preds = [config.id2label[idx] for idx in preds]
            # print(preds[1:index_end])
            tokens = tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0][1:index_end]
            )


            result = get_span(
                inputs.input_ids[0][1:index_end].numpy(),
                preds[1:index_end],
                subid2id,
                ignore_ids,
                tokenizer,
            )
            print(round(time.time() - start, 5) * 1000, "miliseconds")