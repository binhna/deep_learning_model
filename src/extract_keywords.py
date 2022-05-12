from collections import defaultdict
import torch
import numpy as np
import json
import glob
import os

from src.infer import tokenizer, get_span, max_length, config, device


if __name__ == "__main__":
    prediction_files = glob.glob("prediction_ids_batch_*.npz")
    prediction_files = sorted(prediction_files)
    for file in prediction_files:
        print(os.path.basename(file))
        end = int(os.path.basename(file).split("_")[-1].split(".")[0])
        start = end - 1000000
        if end > 6000000:
            start = end - 981065
        predictions = np.load(file, allow_pickle=True)
        predictions = predictions["predictions"]
        # print([predictions.shape, start, end])
        output_keywords = []
        line_in_prediction = 0
        total_raw = predictions.shape[0]

        # max_num_sample = predictions.shape[0]
        with open("data/wiki_sent_segment.src") as f, open("extract_keywords_2.txt", "a+") as f2:
            line_num = 0
            for text in f:
                if line_num < start:
                    line_num += 1
                    continue
                elif line_num > end or line_in_prediction >= total_raw:
                    break
                text = text.strip()
                # text = input("Enter sentence: ")
                # text = re.sub(r"\s+", " ", text)

                # start = time.time()
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

                # print("ignore_ids", ignore_ids)

                inputs = tokenizer(
                    text,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    inputs = inputs.to(device)
                    # output = model(**inputs)
                    # if not model.config.use_crf:
                    #     logits = output.get("logits")
                    #     preds = torch.argmax(logits, dim=-1)[0].numpy()
                    # else:
                    #     preds = output.get("tags")[0].numpy()
                    preds = predictions[line_in_prediction]
                    line_in_prediction += 1
                    index_end = inputs["input_ids"][0].tolist().index(tokenizer.sep_token_id)
                    # print(preds[1:index_end])
                    preds = [config.id2label.get(idx, "O") for idx in preds]
                    # print(preds[1:index_end])
                    tokens = tokenizer.convert_ids_to_tokens(
                        inputs["input_ids"][0][1:index_end]
                    )
                    # print(tokens)
                    # print(subid2id)
                    # get_span(inputs.ids)

                    result = get_span(
                        inputs.input_ids[0][1:index_end].cpu().numpy(),
                        preds[1:index_end],
                        subid2id,
                        ignore_ids,
                        tokenizer,
                    )
                    # print([text, dict(result), line_num])
                    f2.write(json.dumps({"text": text, "keywords": dict(result)}, ensure_ascii=False) + "\n")
                    # print(round(time.time() - start, 5) * 1000, "miliseconds")
                line_num += 1