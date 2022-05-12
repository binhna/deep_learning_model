from underthesea import sent_tokenize
import subprocess
from tqdm import tqdm
import json
from collections import defaultdict
import random
import subprocess

delimiter = "・"


def prepare_training_data(keywords_path):
    keyword2sentence = defaultdict(list)
    num_line = int(subprocess.check_output(["wc", "-l", keywords_path]).split()[0])
    with open(keywords_path) as f:
        for line in tqdm(f, total=num_line, bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}"):
            sample = json.loads(line.strip())
            if sample["keywords"]:
                for entity_type, entities in sample["keywords"].items():
                    for entity in entities:
                        if sample["text"].count(entity) == 1:
                            keyword2sentence[f"{entity_type}{delimiter}{entity}"].append(sample["text"])
            else:
                keyword2sentence[delimiter].append(sample["text"])
    dataset = make_pair_sample(keyword2sentence)
    print(len(dataset["pos"]), len(dataset["neg"]))
    training_data = []

    tmp_neg = dataset["neg"].copy()
    tmp_pos = dataset["pos"].copy()

    while len(dataset["pos"]):
        batch = []
        for _ in range(192):
            if dataset["pos"]:
                batch.append(pop_random(dataset["pos"]))
            else:
                batch.append(random.choice(tmp_pos))
        for _ in range(64):
            if dataset["neg"]:
                batch.append(pop_random(dataset["neg"]))
            else:
                batch.append(random.choice(tmp_neg))
        random.shuffle(batch)
        training_data.extend(batch)
    print(f"{len(training_data)} training sample")
    with open("training_data_convex.txt", "w") as f:
        f.write("\n".join(training_data))


def pop_random(lst):
    idx = random.randrange(0, len(lst))
    return lst.pop(idx)


def make_pair_sample(keyword2sentence):
    dataset = {"pos": [], "neg": []}
    for keyword, sentences in tqdm(keyword2sentence.items()):
        if keyword != delimiter:
            keyword = keyword.split(delimiter)[-1]
            if len(sentences) > 1:
                while(len(sentences) > 1):
                    template_sent = pop_random(sentences)
                    input_sent = pop_random(sentences)
                    dataset["pos"].append(f"{template_sent}\t{input_sent}\t{keyword}")
        else:
            while(len(sentences) > 1):
                template_sent = pop_random(sentences)
                input_sent = pop_random(sentences)
                dataset["neg"].append(f"{template_sent}\t{input_sent}\t{keyword}")
    return dataset


def preprocess_wiki_raw_text(path):
    output = subprocess.run(
        ["wc", "-l", path], universal_newlines=True, stdout=subprocess.PIPE
    )
    corpus = set()
    with open(path) as f:
        for line in tqdm(f, total=int(output.stdout.split()[0])):
            if not line.startswith("Thể loại"):
                lines = sent_tokenize(line)
                corpus.update(lines)
    with open("data/text_corpus/wiki_sent_segment.txt", "w") as f:
        f.write("\n".join(corpus))


if __name__ == "__main__":
    # preprocess_wiki_raw_text("data/text_corpus/corpus_viwiki.txt")
    prepare_training_data("extract_keywords.txt")
