from transformers import AutoTokenizer


def aligment(text_path, tokenizer):
    with open(text_path) as f:
        data = f.read().strip()
    for line in data:
        if line:
            parts = line.split()
            c = parts[0]
            tag = parts[-1]
            