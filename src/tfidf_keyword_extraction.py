import numpy as np
from tqdm import tqdm
from collections import Counter
from CocCocTokenizer import PyTokenizer
import json
from multiprocessing import Pool
import os
import joblib
from underthesea import sent_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


tokenizer = PyTokenizer(load_nontone_data=True)


def keyphrase_score(
    key_phrase: list, len_corpus: int, word_freq: dict, alpha: float = 0.8
):
    sum_ = 0
    for word in key_phrase:
        sum_ += np.log(len_corpus / word_freq.get(word, 1))
    return sum_ / np.power(len_corpus, alpha)


def cal_word_freq(corpus: list, tokenize):
    word_freq = Counter()
    for sentence in tqdm(corpus):
        words = tokenize(sentence)
        word_freq.update(words)
    return dict(word_freq)


def tokenize(sentence):
    words = tokenizer.word_tokenize(sentence, tokenize_option=0)
    # words = sum([w.split("_") for w in words], [])
    # print(words)
    return words


def preprocess_sentence(text):
    text = text.lower()
    words = tokenize(text)
    return " ".join(words)


def get_stopwords(path):
    with open(path) as f:
        stopwords = f.read().split("\n")
    return set([word.strip().replace(" ", "_") for word in stopwords])


def extract_keyword(cv, tfidf_transformer, text, k):
    feature_names = cv.get_feature_names()

    tf_idf_vector = tfidf_transformer.transform(cv.transform([text] if isinstance(text, str) else text))

    sorted_items = sort_coo(tf_idf_vector.tocoo())

    keywords = extrac_topn_from_vector(feature_names, sorted_items, k)

    return keywords


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extrac_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    
    return results


if __name__ == "__main__":
    corpus = set()
    # with open("data/redit_comment/vi_0.json") as f:
    #     for line in f:
    #         line = line.strip()
    #         line = json.loads(line)
    #         corpus.add(line["context"])
    #         corpus.add(line["response"])
    with open("data/text_corpus/corpus_viwiki.txt") as f:
        for line in f:
            line = line.strip()
            line = sent_tokenize(line)
            corpus.update(line)
    
    corpus = list(corpus)
    
    with Pool(os.cpu_count() + 2) as p:
        corpus4tfidf = list(tqdm(p.imap_unordered(preprocess_sentence, corpus),
                            total=len(corpus), desc="Preprocessing corpus"))
    
    stopwords = get_stopwords("data/vi_stopwords.txt")

    cv = CountVectorizer(max_df=0.85, stop_words=stopwords)
    word_count_vector = cv.fit_transform(corpus)

    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(word_count_vector)

    joblib.dump(cv, "cv.pkl")
    joblib.dump(tfidf_transformer, "tfidf_transformer.pkl")


    # extract keyword
    # text = corpus[10]
    text = [" ".join(tokenize(text)) for text in corpus[10:20]]
    print(text)
    cv = joblib.load("cv.pkl")
    tfidf_transformer = joblib.load("tfidf_transformer.pkl")
    keyowrds = extract_keyword(cv, tfidf_transformer, text, 10)
    print(keyowrds)

