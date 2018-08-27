# !/usr/bin/python
# -*- coding: utf-8 -*-
import codecs
import numpy as np

from wmd import WMD

__author__ = 'xuejiao'

vocabulary_min = 3


def load_w2v(dim=256):
    w2v = {}
    for line in codecs.open("/data1/xuejiao/data/embedding/vectors.20170613.w2v", 'r', 'utf-8'):
        line = line.strip().split()
        if len(line) == dim + 1:
            w2v[line[0]] = np.array([float(x) for x in line[1:]]).astype(np.float32)
    print("load w2v size: {}".format(len(w2v)))
    return w2v


def load_stopwords():
    stopwords = []
    for line in codecs.open("/data1/xuejiao/znlp/data/all.stopword", 'r', 'utf-8'):
        line = line.rstrip()
        if len(line) > 0:
            stopwords.append(line)
    print("load stopwords size: {}".format(len(stopwords)))
    return stopwords


def load_question(top=-1):
    questions = []
    index = 0
    for line in codecs.open("/data1/xuejiao/data/duplicate/redirect_question_token.txt", 'r', 'utf-8'):
        if top != -1 and index > top:
            break
        try:
            index += 1
            pieces = line.strip().split("\t")
            questions.append(pieces[1])
            questions.append(pieces[3])
        except:
            pass
    return questions


def parse_nbow(text):
    fine = text.split(",")
    count = float(len(fine))
    count_map = {}
    for w in fine:
        if w in count_map:
            count_map[w] += 1
        else:
            count_map[w] = 1
    for w, c in count_map.items():
        count_map[w] = c / count
    return count_map


def prepare():
    stopwords = load_stopwords()
    w2v = load_w2v()
    w2id = {}
    id2vec = []
    for index, (w, vec) in enumerate(w2v.items()):
        w2id[w] = index
        id2vec.append(vec)

    nbow = {}
    questions = load_question()
    for index, q in enumerate(questions):
        count_map = parse_nbow(q)
        id_count_map = {w2id[w]: v for w, v in count_map.items() if w in w2v and w not in stopwords}
        if len(id_count_map) < vocabulary_min:
            continue
        items = id_count_map.items()
        nbow[q] = (str(index),
                   [x[0] for x in items],
                   np.array([x[1] for x in items], dtype=np.float32))

    calc = WMD(np.array(id2vec), nbow, vocabulary_min=vocabulary_min)
    for q in nbow.keys():
        print("q: {} neighbors: {}\n".format(q, calc.nearest_neighbors(q, k=3)))


def test():
    embeddings = np.array([[0.1, 1], [1, 0.1]], dtype=np.float32)
    nbow = {"first": ("#1", [0, 1], np.array([1.5, 0.5], dtype=np.float32)),
            "second": ("#2", [0, 1], np.array([0.75, 0.15], dtype=np.float32))}
    calc = WMD(embeddings, nbow, vocabulary_min=2)
    print(calc.nearest_neighbors("first"))


if __name__ == '__main__':
    prepare()
