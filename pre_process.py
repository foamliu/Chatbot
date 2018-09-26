import json
from collections import Counter

import jieba

from config import *


def build_wordmap():
    with open(corpus_loc, 'r') as f:
        sentences = f.readlines()

    sentences = [s[2:] for s in sentences if len(s[1:].strip()) > 0]

    word_freq = Counter()

    for sentence in sentences:
        seg_list = jieba.cut(sentence)
        # Update word frequency
        word_freq.update(list(seg_list))

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 2 for v, k in enumerate(words)}
    word_map['<start>'] = 0
    word_map['<end>'] = 1
    print(len(word_map))
    print(words[:10])

    with open('data/WORDMAP.json', 'w') as file:
        json.dump(word_map, file, indent=4)


def build_samples():
    with open(corpus_loc, 'r') as f:
        sentences = f.readlines()

    sentences = [s[2:] for s in sentences if len(s[1:].strip()) > 0]

    print(sentences[0])
    print(sentences[1])
    print(sentences[2])
    print(sentences[3])


if __name__ == '__main__':
    build_wordmap()
    build_samples()
