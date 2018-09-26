import json
from collections import Counter

import jieba
from tqdm import tqdm

from config import *


def encode_text(word_map, c):
    return [word_map.get(word, word_map['<unk>']) for word in c] + [word_map['<end>']]


def build_wordmap():
    with open(corpus_loc, 'r') as f:
        sentences = f.readlines()

    sentences = [s[2:] for s in sentences if len(s[1:].strip()) > 0]

    word_freq = Counter()

    for sentence in tqdm(sentences):
        seg_list = jieba.cut(sentence)
        # Update word frequency
        word_freq.update(list(seg_list))

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 4 for v, k in enumerate(words)}
    word_map['<pad>'] = 0
    word_map['<start>'] = 1
    word_map['<end>'] = 2
    word_map['<unk>'] = 3
    print(len(word_map))
    print(words[:10])

    with open('data/WORDMAP.json', 'w') as file:
        json.dump(word_map, file, indent=4)


def build_samples():
    word_map = json.load(open('data/WORDMAP.json', 'r'))

    with open(corpus_loc, 'r') as f:
        sentences = f.readlines()
    print(len(sentences))

    sentences = [s[2:].strip() for s in sentences if len(s[1:].strip()) > 0]
    print(len(sentences))

    print('building samples')
    samples = []
    for i in tqdm(range(0, len(sentences) - 1, 2)):
        sentence_q = sentences[i]
        seg_list = jieba.cut(sentence_q)
        input_zh = encode_text(word_map, list(seg_list))
        sentence_a = sentences[i + 1]
        seg_list = jieba.cut(sentence_a)
        output_zh = encode_text(word_map, list(seg_list))
        if len(input_zh) <= max_len and len(output_zh) <= max_len:
            samples.append({'input': list(input_zh), 'output': list(output_zh)})

    filename = 'data/samples.json'
    with open(filename, 'w') as f:
        json.dump(samples, f, indent=4)
    print('{} samples created at: {}.'.format(len(samples), filename))


if __name__ == '__main__':
    build_wordmap()
    build_samples()
