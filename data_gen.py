# encoding=utf-8

import itertools

import jieba
import numpy as np
from torch.utils.data import Dataset

from config import *


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in jieba.cut(sentence)] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Returns padded input sequence tensor and lengths
def inputVar(indexes_batch):
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(indexes_batch):
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(pair_batch):
    pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch)
    output, mask, max_target_len = outputVar(output_batch)
    return inp, lengths, output, mask, max_target_len


class ChatbotDataset(Dataset):
    def __init__(self, split):
        self.split = split
        assert self.split in {'train', 'valid'}

        print('loading {} samples'.format(split))
        samples_path = 'data/samples.json'
        self.samples = json.load(open(samples_path, 'r'))

        if split == 'train':
            self.samples = self.samples[:num_training_samples]
        else:
            self.samples = self.samples[num_training_samples:]

    def __getitem__(self, idx):
        i = idx * batch_size
        length = min(batch_size, (len(self.samples) - i))

        pair_batch = []

        for i_batch in range(length):
            sample = self.samples[i + i_batch]
            pair_batch.append((sample['input'], sample['output']))

        return batch2TrainData(pair_batch)

    def __len__(self):
        return int(np.ceil(len(self.samples) / float(batch_size)))
