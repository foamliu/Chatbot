# encoding=utf-8

import itertools

import jieba
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
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
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

        self.dataset_size = len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        inp, lengths, output, mask, max_target_len = batch2TrainData(voc, [[sample['input'], sample['output']]])
        return inp[0], lengths[0], output[0], mask[0], max_target_len[0]

    def __len__(self):
        return self.dataset_size
