# encoding=utf-8
import zipfile
import os
from config import *
import urllib.request
from utils import ensure_folder


def extract(download_url, folder):
    filename = '{}.zip'.format(folder)
    download_corpus(download_url, filename)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


def download_corpus(download_url, filename):
    exists = os.path.isfile(filename)
    if not exists:
        print('Downloading {}...'.format(filename))
        urllib.request.urlretrieve(download_url, filename) 


if __name__ == '__main__':
    ensure_folder('data')

    if not os.path.isdir(xhj_corpus_loc):
        extract(xhj_corpus_url, xhj_corpus_loc)
    
    download_corpus(ptt_corpus_url, ptt_corpus_loc)

