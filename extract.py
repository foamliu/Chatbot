# encoding=utf-8
import zipfile
import os
from config import *
import urllib.request
from utils import ensure_folder


def extract(download_url, folder):
    filename = '{}.zip'.format(folder)
    exists = os.path.isfile(filename)
    if not exists:
        print('Downloading {}...'.format(filename))
        urllib.request.urlretrieve(download_url, filename) 
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


if __name__ == '__main__':
    ensure_folder('data')

    if not os.path.isdir(corpus_loc):
        extract(corpus_url, corpus_loc)

