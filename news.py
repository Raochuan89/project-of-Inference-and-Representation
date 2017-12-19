from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import errno
import torch
import codecs
from google_drive_downloader import GoogleDriveDownloader as gdd
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from scipy.special import digamma
import warnings
from random import shuffle
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import os


class NEWS(data.Dataset):
    """news by category Dataset.

    Attributes:
        idx_to_word: the map between index and word
    """
    

    def __init__(self):
        self.gdrive_ids = '1VmK6HE7ZohKoJheNkVZXvFEym1OYtt_z'
        self.dic_word = '1E_QzruZO688uatG74gq6PembfiMtOFmk'

        self.download()

        self.idx_to_word = pickle.load(open('./news/word_dic.pkl', 'rb'))
 

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sentence, index) 
            sentence is one document in the format of list of integers where each integer represents a word.
            index is the index of the document
        """

        sentence = pickle.load(open('./news/{}.pkl'.format(index), 'rb'))


        return sentence, index

    def __len__(self):

        return 6000


    def download(self):
        """Download the news by categorical data and the index to word dictionary."""
       
        gdd.download_file_from_google_drive(file_id=self.gdrive_ids,
                                        dest_path='./news/news_6000.zip',
                                        unzip=True)

        gdd.download_file_from_google_drive(file_id=self.dic_word,
                                        dest_path='./news/word_dic.pkl',
                                        unzip=False)

        print('Downloading Done!')
        os.remove('./news/news_6000.zip')