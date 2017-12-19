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


class NYT(data.Dataset):
    """new york times Dataset.

    Attributes:
        idx_to_word: the map between index and word
    """
    gdrive_ids = '1-jkltSKoBuw3GsLkiGdiCdlmDeIhD6Xq'
    dic_word = '1H9-YIIh8a2geTfbDPmH04U3vJHZT0YBY'

    def __init__(self):

        self.download()

        self.idx_to_word = pickle.load(open('./nyt/word_dic.pkl', 'rb'))
 

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sentence, index) 
            sentence is one document in the format of list of integers where each integer represents a word.
            index is the index of the document
        """
        sentence = pickle.load(open('./nyt/{}.pkl'.format(index), 'rb'))
        

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image


        return sentence, index

    def __len__(self):

        return 8447


    def download(self):
          """Download the new york times data and the index to word dictionary."""
       
        gdd.download_file_from_google_drive(file_id=gdrive_ids,
                                        dest_path='./nyt/nyt_8447.zip',
                                        unzip=True)

        gdd.download_file_from_google_drive(file_id=dic_word,
                                        dest_path='./nyt/word_dic_nyt.pkl',
                                        unzip=False)

        print('Downloading Done!')

        os.remove('./nyt/nyt_8447.zip')