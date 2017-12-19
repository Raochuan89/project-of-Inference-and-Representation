from __future__ import print_function

import os
import pickle

import numpy as np
import pandas as pd
import torch.utils.data as data
from google_drive_downloader import GoogleDriveDownloader as gdd


class NYT(data.Dataset):
    """new york times Dataset.

    Attributes:
        idx_to_word: the map between index and word
    """

    def __init__(self):
        self.gdrive_ids = '1-jkltSKoBuw3GsLkiGdiCdlmDeIhD6Xq'
        self.dic_word = '1H9-YIIh8a2geTfbDPmH04U3vJHZT0YBY'
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

        gdd.download_file_from_google_drive(file_id=self.gdrive_ids,
                                            dest_path='./nyt/nyt_8447.zip',
                                            unzip=True)

        gdd.download_file_from_google_drive(file_id=self.dic_word,
                                            dest_path='./nyt/word_dic_nyt.pkl',
                                            unzip=False)

        print('Downloading Done!')

        os.remove('./nyt/nyt_8447.zip')
