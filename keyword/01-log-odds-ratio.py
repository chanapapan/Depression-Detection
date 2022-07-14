#!/usr/bin/env python
# coding: utf-8

# # KE-MLM - Log-odd-ratio
# 
# - use domain dataset as background corpus
# - use depression as corpus_i
# - collect top and bottom 500 words

# In[1]:


rm_punctuations   = True
rm_stopwords      = True
save_top_words    = 1500

lower_case        = False # already DONE
tokenizer         = None # use NLTK


# In[2]:


#####################################################################
# LogOddsRatio Class
# 
# A class for computing Log-odds-ratio with informative Dirichlet priors
#
# See http://languagelog.ldc.upenn.edu/myl/Monroe.pdf for more detail
# 
#####################################################################

__author__ = "Kornraphop Kawintiranon"
__email__ = "kornraphop.k@gmail.com"

import math
from loguru import logger
import tqdm
import numpy as np
import pandas as pd
import argparse

import sys, os
sys.path.append('..')

os.environ['TRANSFORMERS_CACHE'] = './cache/'

from transformers import Trainer, TrainingArguments

import json, pickle
import torch
from torch import nn
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from transformers import BertTokenizerFast, AutoModel, AutoModelForSequenceClassification, DataCollatorWithPadding
import numpy as np

from src.dataset import *
from src.utils   import *
# from src.models  import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = get_freer_gpu()
print('device', device)


# In[3]:


import string
import tqdm
import re
import concurrent.futures
import multiprocessing
from nltk.corpus import stopwords
from nltk.tokenize.destructive import NLTKWordTokenizer

def parallel_tokenize(corpus, tokenizer=None, n_jobs=-1):
    if tokenizer == None:
        tokenizer = NLTKWordTokenizer()
    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count() - 1
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        corpus_tokenized = list(
            tqdm.tqdm(executor.map(tokenizer.tokenize, corpus, chunksize=200), total=len(corpus), desc='Tokenizing')
        )
    return corpus_tokenized

def remove_stopwords(corpus, language='english'):
    stop_words = set(stopwords.words(language))
    processed_corpus = []
    for words in corpus:
        
        # print(words[:100])
        words = [w for w in words if not w in stop_words]
        # print(words[:100])
        # asdfasfasdf
        processed_corpus.append(words)
    return processed_corpus

def remove_punctuations(corpus):
    punctuations = string.punctuation
    processed_corpus = []
    for words in corpus:
        # remove single punctuations
        words = [w for w in words if not w in punctuations]
        words = [re.sub(r"""[()#[\]#*+\-/:;<=>@[\]^_`{|}~"\\.?!$%&]""", "", w) for w in words]      
        processed_corpus.append(words)
    return processed_corpus
    
def decontract(corpus):
    processed_corpus = []
    for phrase in tqdm.tqdm(corpus, desc="Decontracting"):
        phrase = re.sub(r"’", "\'", phrase)

        # specific
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)

        processed_corpus.append(phrase)
    return processed_corpus

def get_word_counts(corpus):
    # Initializing Dictionary
    d = {}

    # Counting number of times each word comes up in list of words (in dictionary)
    for words in tqdm.tqdm(corpus, desc="Word Counting"):
        for w in words:
            d[w] = d.get(w, 0) + 1
    return d


# In[5]:


class LogOddsRatio:
    """
    Log-odds-ratio with informative Dirichlet priors
    """

    def __init__(self, corpus_i, corpus_j, background_corpus=None, lower_case=True, rm_stopwords=True, rm_punctuations=True, tokenizer=None):
        """
        Create a class object and prepare word counts for log-odds-ratio computation
        Args:
            corpus_i:        A list of documents, each contains a string
            corpus_j:        A list of documents, each contains a string
            background_corpus (default = None): If None, it will be assigned to a concatenation of `corpus_i` and `corpus_j`
            rm_stopwords:    Whether remove stopwords in preprocessing step
            tokenizer:       To specify a specific tokenizer for tokenization step
        """

        def preprocessing(corpus):
            if lower_case:
                print("lowercasing")
                corpus = [text.lower() for text in corpus]
            corpus = decontract(corpus)
            tokenized_corpus = parallel_tokenize(corpus, tokenizer)

            if rm_stopwords:
                print("removing stopwords")
                tokenized_corpus = remove_stopwords(tokenized_corpus)

            if rm_punctuations:
                print("removing punctuation")
                tokenized_corpus = remove_punctuations(tokenized_corpus)

            print(tokenized_corpus[0][:500])

            return tokenized_corpus

        # Convert a list of string into a list of lists of words
        logger.info("Preprocessing corpus-i")
        corpus_i = preprocessing(corpus_i)
        logger.info("Preprocessing corpus-j")
        corpus_j = preprocessing(corpus_j)
        if background_corpus != None:
            logger.info("Preprocessing corpus-background")
            background_corpus = preprocessing(background_corpus)
        
        # Compute word counts of every words on each corpus separately
        logger.info("Getting word counts from corpus-i")
        self.y_i = get_word_counts(corpus_i)
        logger.info("Getting word counts from corpus-j")
        self.y_j = get_word_counts(corpus_j)
        logger.info("Getting word counts from corpus-background")
        if background_corpus:
            self.alpha = get_word_counts(background_corpus)
        else:
            # Combine words and sum their counts of corpus i and j in case no specified background corpus
            self.alpha = {k: self.y_i.get(k, 0) + self.y_j.get(k, 0) for k in set(self.y_i) | set(self.y_j)}

        # Sort dicts
        logger.debug("Start sorting and backing up to files")
        self.y_i = {k: v for k, v in sorted(self.y_i.items(), key=lambda item: item[1], reverse=True)}
        self.y_j = {k: v for k, v in sorted(self.y_j.items(), key=lambda item: item[1], reverse=True)}
        self.alpha = {k: v for k, v in sorted(self.alpha.items(), key=lambda item: item[1], reverse=True)}

        # Write to files as backup
        with open("vocabs_i.txt", "w") as f:
            for k, v in self.y_i.items():
                f.write(f"{k},{v}\n")
        with open("vocabs_j.txt", "w") as f:
            for k, v in self.y_j.items():
                f.write(f"{k},{v}\n")
        with open("vocabs_alpha.txt", "w") as f:
            for k, v in self.alpha.items():
                f.write(f"{k},{v}\n")

        # Initialize necessary variables
        self.delta = None
        self.sigma_2 = None
        self.z_scores = None

        # Compute
        logger.info("Start computing delta")
        self._compute_delta()
        logger.info("Start computing sigma^2")
        self._compute_sigma_2()
        logger.info("Start computing Z-score")
        self._compute_z_scores()

        # Sort dicts
        logger.debug("Start sorting and backing up to files")
        self.delta = {k: v for k, v in sorted(self.delta.items(), key=lambda item: item[1], reverse=True)}
        self.sigma_2 = {k: v for k, v in sorted(self.sigma_2.items(), key=lambda item: item[1], reverse=True)}
        self.z_scores = {k: v for k, v in sorted(self.z_scores.items(), key=lambda item: item[1], reverse=True)}

        # Write to files as backup
        with open("delta.txt", "w") as f:
            for k, v in self.delta.items():
                f.write(f"{k},{v}\n")
        with open("sigma_2.txt", "w") as f:
            for k, v in self.sigma_2.items():
                f.write(f"{k},{v}\n")
        with open("z_scores.txt", "w") as f:
            for k, v in self.z_scores.items():
                f.write(f"{k},{v}\n")


    def _compute_delta(self):
            """ The usage difference for word w among two corpora i and j
            """
            self.delta = dict()
            n_i = sum(self.y_i.values())
            n_j = sum(self.y_j.values())
            alpha_zero = sum(self.alpha.values())
            logger.debug(f"Size of corpus-i: {n_i}")
            logger.debug(f"Size of corpus-j: {n_j}")
            logger.debug(f"Size of background corpus: {alpha_zero}")

            try:
                for w in set(self.y_i) | set(self.y_j): # iterate through all words among two corpora

                    # print(self.y_i.get(w, 0))
                    # print(self.alpha.get(w, 0))
                    # print(n_i + alpha_zero - self.y_i.get(w, 0) - self.alpha.get(w, 0))

                    first_top    = self.y_i.get(w, 0) + self.alpha.get(w, 0)
                    first_bottom = n_i + alpha_zero - self.y_i.get(w, 0) - self.alpha.get(w, 0)

                    second_top    = self.y_j.get(w, 0) + self.alpha.get(w, 0)
                    second_bottom = n_j + alpha_zero - self.y_j.get(w, 0) - self.alpha.get(w, 0)


                    if first_bottom == 0 and second_bottom == 0:
                        first_log  = 0
                        second_log = 0

                    if first_bottom == 0 and second_bottom != 0:
                        first_log  = 0
                        second_log = math.log10( second_top / second_bottom )

                    if second_bottom == 0:
                        first_log = math.log10( first_top / first_bottom )
                        second_log = 0


                    if first_bottom != 0 and second_bottom != 0:
                        if (first_top / first_bottom) == 0 and (second_top / second_bottom) != 0:
                            first_log = 0
                            second_log = math.log10( second_top / second_bottom )

                        if (first_top / first_bottom) != 0 and (second_top / second_bottom) == 0:
                            first_log  = math.log10( first_top  / first_bottom )
                            second_log = 0

                        if (first_top / first_bottom) != 0 and (second_top / second_bottom) != 0:
                            first_log  = math.log10( first_top  / first_bottom )
                            second_log = math.log10( second_top / second_bottom )

                    self.delta[w] = first_log - second_log

            except ValueError as e:
                logger.debug(f"Y-i of the word {w}:", self.y_i.get(w, 0))
                logger.debug(f"alpha of the word {w}:", self.alpha.get(w, 0))
                logger.debug(f"value:", (self.y_i.get(w, 0) + self.alpha.get(w, 0)) /
                      (n_i + alpha_zero - self.y_i.get(w, 0) - self.alpha.get(w, 0)))
                raise e

    def _compute_sigma_2(self):
        """ Compute estimated values of sigma squared
        """
        self.sigma_2 = dict()
        for w in self.delta:
            if (self.y_i.get(w, 0) + self.alpha.get(w, 0)) == 0 or (self.y_j.get(w, 0) + self.alpha.get(w, 0)) == 0:
                self.sigma_2[w] = 0
            else:
                self.sigma_2[w] = (1 / (self.y_i.get(w, 0) + self.alpha.get(w, 0))) + (1 / (self.y_j.get(w, 0) + self.alpha.get(w, 0)))

    def _compute_z_scores(self):
        self.z_scores = dict()
        for w in self.delta:
            if self.sigma_2.get(w, 0) == 0:
                # score 0 is in the middle so it will not show up in top or bottom which is what we want!
                self.z_scores[w] = 0
            else:
                self.z_scores[w] = self.delta.get(w, 0) / math.sqrt(self.sigma_2.get(w, 0))


# # WITH BG CORPUS

# In[6]:


background_corpus = pickle.load(open("../data/domain/domain_corpus_traindepcon_ratio10.pkl", "rb")) 


# In[7]:


classi_ratio = 1

all_train_depression_text = pickle.load(open(f"../data/classi/classi_corpus_traindep_ratio{classi_ratio}.pkl", "rb"))
all_train_control_text    = pickle.load(open(f"../data/classi/classi_corpus_traincon_ratio{classi_ratio}.pkl", "rb"))


# In[8]:


# DEPRESSION = i
corpus_i  = all_train_depression_text
corpus_j  = all_train_control_text
background_corpus = background_corpus

log_odds_ratio = LogOddsRatio(corpus_i          = corpus_i,
                              corpus_j          = corpus_j, 
                              background_corpus = background_corpus,
                              lower_case        = lower_case, 
                              rm_stopwords      = rm_stopwords, 
                              rm_punctuations   = rm_punctuations, 
                              tokenizer         = None)

# Save top words into a file
if save_top_words != None and save_top_words > 0:
    if save_top_words > len(log_odds_ratio.z_scores):
        raise ValueError("--save_top_words must be less than or equal to vocab size")

    logger.info(f"Saving top and bottom {save_top_words} words ranked by Z-score")
    tops    = list(log_odds_ratio.z_scores.keys())[:save_top_words]
    bottoms = list(log_odds_ratio.z_scores.keys())[-save_top_words:]

    with open(f"./01-logodds-topbot{save_top_words}-R{classi_ratio}-nostops.txt", "w") as f:
        for word in tops:
            f.write(word + "\n")
        for word in bottoms:
            f.write(word + "\n")

