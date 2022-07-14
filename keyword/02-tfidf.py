#!/usr/bin/env python
# coding: utf-8

# # MASKER KEYWORD - TF-IDF

# In[1]:


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


# In[2]:


rm_stopwords      = True
rm_punctuations   = True
save_top_words    = 3000

lower_case        = False # already DONE
tokenizer         = None # use NLTK


# In[3]:


classi_ratio = 1

depression_text = pickle.load(open(f"../data/classi/classi_corpus_traindep_ratio{classi_ratio}.pkl", "rb"))
control_text    = pickle.load(open(f"../data/classi/classi_corpus_traincon_ratio{classi_ratio}.pkl", "rb"))


# In[4]:


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


# In[5]:


depression_text = preprocessing(depression_text)
control_text    = preprocessing(control_text)


# In[6]:


print(len(depression_text))
print(len(control_text))


# In[7]:


import pickle
import numpy as np
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# combine the data into 2 document, one document for depression and one document for control group
def get_tfidf_keyword(depression_text, control_text):

    class_docs = [''] * 2  # concat all texts for each class
    
    # control = class 0
    for text in control_text:
        for word in text:
            word = word + ' '
            class_docs[0] += word
    print(len(class_docs[0]))
    
    # depression = class 1
    for text in depression_text:
        for word in text:
            word = word + ' '
            class_docs[1] += word
    print(len(class_docs[1]))

    tfidf = TfidfVectorizer(ngram_range=(1, 1))
    feat  = tfidf.fit_transform(class_docs).todense()  # (n_classes, vocabs)
    feat  = np.squeeze(np.asarray(feat))  # matrix -> array
            
    # ---------- Control ----------
    con_sorted_idx = feat[0].argsort()[::-1]
    control_keyword = [tfidf.get_feature_names()[idx] for idx in con_sorted_idx]
    # for idx in sorted_idx:
    #     word = tfidf.get_feature_names()[idx]
    #     control_keyword.append(word)

     # ---------- Depression ----------
    dep_sorted_idx = feat[1].argsort()[::-1]
    depression_keyword = [tfidf.get_feature_names()[idx] for idx in dep_sorted_idx]
    # for idx in sorted_idx:
    #     word = tfidf.get_feature_names()[idx]
    #     depression_keyword.append(word)

    return depression_keyword, control_keyword


# In[8]:


depression_keyword, control_keyword = get_tfidf_keyword(depression_text, control_text)


# In[37]:


indep_notincon = [word for word in depression_keyword[:5477] if word not in control_keyword[:5477]]

print(len(indep_notincon))
print(indep_notincon[:10])


# In[38]:


incon_notindep = [word for word in control_keyword[:5477] if word not in depression_keyword[:5477]]

print(len(incon_notindep))
print(incon_notindep[:10])


# In[39]:


with open(f"./02-tfidf-depcon{save_top_words}-R{classi_ratio}-nostops.txt", "w") as f:
    for word in indep_notincon:
        f.write(word + "\n")
    for word in incon_notindep:
        f.write(word + "\n")

