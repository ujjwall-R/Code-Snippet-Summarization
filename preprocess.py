# making a wrapper for preprocessing and tokenization
# check 

import pandas as pd
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle

def clean_text(text):
    text = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', str(text))).split()
    text = ' '.join(text)
    text = re.sub("(\\t|\\r|\\n)", " ", str(text)).lower()
    text = re.sub(r"[<>()|&©ø\[\]\'\",.\}`$\{;@?~*!+=_\//1234567890%]", " ", str(text)).lower()
    text = re.sub(r"\b(\w+)(?:\W+\1\b)+", "", str(text)).lower()
    text = re.sub("(\.\s+|\-\s+|\:\s+)", " ", str(text)).lower()
    text = re.sub("(\s+)", " ", str(text)).lower()
    text = re.sub("(\s+.\s+)", " ", str(text)).lower()
    # text = re.sub("(\s.\s)", " ", str(text)).lower()

    return text

def preprocess(df, max_code_len, max_summary_len):
    '''
    Args:
        df: dataframe with 'code' and 'docstring' columns
        max_code_len: maximum length of code sequence
        max_summary_len: maximum length of summary sequence

    Returns:
        Preprocessed dataframe with added 'sostok' and 'eostok' to docstring
    '''
    
    df.dropna(inplace=True)
    
    df['code'] = df['code'].apply(clean_text)
    df['docstring'] = df['docstring'].apply(clean_text)
    
    # Fitlter out greater than max length rows
    df = df[(df['code'].str.split().str.len() <= max_code_len) & (df['docstring'].str.split().str.len() <= max_summary_len)]

    # Add 'sostok' and 'eostok' to docstring
    df['docstring'] = 'sostok ' + df['docstring'] + ' eostok'

    # Filter out rows with empty docstrings
    df = df[df['docstring'].str.split().str.len() > 2]
    
    return df



def tokenize(texts, max_pad_len, tokenizer_path, thresh=2, fit_on_texts=True):
    '''
    Tokenizes the texts and pads the sequences.

    Args:
        texts: list of strings (code/docstring)
        max_pad_len: maximum length of padded sequence
        tokenizer_path: path to save/load tokenizer object
        thresh: infrequent words threshold
        tokenizer: tokenizer object (optional)
        fit_on_texts: whether to load/save tokenizer object

    Returns:
        text_padded: padded sequences
    '''
    
    if fit_on_texts:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        
        # Remove infrequent words
        total_cnt = len(tokenizer.word_index)
        cnt_infrequent = sum(1 for count in tokenizer.word_counts.values() if count < thresh)
        num_words = total_cnt - cnt_infrequent
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(texts)
            
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        
    # Convert texts to sequences
    text_seqs = tokenizer.texts_to_sequences(texts)
    text_padded = pad_sequences(text_seqs, maxlen=max_pad_len, padding='post')
    
    return text_padded