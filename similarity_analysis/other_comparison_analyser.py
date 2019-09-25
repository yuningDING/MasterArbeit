import pandas as pd
import csv
import numpy as np
from nltk.tokenize import RegexpTokenizer
import re


def intersection(lst1, lst2):
    return len(list(set(lst1) & set(lst2)))


for i in range(1, 11):
    # read original data set
    file_path = 'resources/asap_withoutBG_prompt_' + str(i) + '.txt'
    orig_df = pd.read_csv(file_path, encoding='utf-8', sep='\t', header=0,
                          quoting=csv.QUOTE_NONE,
                          names=['Id', 'EssayText'],
                          dtype={'Id': str, 'EssayText': str})
    tokenizer = RegexpTokenizer(r'\w+')
    orig_df['token_list'] = orig_df.apply(lambda row: tokenizer.tokenize(row['EssayText'].lower()), axis=1)

    # read data generated
    # file_path = 'outputs/GPT2/gpt2_prompt_' + str(i) + '.txt'
    # file_path = 'outputs/content_substitution/content_substitution_prompt_' + str(i) + '.txt'
    file_path = 'outputs/random_ordering/random_ordering_prompt_' + str(i) + '.txt'
    df = pd.read_csv(file_path, encoding='utf-8', sep='\t', header=0,
                     quoting=csv.QUOTE_NONE,
                     names=['Id', 'EssaySet', 'essay_score1', 'essay_score2', 'EssayText'],
                     dtype={'Id': str, 'EssaySet': str, 'essay_score1': np.int32, 'essay_score2': np.int32,
                            'EssayText': str})

    df['token_list'] = df.apply(lambda row: tokenizer.tokenize(row['EssayText'].lower()), axis=1)
    df['token_number'] = df.apply(lambda row: len(row['token_list']), axis=1)
    df['char_number'] = df.apply(lambda row: len(re.sub(r'[^\w\s]','',row['EssayText'])), axis=1)
    df['word_length'] = df.apply(lambda row: row['char_number'] / row['token_number'], axis=1)

    df['words_overlap'] = df.apply(
        lambda row: intersection(orig_df[orig_df.Id == row.Id].iloc[0]['token_list'], row['token_list']) /
                    (len(set(row['token_list']))+len(set(orig_df[orig_df.Id == row.Id].iloc[0]['token_list']))), axis=1)

    print('Prompt ' + str(i) + ":")
    print('average sentence length is ' + str(df['token_number'].mean()) + ', standard deviation is ' + str(
        df.loc[:, 'token_number'].std()))
    print('average word length is ' + str(df['word_length'].mean()) + ', standard deviation is ' + str(
        df.loc[:, 'word_length'].std()))
    print('average word overlap is ' + str(df['words_overlap'].mean()) + ', standard deviation is ' + str(
        df.loc[:, 'words_overlap'].std()))
