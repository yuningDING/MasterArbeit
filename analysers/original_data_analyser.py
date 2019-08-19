import pandas as pd
import csv
import numpy as np
from nltk.tokenize import RegexpTokenizer
import re

# original dataset
for i in range(1, 11):
    file_path = 'resources/asap_withoutBG_prompt_' + str(i) + '.txt'
    df = pd.read_csv(file_path, encoding='utf-8', sep='\t', header=0,
                     quoting=csv.QUOTE_NONE,
                     names=['Id', 'EssayText'],
                     dtype={'Id': str, 'EssayText': str})

    tokenizer = RegexpTokenizer(r'\w+')
    df['token_list'] = df.apply(lambda row: tokenizer.tokenize(row['EssayText'].lower()), axis=1)
    df['token_number'] = df.apply(lambda row: len(row['token_list']), axis=1)
    df['char_number'] = df.apply(lambda row: len(re.sub(r'[^\w\s]','',row['EssayText'])), axis=1)
    df['word_length'] = df.apply(lambda row: row['char_number'] / row['token_number'],axis=1)

    # for index, row in df.iterrows():
    #     # length related metrics
    #     df.loc[index,'token_list'] = row['EssayText'].split(' ')
    #     df.loc[index,'token_number'] = len(row['token_list'])
    #     df.loc[index,'char_number'] = len(row['EssayText'].replace(" ", ""))
    #     df.loc[index,'word_length'] = row['char_number'] / row['token_number']

    print('Prompt ' + str(i) + ":")
    print('average sentence length is ' + str(df['token_number'].mean()) + ', standard deviation is ' + str(
        df.loc[:, 'token_number'].std()))
    print('average word length is ' + str(df['word_length'].mean()) + ', standard deviation is ' + str(
        df.loc[:, 'word_length'].std()))
