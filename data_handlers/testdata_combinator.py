import pandas as pd
import numpy as np
import csv

"""
"To avoid learning model multiple times,
"character/token based generated test data for each prompt will be combined into one file.
"""

for i in range(1, 11):
    df = pd.DataFrame(columns=['Id', 'EssaySet', 'essay_score1', 'essay_score2', 'EssayText'])

    for n in range(1,6):
        print(n)
        #text_df = pd.read_csv('outputs/char_1-5gram/char_' + str(n) + '_gram_prompt_' + str(i) + '_1000.txt',
        text_df = pd.read_csv('outputs/token_1-5gram/token_'+str(n)+'_gram_prompt_'+str(i)+'_1000.txt',
                              encoding='utf-8', sep='\t', header=0, quoting=csv.QUOTE_NONE,
                              names=['Id', 'EssaySet', 'essay_score1', 'essay_score2', 'EssayText'],
                              dtype={'Id': str, 'EssaySet': str, 'essay_score1': np.int32, 'essay_score2': np.int32, 'EssayText': str})
        text_df['Id'] = str(n) + '-' + text_df['Id'].astype(str)
        df = pd.concat([df, text_df], ignore_index=True)

    # df.to_csv('outputs/char_bunch/char_prompt_' + str(i) + '.txt', sep='\t', encoding='utf-8', index=False)
    df.to_csv('outputs/token_bunch/token_prompt_' + str(i) + '.txt', sep='\t', encoding='utf-8', index=False)
