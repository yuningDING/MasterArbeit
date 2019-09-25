import pandas as pd
import numpy as np
import csv

"""
"split answers in ASAP-SAS based on certain scores
"used in selecting possible target answers or generating input of deep_instance system
"""


input = ""
output = ""

# select possible target answers
target_score={'1': 3, '2': 3, '3': 2, '4': 2, '5': 3, '6': 3, '7': 2, '8': 2, '9': 2, '10': 2}
for i in range(1, 11):
    text_df = pd.read_csv(input, encoding='utf-8', sep='\t', header=0,
                          quoting=csv.QUOTE_NONE,
                          names=['id', 'EssaySet', 'score', 'essay_score2', 'text'],
                          dtype={'id': str, 'EssaySet': str, 'score': np.int32, 'essay_score2': np.int32, 'text': str})
    score3_df = text_df[text_df['EssaySet'] == str(i)]
    score3_df = score3_df[score3_df['essay_score1'] == target_score.get(str(i))]
    score3_df = score3_df[score3_df['essay_score2'] == target_score.get(str(i))]
    score3_df.to_csv(output, sep=',', encoding='utf-8', index=False)

# generate input of deep_instance system
target_score = 0
for i in range(1, 11):
    text_df = pd.read_csv(input, encoding='utf-8', sep='\t', header=0,
                          quoting=csv.QUOTE_NONE,
                          names=['id', 'EssaySet', 'score', 'essay_score2', 'text'],
                          dtype={'id': str, 'EssaySet': str, 'score': np.int32, 'essay_score2': np.int32, 'text': str})

    score0_df = text_df[text_df['EssaySet'] == str(i)]
    score0_df = score0_df.drop('EssaySet', 1)
    score0_df = score0_df.drop('essay_score2',1)
    score0_df.to_csv(output, sep=',', encoding='utf-8', index=False)
