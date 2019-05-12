import pandas as pd
import numpy as np
import csv

text_df = pd.read_csv('resources/test_public_repaired.txt', encoding='utf-8', sep='\t', header=0, quoting=csv.QUOTE_NONE,
                        names=['Id', 'EssaySet', 'essay_score1', 'essay_score2', 'EssayText'],
                        dtype={'Id': str, 'EssaySet': str, 'essay_score1': np.int32, 'essay_score2': str, 'EssayText': str})
score0_df = text_df[text_df['essay_score1'] == 0]
score0_df.to_csv('outputs/test_score0.txt',sep='\t', encoding='utf-8', index=False)

target_score={'1': 3, '2': 3, '3': 2, '4': 2, '5': 3, '6': 3, '7': 2, '8': 2, '9': 2, '10': 2}
text_df = pd.read_csv('resources/train_plus_test_repaired.txt', encoding='utf-8', sep='\t', header=0, quoting=csv.QUOTE_NONE,
                        names=['Id', 'EssaySet', 'essay_score1', 'essay_score2', 'EssayText'],
                        dtype={'Id': str, 'EssaySet': str, 'essay_score1': np.int32, 'essay_score2': str, 'EssayText': str})
for i in range(1, 11):
    score3_df = text_df[text_df['EssaySet'] == str(i)]
    score3_df = score3_df[score3_df['essay_score1'] == target_score.get(str(i))]
    score3_df.to_csv('outputs/TA_prompt_'+str(i)+'.txt', sep='\t', encoding='utf-8', index=False)
