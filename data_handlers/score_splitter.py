import pandas as pd
import numpy as np
import csv


# target_score={'1': 3, '2': 3, '3': 2, '4': 2, '5': 3, '6': 3, '7': 2, '8': 2, '9': 2, '10': 2}
target_score = 0
# text_df = pd.read_csv('outputs/char_bunch/char_promp.txt', encoding='utf-8', sep='\t', header=0, quoting=csv.QUOTE_NONE,
#                         names=['id', 'EssaySet', 'score', 'essay_score2', 'text'],
#                         dtype={'id': str, 'EssaySet': str, 'score': np.int32, 'essay_score2': np.int32, 'text': str})
for i in range(1, 11):
    text_df = pd.read_csv('resources/gpt-2/gpt2_prompt_'+str(i)+'.txt', encoding='utf-8', sep='\t', header=0,
                          quoting=csv.QUOTE_NONE,
                          names=['id', 'EssaySet', 'score', 'essay_score2', 'text'],
                          dtype={'id': str, 'EssaySet': str, 'score': np.int32, 'essay_score2': np.int32, 'text': str})
    # score3_df = text_df[text_df['EssaySet'] == str(i)]
    # score3_df = score3_df[score3_df['essay_score1'] == target_score]
    # score3_df = score3_df[score3_df['essay_score2'] == target_score.get(str(i))]
    score0_df = text_df[text_df['EssaySet'] == str(i)]
    #score0_df = score0_df[score0_df['score'] == target_score]
    score0_df = score0_df.drop('EssaySet', 1)
    score0_df = score0_df.drop('essay_score2',1)
    score0_df.to_csv('outputs/neural-sas_input/asap2_item'+str(i)+'.test_gpt2.csv', sep=',', encoding='utf-8', index=False)
