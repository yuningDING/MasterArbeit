import pandas as pd
import numpy as np
import csv

writer = pd.ExcelWriter('outputs/char_1-5gram_experiment_result.xlsx', engine='xlsxwriter')
# read generated answer
for i in range(1, 11):
    for n in range(1, 6):
        print('prompt: '+str(i)+'\nchar: '+str(n))
        text_df = pd.read_csv('outputs/char_' + str(n) + '_gram_prompt_' + str(i) + '_1000.txt',
                              encoding='utf-8', sep='\t', header=0, quoting=csv.QUOTE_NONE,
                              names=['Id', 'EssaySet', 'essay_score1', 'essay_score2', 'EssayText'],
                              dtype={'Id': str, 'EssaySet': str, 'essay_score1': np.int32, 'essay_score2': str, 'EssayText': str})
        score_df = pd.read_csv('id2outcome/char_1-5gram/p'+str(i)+'_c'+str(n)+'.txt',
                               encoding='utf-8', sep='\t', names=['Id1', 'PredictScore', 'GoldStandard', 'Threshold'],
                               dtype={'Id1': str, 'PredictScore': np.int32, 'GoldStandard': str, 'Threshold': str})
        combined_df = pd.concat([text_df, score_df], axis=1, sort=False)
        selected_df = combined_df.drop(['essay_score2', 'Id1', 'GoldStandard', 'Threshold'], axis=1)
        selected_df.to_excel(writer, sheet_name='p'+str(i)+'_c'+str(n), encoding='utf-8',index=False)
writer.save()

writer = pd.ExcelWriter('outputs/token_1-5gram_experiment_result.xlsx', engine='xlsxwriter')
for i in range(1, 11):
    for n in range(1, 6):
        print('prompt: '+str(i)+'\ntoken: '+str(n))
        text_df = pd.read_csv('outputs/token_' + str(n) + '_gram_prompt_' + str(i) + '_1000.txt',
                              encoding='utf-8', sep='\t', header=0, quoting=csv.QUOTE_NONE,
                              names=['Id', 'EssaySet', 'essay_score1', 'essay_score2', 'EssayText'],
                              dtype={'Id': str, 'EssaySet': str, 'essay_score1': np.int32, 'essay_score2': str, 'EssayText': str})
        score_df = pd.read_csv('id2outcome/token_1-5gram/p'+str(i)+'_t'+str(n)+'.txt',
                               encoding='utf-8', sep='\t', names=['Id1', 'PredictScore', 'GoldStandard', 'Threshold'],
                               dtype={'Id1': str, 'PredictScore': np.int32, 'GoldStandard': str, 'Threshold': str})
        combined_df = pd.concat([text_df, score_df], axis=1, sort=False)
        selected_df = combined_df.drop(['essay_score2', 'Id1', 'GoldStandard', 'Threshold'], axis=1)
        selected_df.to_excel(writer, sheet_name='p'+str(i)+'_t'+str(n), encoding='utf-8',index=False)
writer.save()
