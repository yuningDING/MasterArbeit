import pandas
import csv
import numpy as np


# Baseline
df = pandas.read_csv('resources/train_plus_test_repaired.txt', encoding='utf-8', sep='\t', header=0,
                     quoting=csv.QUOTE_NONE,
                     names=['Id', 'EssaySet', 'essay_score1', 'essay_score2', 'EssayText'],
                     dtype={'Id': str, 'EssaySet': str, 'essay_score1': str, 'essay_score2': str,
                            'EssayText': str})


for i in range(1,11):

    # Char 1-5 gram
    df = pandas.read_csv('outputs/char_bunch/char_prompt_'+str(i)+'.txt', encoding='utf-8', sep='\t', header=0,
                         quoting=csv.QUOTE_NONE,
                         names=['Id', 'EssaySet', 'essay_score1', 'essay_score2', 'EssayText'],
                         dtype={'Id': str, 'EssaySet': str, 'essay_score1': str, 'essay_score2': str,
                                'EssayText': str})
    sample_df = df.sample(200)

    print('prompt '+str(i)+':')
    prompt_df = sample_df.loc[df['EssaySet'] == str(i)]
    print('number of answers: '+str(len(prompt_df)))
    ta_df = pandas.read_csv('outputs/TA_CS229/TA_CS229_'+str(i)+'.txt', encoding='utf-8', sep='\t', header=0,
                     quoting=csv.QUOTE_NONE,
                     names=['Id', 'Text'],
                     dtype={'Id': str, 'Text': str})
    ta_list = []

    for index, row in ta_df.iterrows():
        ta_list.append(row['Text'])
    print(ta_list)
    with open('outputs/Input_CS229_multilable_char' + str(i) + '.csv', 'w', encoding='utf-8') as file:
        file.write('studentAnswer\treferenceAnswer\tref_1\tref_2\taccuracy\n')
        for index, row in prompt_df.iterrows():
            label = 'correct'
            if row['essay_score1'] == 0:
                label = 'incorrect'
            file.write(row['EssayText']+'\t'+ta_list[0]+'\t'+ta_list[1]+'\t'+ta_list[2]+'\t'+row['essay_score1']+'\n')
        file.close()