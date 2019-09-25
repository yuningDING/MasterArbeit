import pandas as pd
import csv
import numpy as np

for i in range(1, 11):
    # file_path = 'deep_instance_outputs/neural_sas_char/p' + str(i) + '.txt'
    file_path = 'deep_instance_outputs/neural_sas_token/p' + str(i) + '.txt'
    df = pd.read_csv(file_path, encoding='utf-8', sep='\n', header=0,
                          quoting=csv.QUOTE_NONE,
                          names=['prediction'],
                          dtype={'prediction': np.int32})
    print('[Prompt ', i, ']' )
    char1 = df.iloc[0:1000, :]
    char1_pre = char1['prediction'].value_counts().to_dict()
    print('Accuracy of 1-gram: ', char1_pre.get(0)/1000)
    char2 = df.iloc[1001:2000, :]
    char2_pre = char2['prediction'].value_counts().to_dict()
    print('Accuracy of 2-gram ', char2_pre.get(0) / 1000)
    char3 = df.iloc[2001:3000, :]
    char3_pre = char3['prediction'].value_counts().to_dict()
    print('Accuracy of 3-gram: ', char3_pre.get(0) / 1000)
    char4 = df.iloc[3001:4000, :]
    char4_pre = char4['prediction'].value_counts().to_dict()
    print('Accuracy of 4-gram: ', char4_pre.get(0) / 1000)
    char5 = df.iloc[4001:5000, :]
    char5_pre = char5['prediction'].value_counts().to_dict()
    print('Accuracy of 5-gram: ', char5_pre.get(0) / 1000)

