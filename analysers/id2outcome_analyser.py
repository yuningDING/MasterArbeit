import pandas as pd

for i in range(1, 11):
    df = pd.read_csv('id2outcome/token_simi_id2outcome/p' + str(i) + '.txt', skiprows=3,
                           encoding='utf-8', sep=';', names=['PredictScore', 'GoldStandard', 'Threshold'],
                           dtype={'PredictScore': str, 'GoldStandard': str, 'Threshold': str})
    df['ID'] = df['PredictScore'].str.extract('(\d-\d*)', expand=True)
    df['score'] = df['PredictScore'].str.extract('(=\d)', expand=True)
    df['score'] = df['score'].str.extract('(\d)', expand=True)
    for n in range(1,6):
        prompt_df = df[df['ID'].str.contains(str(n)+'-')]
        print('Prompt '+str(i)+' - Char '+ str(n) + ': '+str(len(prompt_df)) )
        print('Score 0: '+str(len(prompt_df[(df['score'].str.contains('0'))])))
        accuarcy = len(prompt_df[(df['score'].str.contains('0'))])/len(prompt_df)
        print('Accuracy: '+ str(accuarcy))