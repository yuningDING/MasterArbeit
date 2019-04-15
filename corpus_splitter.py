import pandas


df = pandas.read_csv('resources/train_plus_test_repaired.txt', names=['Id','EssaySet','Score1','Score2','EssayText'], sep='\t', lineterminator='\n')
prompts = []
for i in range(1, 11):
    prompts.append(df.loc[df['EssaySet'] == str(i)][['Id', 'EssayText']])

for i in range(1, 11):
    prompts[i-1].to_csv('resources/asap_prompt_'+str(i)+'.txt', sep='\t', index=False, encoding='utf-8', line_terminator='\n' )
