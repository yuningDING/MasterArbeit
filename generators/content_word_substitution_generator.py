import pandas as pd
import numpy as np
import random
import nltk
import csv
from scipy import spatial
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from shallow_similarity.target_answer_cluster import load_glove


def get_word_list(path):
    df = pd.read_csv(path, encoding='utf-8', sep='\t',header=0,
                     names=['Rank', 'Word', 'Part of speech', 'Frequency', 'Dispersion'],
                     dtype={'Rank': str, 'Word': str, 'Part of speech': str, 'Frequency': np.int32,
                            'Dispersion': np.float64})
    noun_list = list(df.loc[df['Part of speech'] == 'n']['Word'])
    verb_list = list(df.loc[df['Part of speech'] == 'v']['Word'])
    adj_list = list(df.loc[df['Part of speech'] == 'j']['Word'])
    return noun_list, verb_list, adj_list


if __name__ == "__main__":
    # load model
    model = load_glove('resources/glove.6B.50d.txt')
    # load word lists
    noun_list, verb_list, adj_list = get_word_list('resources/top5000.txt')
    # load original answers
    for i in range(5, 11):
        file_path = 'resources/asap_withoutBG_prompt_' + str(i) + '.txt'
        text_df = pd.read_csv(file_path, encoding='utf-8', sep='\t', header=0,
                              quoting=csv.QUOTE_NONE,
                              names=['Id', 'EssayText'],
                              dtype={'Id': str, 'EssayText': str})
        input_df = text_df.sample(n=1000)
        # substitute
        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        pos_list = ['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS']
        with open('outputs/content_substitution_prompt_' + str(i) + '.txt', 'w', encoding='utf-8') as file:
            file.write('Id\tEssaySet\tessay_score\tessay_score\tEssayText\n')
            for index, row in input_df.iterrows():
                doc = row['EssayText']
                substitute = ''
                tokens_in_doc = tokenizer.tokenize(doc.lower())
                tags_in_doc = nltk.pos_tag(tokens_in_doc)
                for token in tokens_in_doc:
                    if token not in stop_words:
                        token_tag = tags_in_doc[tokens_in_doc.index(token)][1]
                        print('TAG:'+token_tag)
                        token_sub = token
                        distance = 0
                        # if token is a noun
                        if token_tag in pos_list[:2]:
                            token_lemma = lemmatizer.lemmatize(token, pos=wordnet.NOUN)
                            if token_lemma not in model:
                                continue
                            token_vector = model[token_lemma]
                            while distance < 0.2:
                                token_sub = random.choice(noun_list).lower()
                                token_sub_vector = model[token_sub]
                                distance = spatial.distance.cosine(token_vector, token_sub_vector)
                        # if token is a verb
                        elif token_tag in pos_list[2:8]:
                            token_lemma = lemmatizer.lemmatize(token, pos=wordnet.VERB)
                            if token_lemma not in model:
                                continue
                            token_vector = model[token_lemma]
                            while distance < 0.2:
                                token_sub = random.choice(verb_list).lower()
                                token_sub_vector = model[token_sub]
                                distance = spatial.distance.cosine(token_vector, token_sub_vector)
                        # if token is a adj
                        elif token_tag in pos_list[-3:]:
                            token_lemma = lemmatizer.lemmatize(token, pos=wordnet.ADJ)
                            if token_lemma not in model:
                                continue
                            token_vector = model[token_lemma]
                            while distance < 0.2:
                                token_sub = random.choice(adj_list).lower()
                                token_sub_vector = model[token_sub]
                                distance = spatial.distance.cosine(token_vector, token_sub_vector)
                        substitute = substitute + ' ' + token_sub
                    else:
                        substitute = substitute + ' ' + token
                if len(substitute)==0:
                    substitute = random.choice(noun_list).lower()
                file.write(row['Id'] + '\t' + str(i) + '\t0\t0\t' + substitute + '\n')
            file.close()
