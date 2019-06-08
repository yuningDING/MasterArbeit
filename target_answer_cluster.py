import pandas as pd
import csv
import numpy as np
import nltk
import collections
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


# read docs from file
def get_target_answer(file_path):
    text_df = pd.read_csv(file_path, encoding='utf-8', sep='\t', header=0,
                          quoting=csv.QUOTE_NONE,
                          names=['Id', 'EssaySet', 'essay_score1', 'essay_score2', 'EssayText'],
                          dtype={'Id': str, 'EssaySet': str, 'essay_score1': np.int32, 'essay_score2': np.int32,
                                 'EssayText': str})
    text_df = text_df['EssayText']
    docs = text_df.values.squeeze()
    return docs


# pre-process docs
def pre_process(docs, pos_list):
    token_list = []
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    for doc in docs:
        tokens = []
        tokens_in_doc = tokenizer.tokenize(doc.lower())
        tags_in_doc = nltk.pos_tag(tokens_in_doc)
        for token in tokens_in_doc:
            if token not in stop_words:
                for pos in pos_list:
                    if tags_in_doc[tokens_in_doc.index(token)][1] == pos:
                        tokens.append(token)
        token_list.append(tokens)
        print(token_list)
    return token_list


# load GloVe
def load_glove(glove_file):
    print('Loading GloVe Model...')
    f = open(glove_file, 'r', encoding='utf8')
    model = {}
    for line in f:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]])
        model[word] = embedding
    print('Done. ', len(model), ' words loaded!')
    return model


# turn token list into vector
def get_doc_vector(model, dimension, token_list):
    doc_vectors = []
    for list in token_list:
        sum_vector = np.zeros(dimension)
        for t in list:
            token_vector = np.zeros(dimension)
            if t in model:
                token_vector = model[t]
            sum_vector = np.sum([sum_vector, token_vector], axis=0).tolist()
        doc_vectors.append([x / len(list) for x in sum_vector])
    return doc_vectors


# k-mean clustering
def get_clusters(doc_vectors, cluster_number):
    X = doc_vectors
    kmeans = KMeans(n_clusters=cluster_number)
    kmeans.fit(X)
    clustering = collections.defaultdict(list)
    for idx, label in enumerate(kmeans.labels_):
        clustering[label].append(idx)
    return clustering


# x-mean clustering
def get_x_clusters(doc_vectors):
    # Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
    # start analysis.
    amount_initial_centers = 2
    initial_centers = kmeans_plusplus_initializer(doc_vectors, amount_initial_centers).initialize()
    # Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
    # number of clusters that can be allocated is 20.
    xmeans_instance = xmeans(doc_vectors, initial_centers, 20)
    xmeans_instance.process()
    # Extract clustering results: clusters and their centers
    clusters = xmeans_instance.get_clusters()
    return clusters


if __name__ == "__main__":
    model = load_glove('resources/glove.6B.50d.txt')

    # k-means

    # a small test
    # docs = ["How much vinegar you pour into the cups"]
    # tokens = pre_process(docs, ['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'JJ', 'JJR', 'JJS'])
    # clusters = get_x_clusters(docs)
    # for cluster in dict(clusters).keys():
    #     for index in dict(clusters).get(cluster):
    #         print(str(cluster)+'\t'+docs[index])

    for i in range(1, 11):
        file_path = 'outputs/TA_prompt_' + str(i) + '.txt'
        articles = get_target_answer(file_path)
        tokens = pre_process(articles,
                             ['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'JJ', 'JJR', 'JJS'])
        vectors = get_doc_vector(model, 50, tokens)
        # k =  total number of target answers / 20
        k = int(len(articles) / 20)
        print('Target answers in prompt ' + str(i) + ' has ' + str(k) + ' clusters.')
        clusters = get_clusters(vectors, k)
        with open('outputs/TA_clusters_' + str(i) + '.txt', 'w', encoding='utf-8') as file:
            file.write('Id\tEssaySet\tessay_score\tessay_score\tEssayText\n')
            for cluster in dict(clusters).keys():
                index = dict(clusters).get(cluster)[0]
                file.write(str(index) + '\t' + str(i) + '\t' + '3\t3\t' + articles[index] + '\n')
            file.close()

    # x-means
    #     with open('outputs/TA_X_clusters_' + str(i) + '.txt', 'w', encoding='utf-8') as file:
    #         clusters = get_x_clusters(vectors)
    #         file.write('Cluster\tIndex\tEssayText\n')
    #         for cluster in range(0, len(clusters)):
    #             for index in clusters[cluster]:
    #                 file.write(str(cluster)+'\t'+ str(index)+'\t'+articles[index]+'\n')
    #         file.close()
