import re
import random
from itertools import groupby
from numpy.random import choice


def _get_char_ngrams(file_path, n):
    """ Returns a dict of N-long character n-grams from text in corpus file """
    # key: n-gram in corpus
    # value: list of possible n-grams following key
    ngram_dict = {}
    ngram_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    for i in range(len(text) - n + 1):
        gram = text[i:i+n]
        if gram not in ngram_dict.keys():
            ngram_dict[gram] = []
        if i+n < len(text):
            ngram_dict[gram].append(text[i+n])
            ngram_list.append(gram)
    ngram_set = list(sorted(set(ngram_list)))
    ngram_frequency = []
    for g in ngram_set:
        ngram_dict.get(g).sort()
        ngram_frequency.append(ngram_list.count(g) / len(ngram_list))
    return ngram_dict, ngram_set, ngram_frequency


def _get_ngram_frequency(char_ngrams):
    ngram_frequency = {}
    ngram_char_set = {}
    for ngram in char_ngrams.keys():
        char_list = char_ngrams.get(ngram)
        char_list.sort()
        char_set = list(sorted(set(char_list)))
        ngram_frequency[ngram] = [len(list(group)) for key, group in groupby(char_list)]
        frequency = []
        for n in ngram_frequency[ngram]:
            total = sum(ngram_frequency[ngram])
            frequency.append(n/total)
        ngram_frequency[ngram] = frequency
        ngram_char_set[ngram] = char_set
    return ngram_char_set, ngram_frequency


# # An idea to generate start n-grams using original frequency distribution of start n-grams in corpus.
# # Not used in thesis
# def _get_start_gram(file_path, n):
#     start_ngrams = []
#     with open(file_path, 'r', encoding='utf-8') as file:
#         line = file.readline()
#         while line:
#             start_ngrams.append(line[0: n])
#             line = file.readline()
#     return start_ngrams
#
#
# def _get_start_gram_frequency(start_ngrams):
#     start_ngrams.sort()
#     start_ngrams_set = list(sorted(set(start_ngrams)))
#     start_ngrams_frequency = [len(list(group)) for key, group in groupby(start_ngrams)]
#     frequency = []
#     for n in start_ngrams_frequency:
#         total = sum(start_ngrams_frequency)
#         frequency.append(n/total)
#     return start_ngrams_set, frequency


def _get_next(gram, ngram_dict, ngram_frequency_dict):
    if len(ngram_dict[gram]) is 0 or gram not in ngram_dict.keys():
        return '\n'
    choices = ngram_dict[gram]
    frequency = ngram_frequency_dict[gram]
    draw = choice(a=choices, size=1, p=frequency)
    return draw[0]


def get_text(ngram_set, ngram_frequency, ngram_char_set, ngram_frequency_dict, n, sentence_length, start=None):
    # get start-grams and frequency distribution
    # start_grams = _get_start_gram(file_path, n)
    # start_ngram_set, start_gram_frequency = _get_start_gram_frequency(start_grams)

    # choose a start gram using weighted random
    # numpy.random.RandomState.choice(a, size=None, p=None)
    if start is None:
        start = random.choice(ngram_set)
    while re.match(r'^[a-zA-Z0-9 ]*$', start) is None:
        start = random.choice(ngram_set)
    while start is '\n':
        start = random.choice(ngram_set)
        # choices = choice(a=start_ngram_set, size=1, p=start_gram_frequency)
        # start = choices[0]

    current_sentence = start
    current_length = 0
    while current_length < sentence_length:
        next_gram = ''
        if n > 1:
            next_gram = _get_next(current_sentence[-n:], ngram_char_set, ngram_frequency_dict)
        else:
            next_gram = choice(a=ngram_set, size=1, p=ngram_frequency)[0]
        current_sentence += next_gram
        if '\n' in next_gram:
            end = current_sentence.find('\n')
            current_sentence = current_sentence[0:end]
            return current_sentence

        current_length = len(current_sentence)
    return current_sentence


max_length = {'1': 776, '2': 918, '3': 742, '4': 631, '5': 1478, '6': 1032, '7': 1103, '8': 1651, '9': 1820, '10': 1188}


if __name__ == "__main__":
    index = 0
    for i in range(1, 11):
        n = 1
        with open('outputs/char_1-5gram/char_'+str(n)+'_gram_prompt_'+str(i)+'_1000.txt', 'w', encoding='utf-8') as file:
            print('Generating ' + str(n) + '-gram based answer for' + ' prompt ' + str(i) + ' with max length ' + str(max_length[str(i)]))
            ngram_dict, ngram_set, ngram_frequency = _get_char_ngrams('resources/asap_prompt_'+str(i)+'.txt', n)
            ngram_char_set, ngram_frequency_dict = _get_ngram_frequency(ngram_dict)
            file.write('Id\tEssaySet\tessay_score\tessay_score\tEssayText\n')
            for j in range(1000):
                sentence = get_text(ngram_set, ngram_frequency, ngram_char_set, ngram_frequency_dict, n, max_length[str(i)])
                while len(sentence)<3:
                    sentence = get_text(ngram_set, ngram_frequency, ngram_char_set, ngram_frequency_dict, n,
                                        max_length[str(i)])
                file.write(str(index)+'\t'+str(i)+'\t'+'0\t0\t'+sentence+'\n')
                index += 1
        file.close()

