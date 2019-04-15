import random
from itertools import groupby
from numpy.random import choice


def _get_char_ngrams(file_path, n):
    """ Returns a dict of N-long character n-grams from text in corpus file """
    # key: n-gram in corpus
    # value: list of possible n-grams following key
    char_ngrams = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    for i in range(len(text) - n + 1):
        gram = text[i:i+n]
        if gram not in char_ngrams.keys():
            char_ngrams[gram] = []
        if i+n < len(text):
            char_ngrams[gram].append(text[i+n])
    for g in char_ngrams.keys():
        char_ngrams.get(g).sort()
    return char_ngrams


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
        return '/n'
    choices = ngram_dict[gram]
    frequency = ngram_frequency_dict[gram]
    draw = choice(a=choices, size=1, p=frequency)
    return draw[0]


def get_text(file_path, n=3, sentence_length=140, start=None):
    # get n-grams and frequency distribution
    ngram_dict = _get_char_ngrams(file_path, n)
    ngram_char_set, ngram_frequency_dict = _get_ngram_frequency(ngram_dict)

    # get start-grams and frequency distribution
    # start_grams = _get_start_gram(file_path, n)
    # start_ngram_set, start_gram_frequency = _get_start_gram_frequency(start_grams)

    # choose a start gram using weighted random
    # numpy.random.RandomState.choice(a, size=None, p=None)
    if start is None:
        start = random.choice(list(ngram_dict.keys()))
        # choices = choice(a=start_ngram_set, size=1, p=start_gram_frequency)
        # start = choices[0]

    current_sentence = start
    current_length = 0
    while current_length < sentence_length:
        next_gram = _get_next(current_sentence[-n:], ngram_char_set, ngram_frequency_dict)
        current_sentence += next_gram
        if '\n' in next_gram:
            end = current_sentence.find('\n')
            current_sentence = current_sentence[0:end]
            return current_sentence

        current_length = len(current_sentence)
    return current_sentence


if __name__ == "__main__":
    with open('outputs/char_'+str(5)+'_gram_prompt_1_1000.txt', 'w', encoding='utf-8') as file:
        for i in range(1000):
            sentence = get_text('resources/asap_prompt_1.txt', 5, 776)
            file.write(sentence+'\n')
    file.close()

