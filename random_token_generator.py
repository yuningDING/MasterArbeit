import random, re
from itertools import groupby
from numpy.random import choice

SEP = " "  # token separator symbol


def _preprocess_corpus(filename):
    """ Returns preprocessed text in corpus file"""
    s = open(filename, 'r', encoding='utf-8').read()
    s = re.sub('([^0-9])([.,!?])([^0-9])', r'\1 \2 \3', s)  # pad sentence punctuation chars with whitespace
    s = s.lower()
    return s.split(SEP)


def _get_token_ngrams(token_list, n):
    """ Returns a dict of N-long token n-grams from token list in corpus"""
    # key: n-gram in corpus
    # value: list of possible n-grams following key
    token_ngrams = {}
    for i in range(len(token_list) - n + 1):
        gram = ' '.join(token_list[i:i + n])
        if gram not in token_ngrams.keys():
            token_ngrams[gram] = []
        if i + n < len(token_list):
            token_ngrams[gram].append(token_list[i + n])
    for g in token_ngrams.keys():
        token_ngrams.get(g).sort()
    return token_ngrams


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
            frequency.append(n / total)
        ngram_frequency[ngram] = frequency
        ngram_char_set[ngram] = char_set
    return ngram_char_set, ngram_frequency


def _get_next(gram, ngram_dict, ngram_frequency_dict):
    """ Outputs the next word to add by using frequency weighted random method """
    if len(ngram_dict[gram]) is 0 or gram not in ngram_dict.keys():
        return '\n'
    choices = ngram_dict[gram]
    frequency = ngram_frequency_dict[gram]
    draw = choice(a=choices, size=1, p=frequency)
    return draw[0]


def _postprocess_output(s):
    s = re.sub('\\s+([.,!?])\\s*', r'\1 ', s)  # correct whitespace padding around punctuation
    s = s.capitalize()  # capitalize first letter
    s = re.sub('([.!?]\\s+[a-z])', lambda c: c.group(1).upper(), s)  # capitalize letters following terminated sentences
    return s


def get_text(ngram_dict, ngram_token_set, ngram_frequency_dict, n=3, sentence_length=140, start=None):
    """ Generate a random sentence based on input text corpus """
    if start is None:
        start = random.choice(list(ngram_dict.keys()))

    while re.match(r'^[a-zA-Z0-9 ]*$', start) is None:
        start = random.choice(list(ngram_dict.keys()))

    current_sentence = start
    current_length = 0
    while current_length < sentence_length:
        next_gram = ''
        if n > 1:
            next_gram = _get_next(' '.join(current_sentence.split(SEP)[-n:]), ngram_token_set, ngram_frequency_dict)
        else:
            next_gram = random.choice(list(ngram_dict.keys()))
        current_sentence = current_sentence + ' ' + next_gram
        if '\n' in next_gram:
            end = current_sentence.find('\n')
            current_sentence = current_sentence[0:end]
            return _postprocess_output(current_sentence)

        current_length = len(current_sentence)
    return _postprocess_output(current_sentence)


max_length = {'1': 776, '2': 918, '3': 742, '4': 631, '5': 1478, '6': 1032, '7': 1103, '8': 1651, '9': 1820, '10': 1188}

if __name__ == "__main__":
    index = 0
    for i in range(1, 11):
        for n in range(1, 6):
            with open('outputs/new_token_' + str(n) + '_gram_prompt_' + str(i) + '_1000.txt', 'w',
                      encoding='utf-8') as file:
                print(
                    'Generating ' + str(n) + '-gram based answer for' + ' prompt ' + str(i) + ' with max length ' + str(
                        max_length[str(i)]))
                # get n-grams and frequency distribution
                token_list = _preprocess_corpus('resources/asap_prompt_' + str(i) + '.txt')
                ngram_dict = _get_token_ngrams(token_list, n)
                ngram_token_set, ngram_frequency_dict = _get_ngram_frequency(ngram_dict)
                file.write('Id\tEssaySet\tessay_score\tessay_score\tEssayText\n')
                for j in range(1000):
                    sentence = get_text(ngram_dict, ngram_token_set, ngram_frequency_dict, n, max_length[str(i)])
                    file.write(str(index) + '\t' + str(i) + '\t' + '0\t0\t' + sentence + '\n')
                    index += 1
            file.close()
   