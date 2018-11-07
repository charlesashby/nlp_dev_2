import random
import numpy as np


def tokenize_sentence(sentence):
    tokens = sentence.split(' ')
    tokens_clean = []
    for i, _ in enumerate(tokens):
        if tokens[i] == '<unk':
            tokens_clean.append('{} {}'.format(tokens[i], tokens[i + 1]))
        elif 'w="' == tokens[i][:3]:
            pass
        else:
            tokens_clean.append(tokens[i])
    return tokens_clean


def get_vector(model, token):
    try:
        vec = model.get_vector(token)
    except KeyError:
        vec = np.zeros(shape=(300,))
    return vec


def shuffle_training_set(train_en, train_fr, train_y, file_out):
    lines_en = open(train_en).readlines()
    lines_fr = open(train_fr).readlines()
    lines_y = open(train_y).readlines()
    randomize = np.arange(len(lines_y))
    np.random.shuffle(randomize)
    l_en = [lines_en[i] for i in randomize]
    l_fr = [lines_fr[i] for i in randomize]
    l_y = [lines_y[i] for i in randomize]
    with open(file_out, 'a') as f:
        for l in lines:
            f.write('{}\n'.format(l))
