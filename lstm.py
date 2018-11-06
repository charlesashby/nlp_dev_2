import os, csv
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from gensim.models import KeyedVectors

data_path = '/run/media/ashbylepoc/b79b0a3e-a5b9-41ed-987f-8fa4bdb6b2e6/tmp/data/nlp_dev_2/'
word2vec_dir = '/run/media/ashbylepoc/b79b0a3e-a5b9-41ed-987f-8fa4bdb6b2e6/tmp/data/word2vec'
train_y = os.path.join(data_path, 'train.y')
train_en = os.path.join(data_path, 'train.en')
train_fr = os.path.join(data_path, 'train.fr')
lexicon = os.path.join(data_path, 'lexique.en-fr')
model = KeyedVectors.load_word2vec_format(
            '{}/GoogleNews-vectors-negative300.bin'.format(word2vec_dir),
            binary=True)


class DataReader(object):

    def __init__(self, train_y, train_en, train_fr, max_n_token_sentence_fr, max_n_token_sentence_en,
                 token_emb_size):
        self.reader_train_y = open(train_y, 'r').readlines()
        self.reader_train_en = open(train_en, 'r').readlines()
        self.reader_train_fr = open(train_fr, 'r').readlines()
        self.max_n_token_sentence_fr = max_n_token_sentence_fr
        self.max_n_token_sentence_en = max_n_token_sentence_en
        self.token_emb_size = token_emb_size

    def make_mini_batch_train(self, index, batch_size):

        mini_batch_en = np.zeros(shape=[batch_size,
                                       self.max_n_token_sentence_en,
                                       self.token_emb_size])
        mini_batch_fr = np.zeros(shape=[batch_size,
                                       self.max_n_token_sentence_fr,
                                       self.token_emb_size])
        mini_batch_y = np.zeros(shape=[batch_size, 2])

        lines_en = self.reader_train_en[index * batch_size: (index + 1) * batch_size]
        lines_fr = self.reader_train_fr[index * batch_size: (index + 1) * batch_size]
        lines_y = self.reader_train_y[index * batch_size: (index + 1) * batch_size]

        for i, y in enumerate(lines_y)

        lines = enumerate(lines)
        while act_batch_size < batch_size:
            j, line = next(lines)
            y = line[-1]
            y_pos = self.get_token_dict_pos(y)

            mini_batch_y[act_batch_size][y_pos] = 1.

            # print('actual batch size: {} {}'.format(act_batch_size < batch_size, act_batch_size))
            if y_pos != -2 and y_pos != -3:
                x = line[- (self.max_n_tokens_sentence + 1):-1]
                # print('x tokens used for inference: {}'.format(x))
                # print('y token used for target: {}'.format(y))
                for i, token in enumerate(x):
                    token_pos = self.get_token_dict_pos(token)
                    mini_batch_x[act_batch_size][i][token_pos] = 1.
                act_batch_size += 1
            else:
                pass
        return mini_batch_x, mini_batch_y

    def mb_predict_middle_sentence(self, lines, batch_size, pre=5, suf=3):
        n_lines = len(lines)
        act_batch_size = 0
        mini_batch_x = np.zeros(shape=[batch_size,
                                       pre + suf + 1,
                                       self.n_tokens_dict])
        mini_batch_y = np.zeros(shape=[batch_size, self.n_tokens_dict])
        # We load twice the number of lines we need to make sure there are no
        # "unknown" y's (since we reduced the size of the word dict
        lines = enumerate(lines)
        while act_batch_size < batch_size:
            # import pdb; pdb.set_trace()
            j, line = next(lines)

            x = line
            y = x[pre]
            y_pos = self.get_token_dict_pos(y)

            mini_batch_y[act_batch_size][y_pos] = 1.

            # print('actual batch size: {} {}'.format(act_batch_size < batch_size, act_batch_size))
            if y_pos != -2 and y_pos != -3:
                # x = line[-9:]

                # print('x tokens used for inference: {}'.format(x))
                # print('y token used for target: {}'.format(y))
                for i, token in enumerate(x):
                    if i == pre:
                        token_pos = -1
                        mini_batch_x[act_batch_size][i][token_pos] = 1.

                    else:
                        token_pos = self.get_token_dict_pos(token)
                        mini_batch_x[act_batch_size][i][token_pos] = 1.
                act_batch_size += 1
            else:
                pass


        return mini_batch_x, mini_batch_y

    def load_to_ram(self, batch_size, file):
        n_rows = batch_size
        self.data = []
        while n_rows > 0:
            self.data.append(next(file))
            n_rows -= 1
        if n_rows == 0:
            return True
        else:
            return False

    def iterate_mini_batch(self, batch_size, pre=5, suf=3, dataset='train'):
        if dataset == 'train':
            n_batch = int(2800000 / (batch_size * 1.5))
            for i in range(n_batch):
                # We load twice the number of lines we need to make sure there are no
                # "unknown" y's (since we reduced the size of the word dict
                if self.load_to_ram(int(batch_size * 1.5), self.reader_train):
                    # inputs, targets = self.make_mini_batch_train(self.data, batch_size)
                    inputs, targets = self.mb_predict_middle_sentence(self.data, batch_size, pre, suf)

                    yield inputs, targets
        else:
            # n_batch = int(1000. / batch_size)
            n_batch = 100
            for i in range(n_batch):
                # print('valid: {}'.format(i))
                # We load twice the number of lines we need to make sure there are no
                # "unknown" y's (since we reduced the size of the word dict
                if self.load_to_ram(int(batch_size * 1.5), self.reader_valid):
                    inputs, targets = self.mb_predict_middle_sentence(self.data, batch_size, pre, suf)
                    # inputs, targets = self.make_mini_batch_train(self.data, batch_size)
                    yield inputs, targets
