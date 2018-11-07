import os, csv
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from utils.utils import get_vector
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

    def __init__(self, train_y, train_en, train_fr, max_n_token_sentence_fr,
                 max_n_token_sentence_en, token_emb_size):
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

        for i, y in enumerate(lines_y):
            # tokenize english sentence
            y = int(y[:-1])
            for j, token in enumerate(lines_en[i].split(' ')):
                token_vec = get_vector(model, token)
                mini_batch_en[i, j, :] = token_vec

            # tokenize french sentence
            for j, token in enumerate(lines_fr[i].split(' ')):
                token_vec = get_vector(model, token)
                mini_batch_fr[i, j, :] = token_vec

            mini_batch_y[i][y] = 1.

        return mini_batch_en, mini_batch_fr, mini_batch_y

    def iterate_mini_batch(self, batch_size, dataset='train'):
        if dataset == 'train':
            n_batch = int(995000 / batch_size)
            last_batch_size = batch_size * (n_batch - (995000. / batch_size))
            for i in range(n_batch):
                mb_en, mb_fr, mb_y = self.make_mini_batch_train(i, batch_size)
                yield mb_en, mb_fr, mb_y, batch_size
            if last_batch_size > 0:
                mb_en, mb_fr, mb_y = self.make_mini_batch_train(i, last_batch_size)
                yield mb_en, mb_fr, mb_y, last_batch_size
        else:
            n_batch = int(5000. / batch_size)
            last_batch_size = batch_size * (n_batch - (5000. / batch_size))
            for i in range(n_batch):
                mb_en, mb_fr, mb_y = self.make_mini_batch_train(i + 995000, batch_size)
                yield mb_en, mb_fr, mb_y, batch_size
            if last_batch_size > 0:
                mb_en, mb_fr, mb_y = self.make_mini_batch_train(i + 995000, last_batch_size)
                yield mb_en, mb_fr, mb_y, last_batch_size


if __name__ == '__main__':
    save_path = os.path.join(data_path, 'checkpoints')
    file_out = 'duo_lstm'

    starting_batch_size = 32
    rnn_size = 1024
    learning_rate = 0.0001

    max_n_token_sentence_en = 304
    max_n_token_sentence_fr = 402
    token_emb_size = 300

    batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    en = tf.placeholder('float32', shape=[None, max_n_token_sentence_en, token_emb_size])
    fr = tf.placeholder('float32', shape=[None, max_n_token_sentence_fr, token_emb_size])
    y = tf.placeholder('float32', shape=[None, 2])

    # LSTM for english sentence
    with tf.variable_scope('scope_en'):
        cell_en = rnn.LSTMCell(rnn_size, state_is_tuple=True, forget_bias=0., reuse=False)
        initial_rnn_state_en = cell_en.zero_state(batch_size, dtype='float32')
        outputs_en, final_rnn_state_en = tf.nn.dynamic_rnn(cell_en, en, initial_state=initial_rnn_state_en,
                                                           dtype='float32')
        outputs_en = tf.transpose(outputs_en, [1, 0, 2])
        last_en = outputs_en[-1]

    # LSTM for french sentence
    with tf.variable_scope('scope_fr'):
        cell_fr = rnn.LSTMCell(rnn_size, state_is_tuple=True, forget_bias=0., reuse=False)
        initial_rnn_state_fr = cell_fr.zero_state(batch_size, dtype='float32')
        outputs_fr, final_rnn_state_fr = tf.nn.dynamic_rnn(cell_fr, fr, initial_state=initial_rnn_state_fr,
                                                           dtype='float32')
        outputs_fr = tf.transpose(outputs_fr, [1, 0, 2])
        last_fr = outputs_fr[-1]

    # Classification: concatenate the last outputs and
    # pass the vector through an MLP with softmax activation fn
    concatenated_outputs = tf.concat([last_en, last_fr], axis=1)
    w = tf.get_variable('w', [rnn_size * 2, 2], dtype='float32')
    b = tf.get_variable('b', [2], dtype='float32')
    preds = tf.nn.softmax(tf.matmul(concatenated_outputs, w) + b)

    cost = - tf.reduce_sum(y * tf.log(tf.clip_by_value(preds, 1e-10, 1.0)), axis=1)
    cost = tf.reduce_mean(cost, axis=0)
    predictions = tf.cast(tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1)), dtype='float32')
    acc = tf.reduce_mean(predictions)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    saver = tf.train.Saver()
    best_acc = 0.

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, '{}/lstmp10k5-3v1__{}_{}/lstm'.format(SAVE_PATH, 40000, 0))
    for epoch in range(10):
        data_reader = DataReader(train_y, train_en, train_fr,
                                 max_n_token_sentence_fr=402,
                                 max_n_token_sentence_en=304,
                                 token_emb_size=300)
        train_acc = []
        train_cost = []
        for i, batch in enumerate(data_reader.iterate_mini_batch(starting_batch_size, dataset='train')):
            b_en, b_fr, b_y, b_size = batch
            _, c, a, preds_ = sess.run([optimizer, cost, acc, preds],
                                       feed_dict={en: b_en, fr: b_fr, y: b_y,
                                                  batch_size: b_size})
            train_acc.append(a)
            train_cost.append(c)

            if i % 500 == 0:
                print(np.argmax(preds_, axis=1))
                print(np.argmax(b_y, axis=1))
                print('TRAIN: epoch: {} - iteration: {} - acc: {} - loss: {}'.format(epoch, i, np.mean(train_acc),
                                                                                     np.mean(train_cost)))
                with open('log_{}.txt'.format(file_out), 'a') as f:
                    f.write(
                        'TRAIN: epoch: {} - iteration: {} - acc: {} - loss: {} \n'.format(epoch, i, np.mean(train_acc),
                                                                                          np.mean(train_cost)))
                train_acc = []
                train_cost = []
            if i % 5000 == 0 and i != 0:
                valid_acc = []

                for k, batch_valid in enumerate(
                        data_reader.iterate_mini_batch(starting_batch_size, dataset='valid')):
                    bb_en, bb_fr, bb_y, bb_size = batch_valid
                    # compute accuracy on validation set
                    cc, aa, preds__ = sess.run([cost, acc, preds],
                                               feed_dict={en: bb_en, fr: bb_fr, y: bb_y,
                                                            batch_size: bb_size})
                    valid_acc.append(aa)
                mean_acc = np.mean(valid_acc)
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    os.mkdir('{}/{}_{}_{}'.format(file_out, save_path, i, epoch))
                    save_path = saver.save(sess, '{}/{}_{}_{}/lstm'.format(file_out, save_path, i, epoch))
                    with open('log_{}.txt'.format(file_out), 'a') as f:
                        f.write('saving model: {}/{}_{}_{}/lstm'.format(file_out, save_path, i, epoch))
                        print('saving model: {}/{}_{}_{}/lstm'.format(file_out, save_path, i, epoch))
                print(np.argmax(preds__, axis=1))
                print(np.argmax(bb_y, axis=1))
                print('VALID: epoch: {} - iteration: {} - acc: {} -- last_pred:'.format(epoch, i, mean_acc))
                with open('log_{}.txt'.format(file_out), 'a') as f:
                    f.write('VALID: epoch: {} - iteration: {} - acc: {} \n'.format(epoch, i, mean_acc))
