import os

data_path = '/run/media/ashbylepoc/b79b0a3e-a5b9-41ed-987f-8fa4bdb6b2e6/tmp/data/nlp_dev_2/'
train_y = os.path.join(data_path, 'train.y')
train_en = os.path.join(data_path, 'train.en')
train_fr = os.path.join(data_path, 'train.fr')
lexicon = os.path.join(data_path, 'lexique.en-fr')

lines_train_y = open(train_y).readlines()
lines_train_en = open(train_en).readlines()
lines_train_fr = open(train_fr).readlines()
lines_lexicon = open(lexicon).readlines()

# get max number of tokens for en and fr
# max_en = 304, max_fr = 402
max_en = 0
max_fr = 0
for i, _ in enumerate(lines_train_en):
    n_tokens_en = len(lines_train_en[i].split(' '))
    n_tokens_fr = len(lines_train_fr[i].split(' '))
    if n_tokens_en > max_en:
        max_en = n_tokens_en
    if n_tokens_fr > max_fr:
        max_fr = n_tokens_fr


# get number of words for french and english
# and frequencies
en_words = {}
fr_words = {}
for i, l in enumerate(lines_lexicon):
    if i % 10000 == 0:
        print(i)
    en_word, fr_word = l.split('\t')
    fr_word = fr_word[:-1]
    if en_word in en_words:
        en_words[en_word] += 1.
    else:
        en_words[en_word] = 1.
    if fr_word in fr_words:
        fr_words[fr_word] += 1.
    else:
        fr_words[fr_word] = 1.


# Shuffling sentences

