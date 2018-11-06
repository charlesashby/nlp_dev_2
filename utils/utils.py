import re, unicodedata


def words2vec(model, sentence):
    tokens = sentence.split(' ')
    vectors = []
    for token in tokens:
        vec = model.get_vector(token)
        vectors.append(vec)
    return vectors


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