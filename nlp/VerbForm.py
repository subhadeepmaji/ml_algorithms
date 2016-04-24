from collections import defaultdict
from sklearn import svm
import pattern.en as pattern
import numpy as np
import random

CORPUS_FILE = "./data/corpus.txt"
POS_FILE = "./data/pos_tags.txt"

VERBS = ['am', 'are', 'were', 'was', 'is', 'been', 'being', 'be']


class FeatureGenerator(object):

    def __init__(self, window):
        self.window = window

        f_p = open(POS_FILE, "rbU")
        self.pos_tags = f_p.read().split("\n")
        self.pos_tags = {pos_tag: i for i, pos_tag in enumerate(self.pos_tags)}
        f_p.close()
        self.vec_size = len(self.pos_tags)

    def vectorize(self, pos_tags):
        pos_vec = np.zeros(shape=(self.window - 1, self.vec_size,), dtype=np.int16)
        for index, tag_word in enumerate(pos_tags):
            pos_tag = tag_word[1]
            pos_vec[index][self.pos_tags[pos_tag]] = 1

        pos_vec = pos_vec.reshape(((self.window - 1) * self.vec_size,))
        return pos_vec

    def pos_tag_sentence(self, sentence, verb='^'):
        half_window = (self.window / 2)

        sentence = sentence.decode('utf8', 'replace')
        sentence_pos = pattern.tag(sentence)
        sentence_pos = [e for e in sentence_pos if e[1] in self.pos_tags]
        verb_pos = [v_p for v_p, e in enumerate(sentence_pos) if e[0] == verb][0]
        start_pos, end_pos = verb_pos - half_window, verb_pos + half_window

        start_pos = 0 if start_pos < 0 else start_pos
        word_context = sentence_pos[start_pos: end_pos]

        # pad the first element to form equal length vectors
        if len(word_context) < self.window:
            window_append = [word_context[0]] * (self.window - len(word_context))
            window_append.extend(word_context)
            word_context = window_append

        word_context.pop(half_window)
        return word_context


class CorpusReader(object):
    def __init__(self, window):
        self.window = window
        assert self.window % 2 == 0, "window must be positive even integer"

    def form_dataset(self):
        self.feature_gen = FeatureGenerator(self.window)
        self.form_sentences()
        self.form_POS_vectors()

    def form_sentences(self):
        f_p = open(CORPUS_FILE, "rbU")
        corpus_sentences = pattern.tokenize(f_p.read())
        f_p.close()
        self.sentences = defaultdict(list)
        for sentence in corpus_sentences:
            for v in VERBS:
                if sentence.find(" " + v + " ") != -1:
                    self.sentences[v].append(sentence)

    def form_POS_vectors(self):
        self.X, self.Y = [], []
        for index, verb in enumerate(VERBS):
            sentences_of_verb = self.sentences[verb]
            for sentence in sentences_of_verb:
                word_context = self.feature_gen.pos_tag_sentence(sentence, verb)
                if word_context:
                    self.X.append(self.feature_gen.vectorize(word_context))
                    self.Y.append(index)


class VerbFormClassifier(object):
    def __init__(self, window, labels=None):
        if not labels:
            self.labels = range(len(VERBS))
        else:
            self.labels = labels

        self.feature_gen = FeatureGenerator(window)

    def train(self, X_train, Y_train):
        dataset = zip(X_train, Y_train)
        random.shuffle(dataset)

        self.models = {}
        for label in self.labels:
            X = [e[0] for e in dataset]
            Y = map(lambda e: 1 if e[1] == label else -1, dataset)
            svc_model = svm.LinearSVC(C=0.1, max_iter=15000, penalty='l2',
                                      tol=1e-5, dual=True,
                                      class_weight='balanced', verbose=100,
                                      fit_intercept=True, intercept_scaling=10)

            svc_model.fit(X, Y)
            self.models[label] = svc_model

    def predict_vec(self, X_test):

        predicted = np.zeros(shape=(len(self.labels), len(X_test)))
        for label in self.labels:
            p = self.models[label].decision_function(X_test)
            predicted[label] = p
        return predicted

    def predict(self, sentences):
        sentence_vecs = []
        for sentence in sentences:
            word_context = self.feature_gen.pos_tag_sentence(sentence)
            sentence_vecs.append(self.feature_gen.vectorize(word_context))

        predicted = np.zeros(shape=(len(self.labels), len(sentence_vecs)))
        for label in self.labels:
            p = self.models[label].decision_function(sentence_vecs)
            predicted[label] = p
        return predicted
