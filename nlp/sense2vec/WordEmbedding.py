from __future__ import division

import itertools
import logging
import mutex
import random
import re
from itertools import tee

import numpy as np
import pattern.en as pattern
from gensim.models.word2vec import Word2Vec
from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

TAG_RE = re.compile(r'<[^>]+>')


class WordModel:
    """
    Word embedded model supporting iterative updates 
    to the model consuming a block of text as unit
    
    :param dim         :: dimension of the embedded space
    :param window      :: context window size to consider for training  on context
    :param min_count   :: min count of the word for it be considered to be modeled
    :param parallelism :: degree of parallelism to use for training
                          (works only with cython installed)
    :param iterations  :: Number of iterations to run over the set of training examples
    :param sentence_func :: custom function to apply on each sentence of type f(s) = s1
                            where s and s1 are both strings(sentence)
    """

    def __init__(self, name, dim=100, window=10, min_count=1,
                 parallelism=1, use_stem=False, iterations=100, learning_rate=0.025):
        self.name = name
        self.dimension = dim
        self.window = window
        self.min_count = min_count
        self.parallelism = parallelism
        self.sample = 1
        self.iterations = iterations
        self.alpha = learning_rate
        self.sentence_func = self.strip_non_ascii
        self.stemmer = PorterStemmer().stem if use_stem else lambda e: e
        self.stop_words = stopwords.words('english')
        self.stopwords = {self.stemmer(w): True for w in self.stop_words}
        self.tokenizer = self.form_sentences
        self.train_lock = mutex.mutex()
        self.model = None

    def strip_non_ascii(self, string):
        """Returns the string without non ASCII characters"""
        stripped = (c for c in string if 0 < ord(c) < 127)
        return ''.join(stripped)

    def __sanitizer(self, word):
        if word == 'deliv': return 'deliveri'
        return word

    def __word_filter(self, word, remove_stopword=False):
        if word.isdigit(): return False
        if len(word) <= 2 and not word.isalnum(): return False
        if remove_stopword and self.stopwords.has_key(self.stemmer(word)): return False
        return True

    def __create_model__(func):
        """
        create the word2vec model for the parameters specified
        """
        def func_wrapper(self, *args, **kwargs):
            if not self.model:
                self.model = Word2Vec(size=self.dimension, window=self.window,
                                      min_count=self.min_count, workers=self.parallelism,
                                      min_alpha=self.alpha, sample=self.sample, negative=10)
            return func(self, *args, **kwargs)
        return func_wrapper

    def form_sentences(self, text_block, remove_stopwords=False, stem=True):
        """
        parse a block of text a form a list of word tokenized sentences 
        :param text_block : single block of text as string 
        :param id : id of the text_block, used for hdfs storage
        :param remove_stopwords: remove the stopwords from the text
        :param stem: stem the words to root form 
        """
        sentences = pattern.tokenize(text_block.lower())
        sentences = [sentence.replace('\'', '').replace('(', ' ').replace(')', ' ') \
                         .replace("/", " or ").replace("-", "") for sentence in sentences]
        sentences = [self.sentence_func(TAG_RE.sub('', sentence)) for sentence in sentences]

        l_stemmer = lambda w: self.stemmer(w) if stem else w
        sentences = [[l_stemmer(w) for w in word_tokenize(sentence)
                      if self.__word_filter(w, remove_stopwords)] for sentence in sentences]
        return sentences

    @__create_model__
    def batch_train_iterated(self, dataset_gen):
        """
        batch train the word embedding model given a dataset generator which 
        generates sentence by sentence where each sentence is a list of words 
        :param dataset_gen : dataset sentence generator
        """
        lock_value = self.train_lock.testandset()
        if not lock_value: raise RuntimeError("Training already in progress")

        vocab_iter, sentence_iter = dataset_gen
        model_new = Word2Vec(size=self.dimension, window=self.window,
                             min_count=self.min_count, workers=self.parallelism,
                             min_alpha=self.alpha, sample=self.sample, negative=10)

        # reset the learning rate to initial
        model_new.min_alpha = self.alpha
        model_new.alpha = model_new.min_alpha

        model_new.build_vocab(vocab_iter)
        step_size = self.alpha / self.iterations

        epoch_dataset_iter = tee(sentence_iter, self.iterations)
        for epoch in range(self.iterations):
            model_new.train(epoch_dataset_iter[epoch])
            model_new.alpha -= step_size
            logger.info("learning param for the model : %f" % model_new.alpha)
            model_new.min_alpha = model_new.alpha

        # point the model to newly trained model
        self.model = model_new
        self.train_lock.unlock()

    @__create_model__
    def batch_train(self, text_blocks, tokenizer=None, tokenized=False):
        """
        batch training given a set of text blocks, text blocks are held in-memory 
        :param text_blocks : list of text_block
        :param tokenizer : a tokenizer function which returns word tokenized list
        of sentences for a given text block
        :param tokenized: boolean indiccating if the text_blocks are
        already sentence and word tokenized, tokenizer will be ignored
        """
        lock_value = self.train_lock.testandset()
        if not lock_value: raise RuntimeError("Training in progress")

        sentences = []
        tokenizer = tokenizer if tokenizer else self.tokenizer

        if not tokenized:
            for text_block in text_blocks:
                block_sentences = tokenizer(text_block)
                sentences.extend(block_sentences)
        else:
            sentences = text_blocks

        model_new = Word2Vec(size=self.dimension, window=self.window,
                             min_count=self.min_count, workers=self.parallelism,
                             min_alpha=self.alpha, sample=self.sample, negative=10)

        # reset the learning rate to initial
        model_new.min_alpha = self.alpha
        model_new.alpha = model_new.min_alpha

        model_new.build_vocab(sentences)

        step_size = self.alpha / self.iterations
        for epoch in range(self.iterations):
            # shuffle the sentences, this improves the performance
            logger.info("training epoch :: %d" % epoch)
            random.shuffle(sentences)
            model_new.train(sentences)
            model_new.alpha -= step_size
            logger.info("learning factor for the model :: %f" % model_new.alpha)
            model_new.min_alpha = model_new.alpha
            logger.info("finished training the epoch")

        # point the model to newly created model
        self.model = model_new
        self.train_lock.unlock()

    @__create_model__
    def load_external(self, model_file_name):
        """
        load a word2vec model from the file specified
        :param model_file_name: name of the model file
        :return:
        """
        self.model = Word2Vec.load(model_file_name)

    def similarity(self, words):
        """
        compute the similarity between a set of words from the model 
        :param words :: list of (word, weight), words unknown to the model  
                        would be ignored, for entries which are more 
                        than single word would be replaced with the average 
                        
        """
        single_token_words = {w: True for w in words if len(w.split(" ")) == 1}
        known_word_vectors = [self.model[word] / np.linalg.norm(self.model[word]) for word \
                              in single_token_words.keys() if self.model.vocab.has_key(word)]
        multiple_token_words = [w for w in words if w not in single_token_words]
        word_vector_averages = list(itertools.chain(*[[self.model[t] for t in w.split(" ") if \
                                                       self.model.vocab.has_key(t)] for w \
                                                      in multiple_token_words]))
        word_vector_averages = [w / np.linalg.norm(w) for w in word_vector_averages]
        known_word_vectors.extend(word_vector_averages)
        word_vectors = list(known_word_vectors)

        # base case: #(words) = 1, return 1
        if len(word_vectors) == 1: return 1
        # base case: #(words) = 2, return the cosine similarity of the two vectors
        if len(word_vectors) == 2:
            v1, v2 = word_vectors[0], word_vectors[1]
            sim = np.dot(v1, v2)
            # we are not interested in negative similarity, pinning it to 0
            return sim if sim > 0 else 0
        sims = []
        for w in word_vectors:
            # remove w from the set of word vectors to compute the 
            # dot product of average of this set with the word vector w
            known_word_vectors.remove(w)
            unit_vec_candidate = np.average(known_word_vectors, axis=0)
            unit_vec_candidate /= np.linalg.norm(unit_vec_candidate)
            sims.append(np.dot(unit_vec_candidate, w))
            # add w back to the set of word vectors 
            known_word_vectors.append(w)
        avg_sim = np.average(sims)
        return avg_sim if avg_sim > 0 else 0

    def most_similar(self, words, count=10, include_stopwords=False,
                     stem=True):
        """
        get the most positively correlated words with the words 
        :param words :: non empty list of words or list of (word, weight)
        :param count :: number of similar words to retreive 
        :param inlcude_stopwords :: include stop words in result
        :param stem :: Stem the tokens of the query 
        """

        if not isinstance(words, list): raise RuntimeError("words must be a list")
        words = map(lambda w: w if isinstance(w, tuple) else (w, 1.0), words)
        words = [(w, s) for (w, s) in words if self.model.vocab.get(w)]
        if not words: return None
        to_stem = lambda t: self.stemmer if stem else t
        most_similar = self.model.most_similar(positive=[(self.__sanitizer(to_stem(word)),
                                                          weight) for word, weight in words],
                                               topn=2 * count)

        if include_stopwords: return most_similar[:count]
        # otherwise remove the stopwords
        return [(w, s) for (w, s) in most_similar if w not in self.stopwords][:count]

