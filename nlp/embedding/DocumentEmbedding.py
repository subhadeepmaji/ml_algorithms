from __future__ import division

import logging
import mutex
import random
import re
import pattern.en as pattern
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)
TAG_RE = re.compile(r'<[^>]+>')


class SentenceModel:
    """
    Sentence embedded model supporting iterative updates
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
        self.doc_tags = {}
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
                self.model = Doc2Vec(size=self.dimension, window=self.window,
                                      min_count=self.min_count, workers=self.parallelism,
                                      min_alpha=self.alpha, sample=self.sample, negative=10)
            return func(self, *args, **kwargs)
        return func_wrapper

    def form_sentences(self, text_block, block_id, remove_stopwords=False,
                       stem=True, form_tagged_doc=True):
        """
        parse a block of text a form a list of word tokenized sentences
        :param text_block : single block of text as string
        :param block_id: id of the text block
        :param id : id of the text_block, used for hdfs storage
        :param remove_stopwords: remove the stopwords from the text
        :param stem: stem the words to root form
        :param form_tagged_doc: form a tagged document for the Doc2vec model
        """
        sentences = pattern.tokenize(text_block.lower())
        sentences = [sentence.replace('\'', '').replace('(', ' ').replace(')', ' ') \
                         .replace("/", " or ").replace("-", "") for sentence in sentences]
        sentences = [self.sentence_func(TAG_RE.sub('', sentence)) for sentence in sentences]

        l_stemmer = lambda w: self.stemmer(w) if stem else w
        sentences = [[l_stemmer(w) for w in word_tokenize(sentence)
                      if self.__word_filter(w, remove_stopwords)] for sentence in sentences]

        if not form_tagged_doc:
            return sentences

        sentences = [TaggedDocument(words=words, tags=[str(block_id) + ' ' + str(index)])
                     for index, words in enumerate(sentences)]

        for sentence in sentences:
            self.doc_tags[sentence.tags[0]] = sentence

        return sentences

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
            for block_id, text_block in enumerate(text_blocks):
                block_sentences = tokenizer(text_block, block_id)
                sentences.extend(block_sentences)
        else:
            sentences = text_blocks

        logger.info("Number of sentences formed :: %d" %len(sentences))
        model_new = Doc2Vec(size=self.dimension, window=self.window,
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
        self.model = Doc2Vec.load(model_file_name)

    def most_similar(self, sentence, count=10, stem=True):
        """
        get the most positively correlated sentences with this sentence
        :param sentence :: sentenec to compute similarity for
        :param count :: number of similar words to retrieve
        :param stem :: Stem the tokens of the query
        """
        sentence = self.tokenizer(sentence, block_id=0, form_tagged_doc=False)[0]
        sentence_vec = self.model.infer_vector(sentence)
        return self.model.docvecs.most_similar([sentence_vec], topn = count)
