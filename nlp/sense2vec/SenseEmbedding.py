import logging
from itertools import chain
from multiprocessing import Pool, Manager

import numpy as np
from enum import Enum
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from practnlptools import tools
from collections import defaultdict

from nlp.embedding import WordEmbedding
from nlp.sense2vec import sense_tokenize

logger = logging.getLogger(__name__)


class SenseEmbedding(WordEmbedding.WordModel):
    """
        Implementation of Sense2Vec;  NP, VP and POS tag based embedding
        reference : http://arxiv.org/pdf/1511.06388v1.pdf
        """

    # DO NOT change this ordering, need to figure out a better way to achieve this
    senses = ['NOUN', 'VERB', 'ADJECTIVE', 'CONJUNCTION', 'CARDINAL', 'DEFAULT']

    def __init__(self, data_sources, workers, *args, **kwargs):
        """
        Sense2vec embedding
        :param data_sources: list of data sources to pull data from
        :param workers: number of processes to create in the pool
        """
        WordEmbedding.WordModel.__init__(self, *args, **kwargs)
        self.sources = data_sources
        self.annotator = tools.Annotator()
        self.workers = workers
        self.tokenized_blocks = Manager().list()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.word_to_tag = defaultdict(list)

    def form_tag_tokens(self):
        for word_tag in self.model.vocab:
            word, tag = word_tag.split("|")
            self.word_to_tag[word].append(tag)

    def get_tags_for_word(self, word):
        token_tags = self.word_to_tag.get(word, None)
        if not token_tags: return []
        return [word + "|" + tag for tag in token_tags]

    def tokenize(self, text_block):
        sense_phrases = sense_tokenize(text_block, self.annotator, self.stemmer, self.stop_words)
        self.tokenized_blocks.extend(sense_phrases)

    def get_sense_vec(self, entity, dimension, sense='NOUN'):

        if sense == 'NOUN':
            if self.model.vocab.has_key(entity + '|NOUN'):
                return self.model[entity + '|NOUN']

            elif self.model.vocab.has_key(entity + '|NP'):
                return self.model[entity + '|NP']

            else:
                entities = entity.split(" ")
                entity_vec = [self.model[e + '|NOUN'] for e in entities if e + '|NOUN'
                              in self.model.vocab]
                entity_vec.extend([self.get_vector(e, dimension, 'NOUN') for e in entities
                                   if e + '|NOUN' not in self.model.vocab])
                return np.average(entity_vec, axis=0)

        else:
            if self.model.vocab.has_key(entity + '|VERB'):
                return self.model[entity + '|VERB']

            elif self.model.vocab.has_key(entity + '|VP'):
                return self.model[entity + '|VP']

            else:
                entities = entity.split(" ")
                entity_vec = [self.model[e + '|VERB'] for e in entities if e + '|VERB'
                              in self.model.vocab]
                entity_vec.extend([self.get_vector(e, dimension, 'VERB') for e in entities
                                   if e + '|VERB' not in self.model.vocab])
                return np.average(entity_vec, axis=0)

    def get_vector(self, word, dimension, sense_except='NOUN'):

        words = [word] * (len(SenseEmbedding.senses) - 1)
        senses = list(SenseEmbedding.senses)
        senses.remove(sense_except)
        word_with_sense = [w + '|' + s for w,s in zip(words, senses)]
        for word in word_with_sense:
            if self.model.vocab.has_key(word):
                return self.model[word]

        return np.random.normal(0, 1, dimension)

    def form_model(self):
        text_blocks = []
        for source in self.sources:
            source.start()

        logger.info("Reading the text blocks from the source")
        for item_tuple in chain(*self.sources):
            if not item_tuple:
                logger.warn("item read from source is empty")
                continue

            item = " ".join([t[1] for t in item_tuple])
            if item == '': continue
            text_blocks.append(item)

        logger.info("Read all the text blocks")
        logger.info("Number of text blocks read : %d" % len(text_blocks))
        logger.info("will sentence and word tokenize the text blocks")

        pool = Pool(processes=self.workers)
        pool.map(self.tokenize, text_blocks,chunksize=2*self.workers)
        pool.close()
        pool.join()
        self.batch_train(text_blocks=self.tokenized_blocks, tokenized=True)
        # form the token to tags map
        self.form_tag_tokens()
