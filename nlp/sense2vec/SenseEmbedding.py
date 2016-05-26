import logging
import re
import numpy as np
from itertools import izip, chain
from multiprocessing import Pool, Manager

from enum import Enum
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from practnlptools import tools

import nlp.relation_extraction.data_source.source as DSource
from nlp.relation_extraction.relation_util import utils as relation_util
from nlp.sense2vec import CONJUNCTION, CARDINAL, ADJECTIVE, NOUN, VERB
from nlp.sense2vec import WordEmbedding

logger = logging.getLogger(__name__)

SENT_RE = re.compile(r"([A-Z]*[^\.!?]*[\.!?])", re.M)
TAG_RE = re.compile(r"<[^>]+>")


class SenseEmbedding(WordEmbedding.WordModel):
    """
        Implementation of Sense2Vec;  NP, VP and POS tag based embedding
        reference : http://arxiv.org/pdf/1511.06388v1.pdf
        """
    class VPTags(Enum):
        single = 'S-VP'
        begin = 'B-VP'
        intermediate = 'I-VP'
        end = 'E-VP'

    class NPTags(Enum):
        single = 'S-NP'
        begin = 'B-NP'
        intermediate = 'I-NP'
        end = 'E-NP'

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
        self.phrase_tags = set([e.value for e in chain(*[SenseEmbedding.NPTags, SenseEmbedding.VPTags])])
        self.workers = workers
        self.tokenized_blocks = Manager().list()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    @staticmethod
    def __form_phrases(chunk_parse, sense):

        if sense not in [SenseEmbedding.NPTags, SenseEmbedding.VPTags]:
            raise RuntimeError("Sense must be NPTags or VPTags Enum")

        current_sense, phrases = [], []
        for word, chunk_tag in chunk_parse:
            if chunk_tag == sense.single.value:
                phrases.append(word)

            if chunk_tag in [sense.begin.value, sense.intermediate.value]:
                current_sense.append(word)

            if chunk_tag == sense.end.value:
                current_sense.append(word)
                phrases.append(" ".join(current_sense))
                current_sense = []

        return phrases

    @staticmethod
    def normalize_pos(pos_tag):

        if pos_tag in CONJUNCTION:
            return 'CONJUNCTION'
        elif pos_tag in CARDINAL:
            return 'CARDINAL'
        elif pos_tag in ADJECTIVE:
            return 'ADJECTIVE'
        elif pos_tag in NOUN:
            return 'NOUN'
        elif pos_tag in VERB:
            return 'VERB'
        else:
            return 'DEFAULT'

    def sense_tokenize(self, text_block):
        """
        tokenize a block into sentences which are word tokenized, preserving the sense of the words
        (see the original paper for details)
        :param text_block: block of text (string)
        :return: list of sentences each tokenized into words
        """
        sentences = SENT_RE.findall(text_block)

        for sentence in sentences:
            sentence = sentence.replace('\'', '').replace('(', ' ')\
                .replace(')', ' ').replace("/", " or ").replace("-", "")

            sentence = TAG_RE.sub('', sentence)
            sentence = "".join((c for c in sentence if 0 < ord(c) < 127))
            logger.info("Will sense tokenize : %s" %sentence)
            try:
                senna_annotation = self.annotator.getAnnotations(sentence)
            except Exception as e:
                logger.error("annontator error")
                logger.error(e)
                continue

            chunk_parse, pos_tags, words = senna_annotation['chunk'], senna_annotation['pos'], \
                                           senna_annotation['words']

            single_words = [self.stemmer.stem(word) + '|' + SenseEmbedding.normalize_pos(tag)
                            for word, tag in pos_tags if word not in self.stop_words]
            self.tokenized_blocks.append(single_words)

            noun_phrases = SenseEmbedding.__form_phrases(chunk_parse, SenseEmbedding.NPTags)
            verb_phrases = SenseEmbedding.__form_phrases(chunk_parse, SenseEmbedding.VPTags)

            non_phrase_words = [self.stemmer.stem(word) + '|' + SenseEmbedding.normalize_pos(pos_tag) for
                                ((word, chunk_tag), (_, pos_tag)) in izip(chunk_parse, pos_tags)
                                if chunk_tag not in self.phrase_tags if word not in self.stop_words]

            noun_entities, verb_entities = [], []
            for np in noun_phrases:
                en = relation_util.form_entity(words, np, chunk_parse, pos_tags, 'NP')
                if not en: continue
                noun_entities.append(en + '|NP')

            for vp in verb_phrases:
                en = relation_util.form_entity(words, vp, chunk_parse, pos_tags, 'VP')
                if not en: continue
                verb_entities.append(en + '|VP')

            noun_index, verb_index, non_phrase_index = 0,0,0
            sense_words = []
            for (word, chunk_tag) in chunk_parse:
                if chunk_tag not in self.phrase_tags:
                    if non_phrase_index < len(non_phrase_words):
                        sense_words.append(non_phrase_words[non_phrase_index])
                        non_phrase_index += 1

                if chunk_tag in [SenseEmbedding.NPTags.end.value, SenseEmbedding.NPTags.single.value]:
                    if noun_index < len(noun_entities):
                        sense_words.append(noun_entities[noun_index])
                        noun_index += 1

                if chunk_tag in [SenseEmbedding.VPTags.end.value, SenseEmbedding.VPTags.single.value]:
                    if verb_index < len(verb_entities):
                        sense_words.append(verb_entities[verb_index])
                        verb_index += 1

            if sense_words: self.tokenized_blocks.append(sense_words)

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

            for item in item_tuple:
                if item == '': continue
                text_blocks.append(item)

        logger.info("Read all the text blocks")
        logger.info("Number of text blocks read : %d" % len(text_blocks))
        logger.info("will sentence and word tokenize the text blocks")

        pool = Pool(processes=self.workers)
        pool.map(self.sense_tokenize, text_blocks,chunksize=2*self.workers)
        pool.close()
        pool.join()
        self.batch_train(text_blocks=self.tokenized_blocks, tokenized=True)
