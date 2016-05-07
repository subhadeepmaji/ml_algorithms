
import pattern.en as pattern
import nlp.relation_extraction.data_source.source as DSource
from nlp.sense2vec import WordEmbedding
from nlp.relation_extraction.relation_util import utils as relation_util
from practnlptools import tools
from itertools import izip, chain
from multiprocessing import Pool, Manager
from enum import Enum
from nlp.sense2vec import CONJUNCTION, CARDINAL, ADJECTIVE, NOUN, VERB
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import logging,re
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

    def __init__(self, data_sources, workers, *args, **kwargs):
        """
        Sense2vec embedding
        :param data_sources: list of data sources to pull data from
        :param workers: number of processes to create in the pool
        """
        WordEmbedding.WordModel.__init__(self, *args, **kwargs)
        assert isinstance(data_sources, list), "data sources must be a list"
        for source in data_sources:
            assert isinstance(source, DSource.MongoDataSource), "source must be instance of " \
                                                                "MongoDataSource"
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
            senna_annotation = self.annotator.getAnnotations(sentence)
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
