import logging
import time
from multiprocessing import Pool, Manager
from threading import Thread

import pattern.en as pattern
from nltk.stem import PorterStemmer
from practnlptools import tools as pnt
from queue import Full, Empty

import nlp.relation_extraction.data_sink.sink as DSink
import nlp.relation_extraction.data_source.source as DSource
from nlp.relation_extraction import RelationModifier, RelationArgument, RelationTuple
from nlp.relation_extraction.relation_util import utils as relation_util
logger = logging.getLogger(__name__)


class RelationExtractor:
    """
    Relation Extraction based on Semantic Role Labeling of SENNA
    """
    def __init__(self, data_source=None, relation_sink=None, workers=8):
        """
        :param data_source: data_source object of type DataSource
        :param relation_sink: data_sink object of type DataSink
        :param workers: number of child process workers in source sink mode
        """
        if data_source:
            assert isinstance(data_source, DSource.MongoDataSource),\
                "data_source object must be instance of MongoDataSource"
            self.data_source = data_source

        if relation_sink:
            assert isinstance(relation_sink, DSink.ElasticDataSink), \
                "relation_sink object must be instance of ElasticDataSink"
            self.relation_sink = relation_sink
            self.model_class = self.relation_sink.model_identifier.model_class

        self.relation_annotator = pnt.Annotator()
        self.stemmer = PorterStemmer()
        self.workers = workers
        self.relation_queue = Manager().Queue(maxsize=10000)
        self.persist_attributes = ['relation_annotator', 'stemmer', 'model_class', 'relation_queue']

    def __getstate__(self):
        state = dict()
        for attr in self.persist_attributes:
            state[attr] = self.__dict__[attr]
        return state

    def __setstate(self, d):
        self.__dict__.update(d)

    @staticmethod
    def __populate_arguments(semantic_element):
        """
        form a argument object from the srl semantic element
        :param semantic_element: SRL semantic element
        :return: RelationArgument instance
        """
        return RelationArgument(A0=semantic_element.get('A0'), A1=semantic_element.get('A1'),
                                A2=semantic_element.get('A2'), A3=semantic_element.get('A3'))

    @staticmethod
    def __populate_modifier(semantic_element):
        """
        form a argument modifier object from the srl semantic element
        :param semantic_element: SRL semantic element
        :return: RelationModifier instance
        """
        return RelationModifier(DIR=semantic_element.get('AM-DIR'), MNR=semantic_element.get('AM-MNR'),
                                LOC=semantic_element.get('AM-LOC'), TMP=semantic_element.get('AM-TMP'),
                                EXT=semantic_element.get('AM-EXT'), PNC=semantic_element.get('AM-PNC'),
                                CAU=semantic_element.get('AM-CAU'), NEG=semantic_element.get('AM-NEG'))

    def form_relations(self, text, persist=True):
        """
        form relation(s) on a given text
        :param text: text on which to get the relations on,
        text will be sentence tokenized and relations formed at sentence level
        :param persist: persist the relations extracted from the text in the sink,
        relation_sink needed to be specified
        :return: list of relations
        """
        text_sentences = pattern.tokenize(text)
        relations = []
        for sentence in text_sentences:

            # work with ascii string only
            sentence = "".join((c for c in sentence if 0 < ord(c) < 127))
            senna_annotation = self.relation_annotator.getAnnotations(sentence)

            chunk_parse, pos_tags, role_labeling, tokenized_sentence = \
                senna_annotation['chunk'], senna_annotation['pos'], senna_annotation['srl'], \
                senna_annotation['words']

            # nothing to do here empty srl
            if not role_labeling: continue

            for semantic_element in role_labeling:
                arguments = RelationExtractor.__populate_arguments(semantic_element)
                modifiers = RelationExtractor.__populate_modifier(semantic_element)
                verb = semantic_element.get('V')
                # order of the arguments returned is important, A0 --> A1 --> A2 --> A3
                arguments = [v for v in vars(arguments).itervalues() if v]
                modifiers = [v for v in vars(modifiers).itervalues() if v]

                if not arguments: continue
                argument_pairs = [e for e in ((ai, aj) for i, ai in enumerate(arguments) for j, aj
                                              in enumerate(arguments) if i < j)]

                verb = relation_util.normalize_relation(verb)
                for a0, a1 in argument_pairs:
                    en0 = relation_util.form_entity(tokenized_sentence, a0, chunk_parse, pos_tags)
                    en1 = relation_util.form_entity(tokenized_sentence, a1, chunk_parse, pos_tags)
                    if not en0 or not en1: continue
                    relations.append(RelationTuple(left_entity=en0, right_entity=en1, relation=verb,
                                                   sentence=sentence))
                for arg_modifier in modifiers:
                    mod_pos = sentence.find(arg_modifier)
                    linked_arg = min([(a, abs(mod_pos - sentence.find(a))) for a in arguments], key=lambda e: e[1])[0]
                    en0 = relation_util.form_entity(tokenized_sentence, linked_arg, chunk_parse, pos_tags)
                    en1 = relation_util.form_entity(tokenized_sentence, arg_modifier, chunk_parse, pos_tags)
                    if not en0 or not en1: continue
                    relations.append(RelationTuple(left_entity=en0, right_entity=en1, relation=verb,
                                                   sentence=sentence))

        return relations

    def form_relations_source(self, source_item):
        if not source_item:
            logger.error("got an empty source item")
            return

        for item_entry in source_item:
            if item_entry == ' ': continue
            try:
                relations = self.form_relations(item_entry)
            except RuntimeError as e:
                logger.error("Error generating relations")
                logger.error(e)
                continue

            for relation in relations:
                sink_relation = self.model_class()
                sink_relation.leftEntity = relation.left_entity
                sink_relation.rightEntity = relation.right_entity
                sink_relation.relation = relation.relation
                sink_relation.text = relation.sentence
                logger.info("generated a relation")
                logger.info(sink_relation)

                try:
                    self.relation_queue.put(sink_relation, timeout=1)
                except Full as e:
                    logger.error(e)

    def sink_relations(self):
        while not self.all_sinked:
            try:
                item = self.relation_queue.get_nowait()
                self.relation_sink.sink_item(item)
            except Empty as e:
                pass

    def form_relations_from_source(self):

        if not self.data_source or not self.relation_sink:
            raise RuntimeError("Data source and sink must be set")

        self.data_source.start()
        self.relation_sink.start()

        self.all_sinked = False
        pool = Pool(processes=self.workers)
        t1 = time.time()
        pool.imap(self.form_relations_source, self.data_source, chunksize=8)

        sinker = Thread(target=self.sink_relations, name='Sink-Thread')
        sinker.start()

        pool.close()
        pool.join()
        self.all_sinked = True
        t2 = time.time()
        logger.info("process finished in :: %d  seconds" %(t2 - t1))

