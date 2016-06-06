from __future__ import division

import sys
import logging
from multiprocessing import Pool, Manager
from collections import Counter
from itertools import chain
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from practnlptools import tools
from hashlib import sha256

from nlp.sense2vec import sense_tokenize
from nlp.sense2vec import SenseEmbedding as SE
from nlp.understand_query import QueryTiler as QT

logger = logging.getLogger(__name__)

reload(sys)
sys.setdefaultencoding('utf8')


class WordMoverModel:
    """
    word mover model for information retrieval on bag of vectors model
    using distance computation on queries and documents using an upper bound approximation
    of Earth Mover Distance

    reference : http://jmlr.org/proceedings/papers/v37/kusnerb15.pdf
    """
    def __init__(self, data_sources, workers, embedding):
        self.sources = data_sources
        self.workers = workers
        self.tokenized_blocks = Manager().list()
        self.annotator = tools.Annotator()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        assert isinstance(embedding, SE.SenseEmbedding), "embedding must be instance of SenseEmbedding"
        self.embedding = embedding
        self.document_models = {}
        self.query_tiler = QT.QueryTiler(self.embedding)

    def tokenize(self, block_input):
        block_id, text_block = block_input
        sense_phrases = sense_tokenize(text_block, self.annotator, self.stemmer, self.stop_words)
        self.tokenized_blocks.append((block_id, sense_phrases, text_block))

    def form_doc_bags(self):
        for block_id, sense_phrases, block in self.tokenized_blocks:
            sense_words = set(chain(*sense_phrases))
            # ignore oov words of the documents, ideally this should not happen
            # if the model is trained on the document data
            sense_words = [(w,self.embedding.model[w]) for w in sense_words if
                           self.embedding.model.vocab.has_key(w)]
            if sense_words:
                self.document_models[block_id] = (sense_words, block)

    def assign_nearest_nbh(self, query_words, doc_words):
        dist_arr = np.array([[np.linalg.norm(v - q, ord=2) for _,v in doc_words] for q in query_words])
        label_assignment = np.argmin(dist_arr, axis=1)
        label_assignment = [(index, l) for index, l in enumerate(label_assignment)]
        distances = [dist_arr[l] for l in label_assignment]
        labels = [doc_words[l[1]][0] for l in label_assignment]
        return distances, labels

    def word_mover_distance(self, query, topn=10):
        """
        get the set of documents nearest to the query , as bag of senses EMD
        :param query: query as string will be white space tokenized
        :param topn: number of nearest documents to return, avoid more than 10
        :return: [(block_id, block)] for the topn nearest document neighbours of the query
        """
        tiled_words = self.query_tiler.tile(query, include_stopwords=True)[0]
        if not tiled_words: raise RuntimeError("query could not be tiled by the embedding model")
        tiled_words = [self.embedding.model[w] for w in tiled_words if
                       self.embedding.model.vocab.has_key(w)]
        #uniq_query_tokens = Counter(tiled_words)
        #token_counts = [uniq_query_tokens[t] / len(tiled_words) for t in tiled_words]
        neighbours = []

        for block_id, (sense_words, block) in self.document_models.iteritems():
            distances, labels = self.assign_nearest_nbh(tiled_words, sense_words)
            neighbours.append((block_id, block, np.sum(distances)))

        neighbours = sorted(neighbours, key=lambda e: e[2])
        return neighbours[:topn]

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
                item_id = sha256(item).hexdigest()
                text_blocks.append((item_id, item))

        logger.info("Read all the text blocks")
        logger.info("Number of text blocks read : %d" % len(text_blocks))
        logger.info("will sentence and word tokenize the text blocks")

        pool = Pool(processes=self.workers)
        pool.map(self.tokenize, text_blocks, chunksize=2 * self.workers)
        pool.close()
        pool.join()
        self.form_doc_bags()







