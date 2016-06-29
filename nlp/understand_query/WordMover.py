from __future__ import division

import sys, re
import logging
import multiprocessing as mlp
from itertools import chain, product
import scipy as sp
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from practnlptools import tools
from hashlib import sha256
from collections import defaultdict

from nlp.sense2vec import sense_tokenize
from nlp.sense2vec import SenseEmbedding as SE
from nlp.understand_query import QueryTiler as QT
from nlp.relation_extraction import RelationEmbedding as RE
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import spectral_clustering


logger = logging.getLogger(__name__)

reload(sys)
sys.setdefaultencoding('utf8')
alpha_numeric = re.compile('^[\s\w]+$')

has_pool = False
module_pool = None


def get_pool():
    global module_pool
    if has_pool:
        return module_pool
    query_pool = mlp.Pool(mlp.cpu_count())
    module_pool = query_pool
    return module_pool


class WordMoverModel:
    """
    word mover model for information retrieval on bag of vectors model
    using distance computation on queries and documents using an upper bound approximation
    of Earth Mover Distance

    reference : http://jmlr.org/proceedings/papers/v37/kusnerb15.pdf
    """
    def __init__(self, data_source, workers, embedding, alpha=0.8):
        self.source = data_source
        self.workers = workers
        self.tokenized_blocks = None
        self.annotator = tools.Annotator()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        assert isinstance(embedding, SE.SenseEmbedding), "embedding must be instance of SenseEmbedding"
        self.embedding = embedding
        self.alpha = alpha
        self.document_models = {}
        self.query_tiler = QT.QueryTiler(self.embedding)

    def tokenize(self, block_input):
        block_id, text_block = block_input
        sense_phrases = sense_tokenize(text_block, self.annotator, self.stemmer, self.stop_words)
        return block_id, sense_phrases, text_block

    def _filter_word(self, word_sense):
        word = word_sense.split("|")[0]
        if word in self.embedding.stop_words:
            return False
        return True if re.match(alpha_numeric, word) else False

    def form_distance_matrix(self):
        self.vocab = [w for w in self.embedding.model.vocab if self._filter_word(w)]
        self.vocab = {w: index for index, w in enumerate(self.vocab)}
        self.reverse_vocab = {index: w for index, w in enumerate(self.vocab)}
        words = sorted(self.vocab.iteritems(), key=lambda e: e[1])
        word_vectors = [self.embedding.model[w].astype(np.float64) for w,_ in words
                        if self.vocab.has_key(w)]
        self.distance_matrix = euclidean_distances(word_vectors)

    def vectorize(self, words):
        vector_words = np.zeros(len(self.vocab))
        p_indices = [self.vocab[w] for w in words]
        vector_words[p_indices] = 1
        return vector_words

    def form_doc_bags(self):
        for block_id, sense_phrases, block in self.tokenized_blocks:
            sense_words = set(chain(*sense_phrases))
            # ignore oov words of the documents, ideally this should not happen
            # if the model is trained on the document data
            sense_words = [w for w in sense_words if self.vocab.has_key(w)]
            if sense_words:
                self.document_models[block_id] = (sense_words, block)

    def tokenizer(self, block):
        return block.split(" ")

    def form_idf_model(self):
        documents = []
        for _, sense_phrases, _ in self.tokenized_blocks:
            sense_words = list(chain(*sense_phrases))
            documents.append(" ".join(sense_words))

        tfidf_model = TfidfVectorizer(lowercase=False, analyzer='word', tokenizer=self.tokenizer)
        tfidf_model.fit(documents)

        word_level_features = {index: feature for (index, feature) in enumerate(tfidf_model.get_feature_names())}
        self.word_level_idf = {self.vocab[word_level_features[index]] : idf for (index, idf)
                               in enumerate(tfidf_model.idf_) if self.vocab.has_key(word_level_features[index])}

    @staticmethod
    def label_distance(word1, word2):
        l1 = word1.split("|")[1]
        l2 = word2.split("|")[1]
        return 0 if l1 == l2 else 1

    def assign_nearest_nbh(self, query_doc):

        block_id, query_words, doc_words = query_doc
        query_vector = self.vectorize(query_words)
        doc_vector = self.vectorize(doc_words)
        #distance = emd(query_vector, doc_vector, self.distance_matrix)
        #return block_id, distance

        doc_indices = np.nonzero(doc_vector)[0]
        query_indices = np.nonzero(query_vector)[0]

        query_weights = [self.word_level_idf.get(q_i, 0) for q_i in query_indices]
        doc_weights = [self.word_level_idf.get(d_i, 0) for d_i in doc_indices]

        doc_centroid = np.average([self.embedding.model[self.reverse_vocab[i]] for i in doc_indices], axis=0,
                                  weights=doc_weights)
        query_centroid = np.average([self.embedding.model[self.reverse_vocab[i]] for i in query_indices], axis=0,
                                    weights=query_weights)

        # sklearn euclidean distances may not be a symmetric matrix, so taking
        # average of the two entries
        dist_arr = np.array([[(self.distance_matrix[w_i, q_j] + self.distance_matrix[q_j, w_i]) / 2
                              for w_i in doc_indices] for q_j in query_indices])

        label_assignment = np.argmin(dist_arr, axis=1)
        label_assignment = [(index, l) for index, l in enumerate(label_assignment)]

        distances = [dist_arr[(i,e)] * self.word_level_idf.get(query_indices[i], 1) for i, e in label_assignment]

        distance = (1 - self.alpha) * np.sum(distances) + \
                   self.alpha * sp.spatial.distance.cosine(doc_centroid,query_centroid)
        return block_id, distance

    def word_mover_distance(self, query, tiled=False, topn=10):
        """
        get the set of documents nearest to the query , as bag of senses EMD
        :param query: query as string will be white space tokenized
        :param topn: number of nearest documents to return, avoid more than 10
        :return: [(block_id, block)] for the topn nearest document neighbours of the query
        """
        if not tiled:
            tiled_query_words = self.query_tiler.tile(query, include_stopwords=False)[0]
        else:
            tiled_query_words = query

        if not tiled_query_words: raise RuntimeError("query could not be tiled by the embedding model")
        tiled_query_words = [w for w in tiled_query_words if self.vocab.has_key(w)]
        candidates = [(block_id, tiled_query_words, doc_words) for block_id, (doc_words, _) in
                      self.document_models.iteritems()]
        #pool = Pool(processes=4)
        neighbours = map(self.assign_nearest_nbh, candidates)
        neighbours = sorted(neighbours, key=lambda e: e[1])
        return neighbours[:topn]

    def form_model(self):
        self.form_distance_matrix()
        text_blocks = []
        self.source.start()

        logger.info("Reading the text blocks from the source")
        for item_tuple in self.source:
            if not item_tuple:
                logger.warn("item read from source is empty")
                continue

            item = ''
            for f_name, f_value in item_tuple:
                item += f_value
            if item == '': continue
            item_id = sha256(f_value).hexdigest()
            text_blocks.append((item_id, item))

        logger.info("Read all the text blocks")
        logger.info("Number of text blocks read : %d" % len(text_blocks))
        logger.info("will sentence and word tokenize the text blocks")

        pool = mlp.Pool(4)
        self.tokenized_blocks = pool.map(self.tokenize, text_blocks, chunksize=self.workers)
        pool.close()
        pool.join()
        self.form_doc_bags()


class WordMoverModelRelation:
    """
    word mover model for document retrieval on approx EMD computation of
    relation embedding distances of query and documents
    """
    def __init__(self, relation_blocks, relation_embedding):
        self.relation_blocks = relation_blocks
        assert isinstance(relation_embedding, RE.TransHEmbedding), "relation embedding must be an instance " \
                                                                   "of TransHEmbedding"
        self.relation_embedding = relation_embedding
        self.relation_by_doc = defaultdict(list)
        self.doc_text = {}

    def form_doc_relations(self):
        for relation in self.relation_blocks:
            self.relation_by_doc[relation.block_id].append((relation.left_entity, relation.relation,
                                                            relation.right_entity))
            self.doc_text[relation.block_id] = relation.text

    def cluster_relations(self, affinity_matrix=None, num_clusters=10):

        labels = spectral_clustering(affinity=affinity_matrix, n_clusters=num_clusters)
        clustering_labels = zip([(e.left_entity, e.relation, e.right_entity) for e
                                 in self.relation_embedding.kb_triples], labels)
        self.clusters = defaultdict(list)
        self.labeling = {}

        for index, (triple, label) in enumerate(clustering_labels):
            self.clusters[label].append((index, triple))
            self.labeling[triple] = label

        self.cluster_affinity = np.ndarray(shape=(num_clusters, num_clusters))
        for label_i, label_j in product(*[self.clusters.keys(), self.clusters.keys()]):
            triples_i = self.clusters[label_i]
            triples_j = self.clusters[label_j]
            self.cluster_affinity[label_i, label_j] = np.average([affinity_matrix[i, j] for (i, _), (j, _)
                                                                  in product(*[triples_i, triples_j])])

        self.cluster_representative = {}
        for label, triples in self.clusters.items():
            triples_indices = [t[0] for t in triples]
            cluster_triple_affinity = affinity_matrix[triples_indices]
            cluster_triple_affinity = cluster_triple_affinity[:, [triples_indices]]
            representative = np.argmax(cluster_triple_affinity.sum(axis=0))
            self.cluster_representative[label] = self.relation_embedding.kb_triples[triples_indices[representative]]

        self.clusters_by_doc = {}
        for block_id, doc_relations in self.relation_by_doc.iteritems():
            doc_clusters = set([self.labeling[rel] for rel in doc_relations if rel in self.labeling])
            self.clusters_by_doc[block_id] = doc_clusters

    def compute_word_mover(self, query):
        block_id, query_relations, query_affinity, block_relations = query

        kernel_densities = np.array([[self.relation_embedding.kernel_density_pair
                                      ((query_relation, block_relation))[2]
                                      for block_relation in block_relations] for query_relation in query_relations])

        label_assignment = np.argmax(kernel_densities, axis=1)
        affinity = np.sum([query_affinity[i][self.labeling[block_relations[l]]][2]
                           for i,l in enumerate(label_assignment) if self.labeling.has_key(block_relations[l])])

        label_assignment = [(index, l) for index, l in enumerate(label_assignment)]
        densities = [kernel_densities[(i, e)] for i, e in label_assignment]

        return block_id, 0.6 * np.sum(densities) + 0.4 * affinity

    def compute_nearest_docs(self, query, topn=10):

        query_affinity = []
        for query_triple in query:
            candidates = [(query_triple, (n.left_entity, n.relation, n.right_entity))
                          for n in self.cluster_representative.values()]
            affinity = get_pool().map(self.relation_embedding.kernel_density_pair, candidates)
            affinity = {c : e for c, e in enumerate(affinity)}
            query_affinity.append(affinity)

        candidates = [(block_id, query, query_affinity, block_relations) for block_id, block_relations
                      in self.relation_by_doc.iteritems()]

        density_by_doc = get_pool().map(self.compute_word_mover, candidates)
        density_by_doc = sorted(density_by_doc, key=lambda e: -e[1])[:topn]
        nearest_docs = [(block_id, score, self.doc_text[block_id]) for block_id, score in density_by_doc]
        return nearest_docs













