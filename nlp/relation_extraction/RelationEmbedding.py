from __future__ import division
import logging
import random

import numpy as np
import scipy as sp
import theano
import theano.tensor as T
import copy_reg, types
from itertools import izip, chain
from gensim.models.word2vec import Word2Vec
from theano.ifelse import ifelse
import multiprocessing as mlp
from collections import defaultdict
from itertools import product

from nlp.relation_extraction import RelationTuple
from nlp.sense2vec import SenseEmbedding as SE

logger = logging.getLogger(__name__)
has_pool = False
module_pool = None


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

def get_pool():
    global module_pool
    if has_pool:
        return module_pool
    query_pool = mlp.Pool(mlp.cpu_count())
    module_pool = query_pool
    return module_pool

class BengioEmbedding:

    def __init__(self, dimension, learning_rate=1e-2, tolerance=1e-5, max_patience=5):
        """
        embedding of relation tuples, where relation tuples are expressed
        as (el, r, er),

        :param dimension: dimension of the entity and relation embedding
        :param learning_rate: learning rate for SGD
        :param tolerance: tolerance parameter for convergence

        reference : http://ronan.collobert.com/pub/matos/2011_knowbases_aaai.pdf
        """
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_patience = max_patience
        self.knowledge_triples = None
        self.entity_indices, self.relation_indices = {}, {}
        self.params = None
        self.is_inited, self.has_model = False, False
        self.param_value_new, self.param_value_old = None, None
        self.patience = 0

    def initialize_model(self, knowledge_triples, embedding=None):
        """
        initialize the params of the model with the embedding vectors
        :param embedding: a word2vec model object, default None for random initialization of the
        entity and relation phrases
        :param knowledge_triples: list of RelationTuple
        :return:
        """
        if embedding:
            assert issubclass(embedding.__class__, SE.SenseEmbedding), \
                "embedding must be subclass of sense embedding"
            assert isinstance(embedding.model, Word2Vec), \
                "embedding model must be instance of Word2Vec"

            assert embedding.model.vector_size == self.dimension, \
                "dimension of embedding model must match of supplied word embedding"

        self.knowledge_triples = knowledge_triples
        entities, relations = {}, {}

        for knowledge_tuple in self.knowledge_triples:
            if not isinstance(knowledge_tuple, RelationTuple):
                raise RuntimeError("relation must be an instance of RelationTuple")

            l_entity, r_entity = knowledge_tuple.left_entity, knowledge_tuple.right_entity
            relation = knowledge_tuple.relation

            entities[l_entity] = BengioEmbedding.form_vec(l_entity, embedding, self.dimension, sense='NOUN')
            entities[r_entity] = BengioEmbedding.form_vec(r_entity, embedding, self.dimension, sense='NOUN')
            relations[relation] = BengioEmbedding.form_vec(relation, embedding, self.dimension, sense='VERB')

        entity_arr = np.array(entities.values(), dtype=theano.config.floatX).T
        relation_arr = np.array(relations.values(), dtype=theano.config.floatX)

        # initialize the parameters of the model with the values of the embedding
        self.Entity = theano.shared(name='Entity', borrow=True, value=entity_arr)
        self.Relation_L = theano.shared(name='Relation_L', borrow=True, value=relation_arr)
        self.Relation_R = theano.shared(name='Relation_R', borrow=True, value=relation_arr)

        self.params = [self.Entity, self.Relation_L, self.Relation_R]

        # form the entity and relation indices
        for index, entity in enumerate(entities.keys()):
            self.entity_indices[entity] = index

        for index, relation in enumerate(relations.keys()):
            self.relation_indices[relation] = index

        logger.info("number of unique entities : %d" %len(self.entity_indices))
        logger.info("number of unique relations : %d" %len(self.relation_indices))
        self.is_inited = True

    @staticmethod
    def form_vec(entity, embedding, dimension, sense='NOUN'):

        if not embedding and sense == 'NOUN':
            return np.random.normal(0, 1, dimension)

        if not embedding and sense == 'VERB':
            return np.random.normal(0, 1, (dimension,dimension))

        sense_vec = embedding.get_sense_vec(entity, dimension, sense)
        if sense == 'NOUN': return sense_vec
        else: return np.tile(sense_vec.reshape(dimension, 1), (1, dimension))

    def __objective_triple(self, triple):
        """
        form the objective function value of a triple
        :param triple: (entity_l, entity_r, relation)
        :return:
        """
        l_index, r_index, relation_index = triple
        return T.nlinalg.norm(T.mul(self.Relation_L[relation_index, :, :], self.Entity[:, l_index]) -
                              T.mul(self.Relation_R[relation_index, :, :], self.Entity[:, r_index]),
                              ord=1)

    def __objective(self, pos_triple, neg_triple):
        f = 1 - self.__objective_triple(neg_triple) + self.__objective_triple(pos_triple)
        return ifelse(T.gt(f, theano.shared(0.0)), f, theano.shared(0.0))

    def __gradients(self, pos_triple, neg_triple):
        objective = self.__objective(pos_triple, neg_triple)

        grad_E = T.grad(objective, wrt=self.Entity)
        grad_RL = T.grad(objective, wrt=self.Relation_L)
        grad_RR = T.grad(objective, wrt=self.Relation_R)

        return grad_E, grad_RL, grad_RR

    def __converged(self, epoch, max_epochs):
        if epoch <= 1: return False
        if epoch >= max_epochs:
            logger.warn("Reaching maximum iterations, model parameters may not have converged")
            return True

        diff_params = [v_new - v_old for (v_new, v_old) in izip(self.param_value_new, self.param_value_old)]
        above_tolerance = [np.count_nonzero(e > self.tolerance) for e in diff_params]

        if not max(above_tolerance):
            if self.patience < self.max_patience:
                logger.info("used up one unit of patience")
                self.patience += 1
                return False
            else:
                return True
        else:
            if self.patience > 0:
                logger.info("we found another minima, resetting patience to 0")
                self.patience = 0
            return False

    def __form_input_tensor(self, name):

        left_entity = T.scalar(name='le_' + name, dtype='int32')
        right_entity = T.scalar(name='re_' + name, dtype='int32')
        relation = T.scalar(name='rel_' + name, dtype='int32')

        return T.stack([left_entity, right_entity, relation])

    def form_model(self):
        if not self.is_inited: raise RuntimeError("model must be initialized first before creating")

        pos_triple = self.__form_input_tensor('pos_triple')
        neg_triple = self.__form_input_tensor('neg_triple')

        grad_E, grad_RL, grad_RR = self.__gradients(pos_triple, neg_triple)
        relaxed_E = self.Entity - self.learning_rate * grad_E

        # enforce the constraint for all i ; ||E_i|| = 1
        relaxed_E = relaxed_E / relaxed_E.norm(2, axis =0)

        updates = [(self.Entity, relaxed_E),
                    (self.Relation_L, self.Relation_L - self.learning_rate * grad_RL),
                    (self.Relation_R, self.Relation_R - self.learning_rate * grad_RR)]

        logger.info("Will form the Relation embedding model")
        self.model = theano.function(inputs=[pos_triple, neg_triple], updates=updates)
        logger.info("successfully formed the model ")
        self.has_model = True

    def train(self, max_epochs=10000):
        if not self.is_inited or not self.has_model:
            raise RuntimeError("model must be initialized and then created first before training")

        epochs = 1
        self.patience = 0
        pos_triple_train = self.__form_input_tensor('pos_triple_train')
        neg_triple_train = self.__form_input_tensor('neg_triple_train')

        train_objective = self.__objective(pos_triple=pos_triple_train, neg_triple=neg_triple_train)

        while not self.__converged(epochs, max_epochs):
            # choose a triple randomly with replacement from knowledge triples
            if epochs % 100 == 0:
                logger.info("=====starting training epoch : %d=====" %epochs)

            triple = random.choice(self.knowledge_triples)
            l_entity, r_entity = self.entity_indices[triple.left_entity], \
                                 self.entity_indices[triple.right_entity]

            relation = self.relation_indices[triple.relation]
            pos_triple_val = (l_entity, r_entity, relation)

            choice = random.randint(0, 1)
            entity = random.choice(self.entity_indices.values())
            l_entity = entity if choice == 0 else l_entity
            r_entity = entity if choice == 1 else r_entity
            neg_triple_val = (l_entity, r_entity, relation)

            objective_value = train_objective.eval({pos_triple_train: pos_triple_val,
                                                    neg_triple_train: neg_triple_val})

            if objective_value.any():
                self.param_value_old = [T.copy(p).get_value() for p in self.params]
                self.model(pos_triple_val, neg_triple_val)
                self.param_value_new = [p.get_value() for p in self.params]
                epochs += 1


class TransEEmbedding:
    """
    Relation embedding using the TransE algorithm

    :param dimension: dimension of the entity and relation embedding
    :param learning_rate: learning rate for batch GD
    :param tolerance: tolerance parameter for convergence
    :param margin : error margin to used for considering a tuple for training
    :param batch_size : batch size of mini batch gradient descent training
    :param max_patience : number of batches to hold patience for
    reference : https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf
    """
    def __init__(self, dimension, learning_rate=1e-6, tolerance=1e-10, margin=1,
                 batch_size=20, max_patience=10):
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.margin = margin
        self.batch_size = batch_size
        self.entity_indices, self.relation_indices = {}, {}
        self.entity_reverse_indices = {}
        self.relation_reverse_indices = {}
        self.params = None
        self.kb_triples = None
        self.max_patience = max_patience
        self.patience = 0
        self.__inited, self.__has_model = False, False
        self.param_value_new, self.param_value_old = False, False

    def initialize_model(self, kb_triples):
        lower_bound = -6 / np.sqrt(self.dimension)
        upper_bound = 6 / np.sqrt(self.dimension)
        self.kb_triples = kb_triples
        entities, relations = set(), set()

        for knowledge_tuple in self.kb_triples:
            if not isinstance(knowledge_tuple, RelationTuple):
                raise RuntimeError("relation must be an instance of RelationTuple")

            l_entity, r_entity, relation = knowledge_tuple.left_entity, knowledge_tuple.right_entity, \
                                           knowledge_tuple.relation
            entities.add(l_entity)
            entities.add(r_entity)
            relations.add(relation)

        entities  = list(entities)
        relations = list(relations)
        num_entities, num_relations = len(entities), len(relations)

        entity_matrix = np.random.uniform(low=lower_bound, high=upper_bound, size=(self.dimension, num_entities))
        relation_matrix = np.random.uniform(low=lower_bound, high=upper_bound, size=(self.dimension, num_relations))

        relation_matrix /= np.linalg.norm(relation_matrix, axis=0, ord=2)

        # initialize the theano variables for entity and relations
        self.Entity = theano.shared(name='Entity', borrow=True, value=entity_matrix)
        self.Relation = theano.shared(name='Relation', borrow=True, value=relation_matrix)
        self.params = [self.Entity, self.Relation]

        # form the entity and relation indices
        for index, entity in enumerate(entities):
            self.entity_indices[entity] = index
            self.entity_reverse_indices[index] = entity

        for index, relation in enumerate(relations):
            self.relation_indices[relation] = index
            self.relation_reverse_indices[index] = relation

        self.__inited = True

    def __objective_triple(self, triple):
        l_index,r_index,relation_index = triple[0], triple[1], triple[2]
        left_add_relation = self.Entity[:,l_index] + self.Relation[:, relation_index]
        right = self.Entity[:,r_index]
        difference = left_add_relation - right
        return difference.norm(2)

    def __mapper(self, train_example):
        pos_triple, neg_triple = train_example[0:3], train_example[3:]
        f = self.margin - self.__objective_triple(neg_triple) + self.__objective_triple(pos_triple)
        margin_relaxed = ifelse(T.gt(f, theano.shared(0.0)), f, theano.shared(0.0))
        return margin_relaxed

    def __objective(self, mini_batch):
        relaxed_margins, updates = theano.scan(lambda e: self.__mapper(e), sequences=mini_batch)
        return T.sum(relaxed_margins)

    def __gradients(self, mini_batch):
        objective = self.__objective(mini_batch)
        gradient_entity = T.grad(objective, wrt=self.Entity)
        gradient_relation = T.grad(objective, wrt=self.Relation)
        return gradient_entity, gradient_relation

    def __converged(self, epoch, max_epochs):
        if epoch <= 1: return False
        if epoch >= max_epochs:
            logger.warn("Reaching maximum iterations, model parameters may not have converged")
            return True

        diff_params = [v_new - v_old for (v_new, v_old) in izip(self.param_value_new, self.param_value_old)]
        above_tolerance = [np.count_nonzero(e > self.tolerance) for e in diff_params]

        if not max(above_tolerance):
            if self.patience < self.max_patience:
                logger.info("used up one unit of patience")
                self.patience += 1
                return False
            else:
                return True
        else:
            if self.patience > 0:
                logger.info("we found another minima, resetting patience to 0")
                self.patience = 0
            return False

    def form_model(self):
        if not self.__inited: raise RuntimeError("model must be initialized first before creating")

        mini_batch = T.matrix('mini_batch', dtype='int32')
        gradient_entity, gradient_relation = self.__gradients(mini_batch)

        # enforce the constraint for all i ; ||E_i|| = 1
        updated_entity = self.Entity - self.learning_rate * gradient_entity
        updated_entity = updated_entity / updated_entity.norm(2, axis=0)

        updated_relation = self.Relation - self.learning_rate * gradient_relation

        updates = [(self.Entity, updated_entity),(self.Relation, updated_relation)]

        logger.info("Will form the Relation embedding model")
        self.model = theano.function(inputs=[mini_batch], updates=updates)
        logger.info("successfully formed the model ")
        self.__has_model = True

    def train(self, max_epochs = 10000):
        if not self.__inited or not self.__has_model:
            raise RuntimeError("model must be initialized and then created first before training")

        epochs = 1
        self.patience = 0

        triple_indices = range(len(self.kb_triples))
        entity_indices = self.entity_indices.values()
        while not self.__converged(epochs, max_epochs):
            s_batch = np.random.choice(triple_indices, size=self.batch_size)
            s_batch = [self.kb_triples[i] for i in s_batch]
            training_batch = []

            if epochs % 100 == 0:
                logger.info("=====starting training epoch : %d=====" % epochs)

            # form a mini batch of input
            for triple in s_batch:
                left_entity, right_entity = self.entity_indices[triple.left_entity], \
                                            self.entity_indices[triple.right_entity]
                relation = self.relation_indices[triple.relation]
                pos_triple = [left_entity, right_entity, relation]
                choice = random.randint(0, 1)
                entity = random.choice(entity_indices)

                left_entity = entity if choice == 0 else left_entity
                right_entity = entity if choice == 1 else right_entity
                # form a polluted triple by randomly disturbing either the left entity
                # or the right entity of a golden triple
                neg_triple = [left_entity, right_entity, relation]

                training_triple = pos_triple
                training_triple.extend(neg_triple)
                training_batch.append(training_triple)

            self.param_value_old = [T.copy(p).get_value() for p in self.params]
            self.model(training_batch)
            self.param_value_new = [p.get_value() for p in self.params]
            epochs += 1

    def predict(self, left_entity=None, right_entity=None, relation=None, topn=10):
        """
        predict the completion for a partial triple, supported
        inputs are (le,re), (le,rel), (re,rel),(le, re, rel)
        if all are specified, the method returns the likelihood of existence of
        the the input relation triple

        :param left_entity: left entity of the relation triple
        :param right_entity: right entity of the relation triple
        :param relation: relation of relation triple
        :param topn: return topn number of best completions, when working for
        completion prediction otherwise ignored
        :return: list of entity, list of relation or likelihood value
        """
        assert left_entity or relation or right_entity, "all inputs cannot be left unspecified"
        is_le, is_re = bool(left_entity), bool(right_entity)
        is_relation = bool(relation)

        if int(is_le) + int(is_relation) + int(is_re) < 2:
            raise RuntimeError("Atleast two of the inputs must be specified")

        if left_entity and not self.entity_indices.has_key(left_entity):
            raise RuntimeError("entity is not known %s" %left_entity)

        if right_entity and not self.entity_indices.has_key(right_entity):
            raise RuntimeError("entity is not known %s" % right_entity)

        if relation and not self.relation_indices.has_key(relation):
            raise RuntimeError("realation not known %s" %relation)

        # compute the likelihood of the relation triple
        if left_entity and relation and right_entity:
            le_index = self.entity_indices[left_entity]
            re_index = self.entity_indices[right_entity]
            rel_index = self.relation_indices[relation]

            le_vec, re_vec = self.Entity.get_value()[:, le_index], self.Entity.get_value()[:, re_index]
            rel_vec = self.Relation.get_value()[:, rel_index]
            similarity = le_vec + rel_vec - re_vec
            if not similarity.any(): return 1
            return 1 / np.linalg.norm(similarity, ord=2)

        # complete either the left entity or the right entity of the relation
        if (left_entity and relation) or (right_entity and relation):
            entity = left_entity if left_entity else right_entity
            entity_index = self.entity_indices[entity]
            entity_vec = self.Entity.get_value()[:, entity_index]

            rel_index = self.relation_indices[relation]
            relation_vec = self.Relation.get_value()[:, rel_index]
            entity = entity_vec + relation_vec if left_entity else entity_vec - relation_vec
            entity_norm = np.linalg.norm(entity, ord=2)

            candidates = [(index, np.dot(entity, candidate) / (entity_norm * np.linalg.norm(candidate, 2)))
                          for index, candidate in enumerate(self.Entity.get_value().T)]
            candidates = sorted(candidates, key=lambda e: -e[1])
            return [(self.entity_reverse_indices[index], score) for index, score in candidates[:topn]]

        # complete a relation for a given left and right entity
        if left_entity and right_entity:
            le_index = self.entity_indices[left_entity]
            re_index = self.entity_indices[right_entity]
            le_vec, re_vec = self.Entity.get_value()[:,le_index], self.Entity.get_value()[:,re_index]

            relation = re_vec - le_vec
            relation_norm = np.linalg.norm(relation, ord=2)

            candidates = [(index, np.dot(relation, candidate) / (relation_norm * np.linalg.norm(candidate, 2)))
                          for index, candidate in enumerate(self.Relation.get_value().T)]
            candidates = sorted(candidates, key=lambda e: -e[1])
            return [(self.relation_reverse_indices[index], score) for index, score in candidates[:topn]]


class TransHEmbedding:
    """
    Embedding of relations where relations are expressed as
    (el, r, er)

    :param dimension: dimension of the entity and relation embedding
    :param learning_rate: learning rate for batch GD
    :param tolerance: tolerance parameter for convergence
    :param margin : error margin to used for considering a tuple for training
    :param batch_size : batch size of mini batch gradient descent training
    :param max_patience : number of mini batches to hold patience for
    reference : http://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531/8546
    """
    def __init__(self, dimension, learning_rate=1e-2, tolerance=1e-5, margin=1,
                 epsilon = 1e-5, regularize_factor=0.5, batch_size=10, max_patience=10):
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.margin = margin
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.regularize_factor = regularize_factor
        self.entity_indices, self.relation_indices = {}, {}
        self.entity_reverse_indices = {}
        self.relation_reverse_indices = {}
        self.params = None
        self.kb_triples = None
        self.max_patience = max_patience
        self.patience = 0
        self.lower_bound = -6 / np.sqrt(self.dimension)
        self.upper_bound = 6 / np.sqrt(self.dimension)
        self.__inited, self.__has_model = False, False

        self.left_entity_relation = defaultdict(list)
        self.right_entity_relation = defaultdict(list)
        self.left_entity_triples = defaultdict(list)
        self.right_entity_triples = defaultdict(list)
        self.relation_triples = defaultdict(list)

    def __getstate__(self):
        state = dict()
        state["entity_indices"] = self.__dict__["entity_indices"]
        state["relation_indices"] = self.__dict__["relation_indices"]
        state["entity_idf"] = self.__dict__["entity_idf"]
        state["relation_norm_factor"] = self.__dict__["relation_norm_factor"]
        return state

    def __setstate__(self, d):
        self.__dict__.update(d)

    def form_vec(self, entity, embedding, sense='NOUN'):
        if not embedding:
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dimension)
        return embedding.get_sense_vec(entity, self.dimension, sense)

    def initialize_model(self, kb_triples, embedding=None):

        if embedding:
            assert issubclass(embedding.__class__, SE.SenseEmbedding), \
                "embedding must be subclass of sense embedding"
            assert isinstance(embedding.model, Word2Vec), \
                "embedding model must be instance of Word2Vec"

            assert embedding.model.vector_size == self.dimension, \
                "dimension of embedding model must match of supplied word embedding"

        self.kb_triples = kb_triples
        entities, relations = set(), set()

        for knowledge_tuple in self.kb_triples:
            if not isinstance(knowledge_tuple, RelationTuple):
                raise RuntimeError("relation must be an instance of RelationTuple")

            l_entity, r_entity, relation = knowledge_tuple.left_entity, knowledge_tuple.right_entity, \
                                           knowledge_tuple.relation
            entities.add(l_entity)
            entities.add(r_entity)
            relations.add(relation)

            self.left_entity_relation[(l_entity, relation)].append(knowledge_tuple)
            self.right_entity_relation[(r_entity, relation)].append(knowledge_tuple)
            self.left_entity_triples[l_entity].append(knowledge_tuple)
            self.right_entity_triples[r_entity].append(knowledge_tuple)
            self.relation_triples[relation].append(knowledge_tuple)

        entities = list(entities)
        relations = list(relations)
        num_entities, num_relations = len(entities), len(relations)

        if embedding:
            entity_vectors, relation_vectors = [], []
            for entity in entities:
                entity_vectors.append(self.form_vec(entity, embedding, sense='NOUN'))
            for relation in relations:
                relation_vectors.append(self.form_vec(relation, embedding, sense='VERB'))

            entity_matrix = np.array(entity_vectors, dtype=np.float).T
            relation_normal = np.array(relation_vectors, dtype=np.float).T
        else:

            entity_matrix = np.random.uniform(low=self.lower_bound,high=self.upper_bound,
                                              size=(self.dimension, num_entities))
            relation_normal = np.random.uniform(low=self.lower_bound,high=self.upper_bound,
                                                size=(self.dimension, num_relations))

        # initialize relation matrices to ~ uniform(low_bound, upper_bound)
        relation_matrix = np.random.uniform(low=self.lower_bound, high=self.upper_bound,
                                            size=(self.dimension, num_relations))

        relation_normal /= np.linalg.norm(relation_normal, axis=0, ord=2)

        # initialize the theano variables for entity and relations
        self.Entity = theano.shared(name='Entity', borrow=True, value=entity_matrix)
        self.Relation = theano.shared(name='Relation', borrow=True, value=relation_matrix)
        self.RelationNormal = theano.shared(name='RelationNormal', borrow=True, value=relation_normal)

        self.params = [self.Entity, self.Relation, self.RelationNormal]

        # form the entity and relation indices
        for index, entity in enumerate(entities):
            self.entity_indices[entity] = index
            self.entity_reverse_indices[index] = entity

        for index, relation in enumerate(relations):
            self.relation_indices[relation] = index
            self.relation_reverse_indices[index] = relation

        self.__inited = True

    def __objective_triple(self, triple):
        l_index, r_index, relation_index = triple[0], triple[1], triple[2]

        h = self.Entity[:,l_index]
        t = self.Entity[:,r_index]
        d_r = self.Relation[:,relation_index]
        w_r = self.RelationNormal[:,relation_index]

        l = h - T.dot(w_r, h) * w_r
        e = t - T.dot(w_r, t) * w_r
        return T.square((l + d_r - e).norm(2))

    def __compute_objective(self, triple):
        l_index, r_index, relation_index = triple[0], triple[1], triple[2]

        h = self.Entity.get_value()[:, l_index]
        t = self.Entity.get_value()[:, r_index]
        d_r = self.Relation.get_value()[:, relation_index]
        w_r = self.RelationNormal.get_value()[:, relation_index]

        l = h - np.dot(w_r, h) * w_r
        e = t - np.dot(w_r, t) * w_r
        return np.linalg.norm(l + d_r - e, ord=2) ** 2

    def __mapper(self, train_example):
        pos_triple, neg_triple = train_example[0:3], train_example[3:]

        unconstrained_objective = self.margin - self.__objective_triple(neg_triple) \
                                  + self.__objective_triple(pos_triple)

        entity_normalize = T.sum(T.square(self.Entity.norm(2, axis=0)) - 1)
        relation_normalize = T.square(self.Relation.norm(2, axis=0))
        surface_normalize = T.square(T.diagonal(T.dot(self.RelationNormal.T, self.Relation))) / relation_normalize

        surface_normalize = T.sum(surface_normalize - self.epsilon ** 2)

        unconstrained_objective_positive = ifelse(T.gt(unconstrained_objective, theano.shared(0.0)),
                                                  unconstrained_objective, theano.shared(0.0))

        entity_normalize_positive = ifelse(T.gt(entity_normalize, theano.shared(0.0)),
                                           entity_normalize, theano.shared(0.0))

        surface_normalize_positive = ifelse(T.gt(surface_normalize, theano.shared(0.0)),
                                            surface_normalize, theano.shared(0.0))

        return unconstrained_objective_positive + self.regularize_factor \
                                                  * (surface_normalize_positive + entity_normalize_positive)

    def __objective(self, mini_batch):
        relaxed_margins, updates = theano.scan(lambda e: self.__mapper(e), sequences=mini_batch)
        return T.sum(relaxed_margins)

    def __gradients(self, mini_batch):
        objective = self.__objective(mini_batch)
        gradient_entity = T.grad(objective, wrt=self.Entity)
        gradient_relation = T.grad(objective, wrt=self.Relation)
        gradient_surface = T.grad(objective, wrt=self.RelationNormal)

        return gradient_entity, gradient_relation, gradient_surface

    def __converged(self, epoch, max_epochs, param_value_old, param_value_new):
        if epoch <= 1: return False
        if epoch >= max_epochs:
            logger.warn("Reaching maximum iterations, model parameters may not have converged")
            return True

        diff_params = [v_new - v_old for (v_new, v_old) in izip(param_value_new, param_value_old)]
        above_tolerance = [np.count_nonzero(e > self.tolerance) for e in diff_params]

        if not max(above_tolerance):
            if self.patience < self.max_patience:
                logger.info("used up one unit of patience")
                self.patience += 1
                return False
            else:
                return True
        else:
            if self.patience > 0:
                logger.info("we found another minima, resetting patience to 0")
                self.patience = 0
            return False

    def form_model(self):
        if not self.__inited: raise RuntimeError("model must be initialized first before creating")

        mini_batch = T.matrix('mini_batch', dtype='int32')
        gradient_entity, gradient_relation, gradient_surface = self.__gradients(mini_batch)

        updated_entity = self.Entity - self.learning_rate * gradient_entity
        #updated_entity = updated_entity / updated_entity.norm(2, axis=0)
        updated_relation = self.Relation - self.learning_rate * gradient_relation

        updated_relation_normal = self.RelationNormal - self.learning_rate * gradient_surface
        updated_relation_normal = updated_relation_normal / updated_relation_normal.norm(2, axis=0)

        updates = [(self.Entity, updated_entity), (self.Relation, updated_relation),
                   (self.RelationNormal, updated_relation_normal)]

        logger.info("Will form the Relation embedding model")
        self.model = theano.function(inputs=[mini_batch], updates=updates)
        logger.info("successfully formed the model ")
        self.__has_model = True

    def train(self, max_epochs=10000):
        if not self.__inited or not self.__has_model:
            raise RuntimeError("model must be initialized and then created first before training")

        epochs = 1
        self.patience = 0

        triple_indices = range(len(self.kb_triples))
        entity_indices = self.entity_indices.values()
        param_value_old, param_value_new = None, None

        while not self.__converged(epochs, max_epochs, param_value_old, param_value_new):
            s_batch = np.random.choice(triple_indices, size=self.batch_size)
            s_batch = [self.kb_triples[i] for i in s_batch]
            training_batch = []

            if epochs % 100 == 0:
                logger.info("=====starting training epoch : %d=====" % epochs)

            # form a mini batch of input
            for triple in s_batch:
                left_entity, right_entity = self.entity_indices[triple.left_entity], \
                                            self.entity_indices[triple.right_entity]
                relation = self.relation_indices[triple.relation]
                pos_triple = [left_entity, right_entity, relation]
                choice = random.randint(0, 1)
                entity = random.choice(entity_indices)

                left_entity = entity if choice == 0 else left_entity
                right_entity = entity if choice == 1 else right_entity
                # form a polluted triple by randomly disturbing either the left entity
                # or the right entity of a golden triple
                neg_triple = [left_entity, right_entity, relation]

                training_triple = pos_triple
                training_triple.extend(neg_triple)
                training_batch.append(training_triple)

            param_value_old = [T.copy(p).get_value() for p in self.params]
            self.model(training_batch)
            param_value_new = [p.get_value() for p in self.params]
            epochs += 1
        self.normalize_relations()

    def normalize_relations(self):
        relation_counts = defaultdict(int)
        entity_counts = defaultdict(int)
        entity_epsilon = 1e-4

        for triple in self.kb_triples:
            relation_counts[triple.relation] += 1
            entity_counts[triple.left_entity] += 1
            entity_counts[triple.right_entity] += 1
            for entity in chain(triple.left_entity.split(" "), triple.right_entity.split(" ")):
                entity_counts[entity] += 1

        max_relation_count = max(relation_counts.values())
        min_relation_count = min(relation_counts.values())
        normalize_factor = max_relation_count - min_relation_count
        num_triples = len(self.kb_triples)

        self.relation_norm_factor = {rel : np.exp(-(count - min_relation_count) / (normalize_factor))
                                     for rel, count in relation_counts.iteritems()}

        self.entity_idf = {entity : np.log(num_triples / count) for entity, count
                           in entity_counts.iteritems()}

        max_idf = max(self.entity_idf.values())
        min_idf = min(self.entity_idf.values())
        normalize_factor = max_idf - min_idf
        self.entity_idf = {entity : (idf - min_idf + entity_epsilon) / normalize_factor for entity, idf
                           in self.entity_idf.iteritems()}

    def predict(self, left_entity=None, right_entity=None, relation=None, topn=10):
        """
        predict the completion for a partial triple, supported
        inputs are (le,re), (le,rel), (re,rel),(le, re, rel)
        if all are specified, the method returns the likelihood of existence of
        the the input relation triple

        :param left_entity: left entity of the relation triple
        :param right_entity: right entity of the relation triple
        :param relation: relation of relation triple
        :param topn: return topn number of best completions, when working for
        completion prediction otherwise ignored
        :return: list of entity, list of relation or likelihood value
        """
        assert left_entity or relation or right_entity, "all inputs cannot be left unspecified"
        is_le, is_re = bool(left_entity), bool(right_entity)
        is_relation = bool(relation)

        if int(is_le) + int(is_relation) + int(is_re) < 2:
            raise RuntimeError("Atleast two of the inputs must be specified")

        if left_entity and not self.entity_indices.has_key(left_entity):
            raise RuntimeError("entity is not known %s" % left_entity)

        if right_entity and not self.entity_indices.has_key(right_entity):
            raise RuntimeError("entity is not known %s" % right_entity)

        if relation and not self.relation_indices.has_key(relation):
            raise RuntimeError("relation not known %s" % relation)

        # complete the right entity of the relation
        if left_entity and relation:

            candidates = ((left_entity, relation, right_entity) for right_entity in self.entity_indices.keys())
            candidates = get_pool().map(self.kernel_density_estimate, candidates)
            #candidates = [((left_entity, relation, right_entity),
            #               self.kernel_density_estimate((left_entity, relation, right_entity)))
            #              for right_entity in self.entity_indices.keys()]

            candidates = sorted(candidates, key=lambda e: -e[1])
            return candidates[:topn]

        # complete the left entity of the relation
        if right_entity and relation:
            candidates = ((left_entity, relation, right_entity) for left_entity in self.entity_indices.keys())
            candidates = get_pool().map(self.kernel_density_estimate, candidates)

            #candidates = [(l_index, self.__compute_objective([l_index, r_index, rel_index]))
            #              for l_index, candidate in enumerate(self.Entity.get_value().T)]
            candidates = sorted(candidates, key=lambda e: -e[1])
            return candidates[:topn]

        # complete the relation for the triple
        if right_entity and left_entity:
            candidates = ((left_entity, relation, right_entity) for relation in self.relation_indices.keys())
            candidates = get_pool().map(self.kernel_density_estimate, candidates)

            candidates = sorted(candidates, key=lambda e: -e[1])
            return candidates[:topn]

    def kernel_density_estimate(self, x, std = 1):
        """
        compute the kernel density estimate for a triple
        :param xi: (l, r, h) of the relation triple
        :param std: standard deviation of the Gaussian Kernel
        :return: Density estimate for the relation triple
        """
        density = 0
        (l_e, rel, r_e) = x
        triples = self.left_entity_relation[(l_e, rel)]
        triples.extend(self.left_entity_relation[(r_e, rel)])

        if not triples:
            return x, 0.0

        for triple in triples:
            triple = triple.left_entity, triple.relation, triple.right_entity
            density += self.kernel_density_pair((x, triple), std=std)[2]
        return x, density / len(triples)

    def kernel_density_pair(self, x_pair, std = 1, relation_normal=None,
                            entity=None, mmaped=False):
        """
        compute the kernel density metric distance between the relation triples
        xi and xj
        :param x_pair: ((l1,r1,h1), (l2, r2, h2)) of the triple
        :param std : standard deviation of the Gaussian Kernel
        :param relation_normal:
        :param entity:
        :param mmaped:
        :return: kernel density distance between the two triples
        """
        xi, xj = x_pair
        (le_i, rel_i, re_i) = xi
        (le_j, rel_j, re_j) = xj

        try:
            (li_index, ri_index, hi_index) = self.entity_indices[le_i], self.relation_indices[rel_i], \
                                             self.entity_indices[re_i]
            (lj_index, rj_index, hj_index) = self.entity_indices[le_j], self.relation_indices[rel_j], \
                                         self.entity_indices[re_j]
        except KeyError, e:
            return xi, xj, 0

        if mmaped:
            li, lj = entity[:, li_index], entity[:, lj_index]
            hi, hj = entity[:, hi_index], entity[:, hj_index]
        else:
            li = self.Entity.get_value()[:, li_index]
            lj = self.Entity.get_value()[:, lj_index]
            hi = self.Entity.get_value()[:, hi_index]
            hj = self.Entity.get_value()[:, hj_index]

        if mmaped:
            ri_normal = relation_normal[:, ri_index]
            rj_normal = relation_normal[:, rj_index]
        else:
            ri_normal = self.RelationNormal.get_value()[:,ri_index]
            rj_normal = self.RelationNormal.get_value()[:, rj_index]

        li_plane = li - (np.dot(ri_normal, li) * ri_normal)
        lj_plane = lj - (np.dot(rj_normal, lj) * rj_normal)
        hi_plane = hi - (np.dot(ri_normal, hi) * ri_normal)
        hj_plane = hj - (np.dot(rj_normal, hj) * rj_normal)

        relation_norm_factor = self.relation_norm_factor[rel_i]
        left_entity_idf = self.entity_idf[le_i]
        right_entity_idf = self.entity_idf[re_i]

        entity_density_estimate = left_entity_idf * sp.spatial.distance.cosine(li_plane,lj_plane) \
                                  + right_entity_idf * sp.spatial.distance.cosine(hi_plane,hj_plane)
        relation_density_estimate = relation_norm_factor * sp.spatial.distance.cosine(ri_normal, rj_normal)
        weight_factor = 2 * (left_entity_idf + right_entity_idf) / relation_norm_factor
        return xi, xj, np.exp(-(weight_factor * entity_density_estimate + relation_density_estimate)
                              / 2 * (std ** 2)) / (2 * np.pi * std)

    def compute_affinity(self):
        triples = (((ti.left_entity, ti.relation, ti.right_entity), (tj.left_entity, tj.relation, tj.right_entity))
                   for ti,tj in product(*[self.kb_triples, self.kb_triples]))
        affinity = get_pool().map(self.kernel_density_pair, triples)
        affinity = np.array([a[2] for a in affinity])
        self.affinity_matrix = affinity.reshape(len(self.kb_triples), len(self.kb_triples))

    def nearest_neighbour(self, triple, topn=10):
        """
        compute the nearest topn neibhours of the relation triple
        :param triple: relation triple of the form (le, rel, re)
        :return: topn nearest neighbours of the relation triple
        """

        #nearby_leftentity_triples = chain.from_iterable([self.left_entity_triples[le] for le, _, _ in triples])
        #nearby_rightentity_triples = chain.from_iterable([self.right_entity_triples[re] for _, _, re in triples])
        #nearby_relation_triples = chain.from_iterable([self.relation_triples[rel] for _, rel, _ in triples])

        #kb_triples = chain(*[nearby_rightentity_triples, nearby_leftentity_triples, nearby_relation_triples])

        pairs = product(*[[triple], ((neighbour.left_entity, neighbour.relation, neighbour.right_entity)
                                    for neighbour in self.kb_triples)])

        distances = get_pool().map(self.kernel_density_pair, pairs)
        candidates = sorted(distances, key=lambda e : -e[2])[:2 * topn]
        candidates = {e[1] : e[2] for e in candidates}
        return sorted(candidates.iteritems(), key=lambda e:-e[1])[:topn]





