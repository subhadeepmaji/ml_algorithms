from __future__ import division
import logging
import random

import numpy as np
import theano
import theano.tensor as T
from itertools import izip
from gensim.models.word2vec import Word2Vec
from theano.ifelse import ifelse

from nlp.relation_extraction import RelationTuple
from nlp.sense2vec import SenseEmbedding as SE

logger = logging.getLogger(__name__)


class BengioEmbedding:

    def __init__(self, dimension, learning_rate=1e-6, tolerance=1e-10,
                 max_patience=5):
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

    def initialize_model(self, kb_triples, embedding=None):
        if embedding:
            assert issubclass(embedding.__class__, SE.SenseEmbedding), \
                "embedding must be subclass of sense embedding"
            assert isinstance(embedding.model, Word2Vec), \
                "embedding model must be instance of Word2Vec"

            assert embedding.model.vector_size == self.dimension, \
                "dimension of embedding model must match of supplied word embedding"

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




