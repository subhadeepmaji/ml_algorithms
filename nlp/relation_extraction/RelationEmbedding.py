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


class RelationEmbedding:

    def __init__(self, dimension, learning_rate=1e-6, tolerance=1e-10,
                 max_patience=5):
        """
        embedding of relations, where relations are expressed as (el, r, er)
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

            entities[l_entity] = RelationEmbedding.form_vec(l_entity, embedding,
                                                            self.dimension, sense='NOUN')
            entities[r_entity] = RelationEmbedding.form_vec(r_entity, embedding,
                                                            self.dimension, sense='NOUN')
            relations[relation] = RelationEmbedding.form_vec(relation, embedding,
                                                             self.dimension, sense='VERB')

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

        logger.info("number of nonconverged params ::")
        logger.info(above_tolerance)

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
        relaxed_E = relaxed_E / relaxed_E.sum(axis=0)

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

            logger.info("positive triple ::")
            logger.info(pos_triple_val)

            logger.info("negative triple ::")
            logger.info(neg_triple_val)

            objective_value = train_objective.eval({pos_triple_train: pos_triple_val,
                                                    neg_triple_train: neg_triple_val})

            if objective_value.any():
                self.param_value_old = [T.copy(p).get_value() for p in self.params]
                self.model(pos_triple_val, neg_triple_val)
                self.param_value_new = [p.get_value() for p in self.params]
                epochs += 1


