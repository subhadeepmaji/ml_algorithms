from __future__ import division
from collections import namedtuple

import numpy as np
import itertools

HMMEmission = namedtuple("HMMEmission", ['em_id', 'value'])


class HMMState:
    """
    An HMM state, holds the transition and emission probabilities
    """

    def __init__(self, node_id, name=None):
        self.state_id = node_id
        self.state_name = name
        self.transitions = dict()
        self.emissions = dict()

    def set_transition_prob(self, state_id, p_value):
        self.transitions[state_id] = p_value

    def set_emission_prob(self, emission_id, p_value):
        self.emissions[emission_id] = p_value

    def get_transition_prob(self, state_id):
        if self.transitions.has_key(state_id):
            return self.transitions[state_id]
        else:
            raise RuntimeError("transition probability not known for state_id :%s" % state_id)

    def get_emission_prob(self, emission_id):
        if self.emissions.has_key(emission_id):
            return self.emissions[emission_id]
        else:
            raise RuntimeError("emission probability not known for emission :%s" % emission_id)


class HMM:
    """
    An implementation of Hidden Markov Model with discrete state transitions
    and discrete output value(s)
    """
    def __init__(self):
        self.states, self.emissions = [], []
        self.num_states, self.num_emissions = 0, 0
        self.iterations = 0

    def add_state(self, state_name):
        state = HMMState(self.num_states, state_name)
        self.states.append(state)
        self.num_states += 1

    def add_emission(self, emission_value):
        emission = HMMEmission(em_id=self.num_emissions, value=emission_value)
        self.num_emissions += 1
        self.emissions.append(emission)

    def add_transition_prob(self, prob_matrix):
        """
        add a transition matrix for the state transitions
        :param prob_matrix: a numpy array of shape (states, states)
        :return:
        """
        assert prob_matrix.shape == (self.num_states, self.num_states), "shape mismatch of transition matrix"
        for state_id in xrange(prob_matrix.shape[0]):
            state = self.states[state_id]
            probabilities = prob_matrix[state_id]

            for _id, p in enumerate(probabilities):
                state.set_transition_prob(_id, p)

    def add_emission_prob(self, prob_matrix):
        """
        add a emission probability matrix
        :param prob_matrix: a numpy array of shape (states, emissions)
        :return:
        """
        assert prob_matrix.shape == (self.num_states, self.num_emissions), "shape mismatch of emissions matrix"
        for state_id in xrange(prob_matrix.shape[0]):
            state = self.states[state_id]
            emission_probabilities = prob_matrix[state_id]
            for emission_id, p in enumerate(emission_probabilities):
                state.set_emission_prob(emission_id, p)

    def compute_forward_prob(self, emitted_seq):
        """
        compute the forward transition probability matrix
        :param emitted_seq: emission id sequence of emissions as training
        :return: forward probability matrix of the sequence
        """
        sequence_len = len(emitted_seq)
        forward_prob = np.ndarray(shape=(self.num_states, sequence_len), dtype=np.float128)

        for i in range(self.num_states):
            forward_prob[i, 0] = (1 / self.num_states) * self.states[i].get_emission_prob(emitted_seq[0])

        for time_seq in xrange(1, sequence_len):
            for j in xrange(self.num_states):
                forward_prob[j, time_seq] = sum([forward_prob[i, time_seq - 1] * self.states[i]
                                                .get_transition_prob(j) * self.states[j]
                                                .get_emission_prob(emitted_seq[time_seq])
                                                 for i in xrange(self.num_states)])
            # Avoid underflow on probabilities
            for j in xrange(self.num_states):
                forward_prob[j, time_seq] = forward_prob[j, time_seq] / np.sum(forward_prob[:, time_seq])

        return forward_prob

    def compute_backward_prob(self, emitted_seq):
        """
        compute the backward transition probability matrix
        :param emitted_seq: emission id sequence of emissions as training
        :return: backward transitions probability matrix of the sequence
        """
        sequence_len = len(emitted_seq)
        bckward_prob = np.ndarray(shape=(self.num_states, sequence_len), dtype=np.float128)

        for i in range(self.num_states):
            bckward_prob[i, sequence_len - 1] = 1

        for time_seq in xrange(sequence_len - 2, 0, -1):
            for i in xrange(self.num_states):
                em_symbol = emitted_seq[time_seq + 1]
                bckward_prob[i, time_seq] = sum([bckward_prob[j, time_seq + 1] * self.states[i]
                                                .get_transition_prob(j) * self.states[j]
                                                .get_emission_prob(em_symbol) for j in xrange(self.num_states)])

            # Avoid underflow on probabilities
            for j in xrange(self.num_states):
                bckward_prob[j, time_seq] = bckward_prob[j, time_seq] / np.sum(bckward_prob[:, time_seq])

        return bckward_prob

    def __converged(self, max_iterations):
        """
        check for the convergance of the EM step
        :param max_iterations: maximum number of iterations as upper cap for covergance
        :return: true if converged false otherwise
        """
        # To Do : implement a sophisticated criteria to check convergance,
        # for now optimizing till we hit max iterations
        if self.iterations >= max_iterations:
            return True

        return False

    def train(self, emission_seq, max_iter=100):
        """
        Train the HMM on the Emission sequence
        :param emission_seq: a sequence of emission value
        :param max_iter: maximum number of iterations to wait for convergence
        :return:
        """
        # compute the forward probability of the sequence
        emissions = {}
        for emission in self.emissions:
            emissions[emission.value] = emission.em_id

        emitted_seq = [emissions[val] for val in emission_seq]
        sequence_len = len(emitted_seq)

        learning_matrix = np.ndarray(shape=(sequence_len, self.num_states, self.num_states), dtype=np.float128)
        transition_matrix = np.ndarray(shape=(self.num_states, self.num_states), dtype=np.float128)
        emission_matrix = np.ndarray(shape=(self.num_states, self.num_emissions), dtype=np.float128)

        self.iterations = 0
        while not self.__converged(max_iter):

            forward_prob = self.compute_forward_prob(emitted_seq)
            bckward_prob = self.compute_backward_prob(emitted_seq)

            # E-step of EM
            for time_seq in xrange(sequence_len):
                for (i, j) in itertools.product(xrange(self.num_states), xrange(self.num_states)):
                    bck_prob = 1 if time_seq == sequence_len - 1 else bckward_prob[j, time_seq + 1]
                    learning_matrix[time_seq, i, j] = forward_prob[i, time_seq] \
                                                      * self.states[i].get_transition_prob(j) \
                                                      * self.states[j].get_emission_prob(emitted_seq[time_seq]) \
                                                      * bck_prob

            # M-step of E
            for i in xrange(self.num_states):
                for j in xrange(self.num_states):
                    transition_matrix[i, j] = np.sum(learning_matrix[:, i, j]) / np.sum(learning_matrix[:, i, :])

            for j in xrange(self.num_states):
                for k in xrange(self.num_emissions):
                    matching_emission_seq = [_id for _id, emission_id in enumerate(emitted_seq) if emission_id == k]
                    print matching_emission_seq
                    emission_matrix[j, k] = np.sum(learning_matrix[matching_emission_seq, j, :]) / \
                                            np.sum(learning_matrix[:, j, :])

            print "emission_matrix", emission_matrix
            print "transition_matrix", transition_matrix

            self.add_emission_prob(emission_matrix)
            self.add_transition_prob(transition_matrix)
            self.iterations += 1

    def predict(self, emission_seq):
        """
        predict the most likely sequence of states for the emission sequence
        :param emission_seq: the emission sequence as test input
        :return: sequence of HMMState, forward probability matrix
        """
        emissions = {}
        for emission in self.emissions:
            emissions[emission.value] = emission.em_id

        emitted_seq = [emissions[val] for val in emission_seq]
        sequence_len = len(emitted_seq)

        forward_prob = np.ndarray(shape=(self.num_states, sequence_len), dtype=np.float128)
        optimal_states = [None] * sequence_len

        for i in range(self.num_states):
            forward_prob[i, 0] = (1 / self.num_states) * self.states[i].get_emission_prob(emitted_seq[0])

        optimal_states[0] = self.states[np.argmax(forward_prob[:, 0])]

        for time_seq in xrange(1, sequence_len):
            for j in xrange(self.num_states):
                optimal_states[time_seq], forward_prob[j, time_seq] = \
                    max([(self.states[i], np.exp(np.log(forward_prob[i, time_seq - 1]) +
                          np.log(self.states[i].get_transition_prob(j)) +
                          np.log(self.states[j].get_emission_prob(emitted_seq[time_seq]))))
                         for i in xrange(self.num_states)], key=lambda e: e[1])

        return optimal_states, forward_prob
