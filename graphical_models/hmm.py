from __future__ import division
from collections import namedtuple

import numpy as np
import itertools

HMMEmission = namedtuple("HMMEmission", ['em_id', 'value'])


class HMMState:
    """
    An HMM state, holds the transition and emission probabilities
    """

    def __init__(self, node_id, name=None, prior=None):
        self.state_id = node_id
        self.state_name = name
        self.prior = prior
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
    def __init__(self, smoothing=0.01):
        self.states, self.emissions = [], []
        self.num_states, self.num_emissions = 0, 0
        self.iterations = 0
        self.smoothing = smoothing

    def add_state(self, state_name, prior=None):
        state = HMMState(self.num_states, state_name, prior)
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

            if not state.prior:
                state.prior = (1 / self.num_states)

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
            forward_prob[i, 0] = self.states[i].prior * self.states[i].get_emission_prob(emitted_seq[0])

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

    def train_supervised(self, train_sequences):
        """
        Train the HMM in a supervised manner using smoothed MLE estimates
         from the training set
        :param train_sequences: a sequence of sequences where each sequence is of type
        [(o,z)] where o is the observed output and z is the observed state
        :return:
        """

        obs_emissions = {}
        obs_states = {}

        for emission in self.emissions:
            obs_emissions[emission.value] = emission.em_id
        for state in self.states:
            obs_states[state.state_name] = state.state_id

        # initialize the count DS to all zero's
        state_count = np.zeros(shape=(self.num_states,), dtype=np.uint32)
        state_bi_count = np.zeros(shape=(self.num_states,self.num_states), dtype=np.uint32)
        state_initial = np.zeros(shape=(self.num_states,), dtype=np.uint32)
        state_emission_count = np.zeros(shape=(self.num_states,self.num_emissions), dtype=np.uint32)

        for train_sequence in train_sequences:
            emission_seq, state_seq = zip(*train_sequence)

            emitted_seq = np.array([obs_emissions[em_symbol] for em_symbol in emission_seq])
            observed_state_seq = np.array([obs_states[obs_state] for obs_state in state_seq])

            state_initial[observed_state_seq[0]] += 1

            for state_id in xrange(self.num_states):
                state_count[state_id] += np.count_nonzero(observed_state_seq == state_id)

            for (state_id_prev, state_id_id_current) in itertools.izip(observed_state_seq
                    ,observed_state_seq[1:]):
                state_bi_count[state_id_prev, state_id_id_current] += 1

            for (state_id, emission_id) in itertools.izip(observed_state_seq
                    , emitted_seq):
                state_emission_count[state_id, emission_id] += 1

        transition_matrix = (state_bi_count + self.smoothing) / \
                            (state_count[:, None] + self.smoothing * self.num_states)
        emission_matrix = (state_emission_count + self.smoothing) / \
                          (state_count[:, None] + self.smoothing * self.num_emissions)

        for state_id in xrange(self.num_states):
            self.states[state_id].prior = (state_initial[state_id] + self.smoothing)/ \
                                          (len(train_sequences) + self.smoothing * self.num_states)

        self.add_transition_prob(transition_matrix)
        self.add_emission_prob(emission_matrix)

    def train_unsupervised(self, emission_seq, max_iter=100):
        """
        Train the HMM on the Emission sequence using EM (Baum-Welch)
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
                    transition_matrix[i, j] = (np.sum(learning_matrix[:, i, j]) +
                                               self.smoothing) / (np.sum(learning_matrix[:, i, :]) +
                                                                  self.smoothing * self.num_states)

            for j in xrange(self.num_states):
                for k in xrange(self.num_emissions):
                    matching_emission_seq = [_id for _id, emission_id in enumerate(emitted_seq) if emission_id == k]
                    emission_matrix[j, k] = (np.sum(learning_matrix[matching_emission_seq, :, j]) + self.smoothing) / \
                                            (np.sum(learning_matrix[:, :, j]) + self.smoothing * self.num_emissions)

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
            forward_prob[i, 0] = self.states[i].prior * self.states[i].get_emission_prob(emitted_seq[0])

        optimal_states[0] = self.states[np.argmax(forward_prob[:, 0])]

        for time_seq in xrange(1, sequence_len):
            for j in xrange(self.num_states):
                optimal_states[time_seq], forward_prob[j, time_seq] = \
                    max([(self.states[i], np.exp(np.log(forward_prob[i, time_seq - 1]) +
                          np.log(self.states[i].get_transition_prob(j)) +
                          np.log(self.states[j].get_emission_prob(emitted_seq[time_seq]))))
                         for i in xrange(self.num_states)], key=lambda e: e[1])

        return optimal_states, forward_prob
