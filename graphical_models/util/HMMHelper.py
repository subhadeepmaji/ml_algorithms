import graphical_models.hmm as hmm_model
import numpy as np


class HMMHelper:
    """
    class to help creating hmm instances
    """
    def __init__(self, num_states, num_emissions):
        self.__num_states = num_states
        self.__num_emissions = num_emissions
        self.hmm_model = hmm_model.HMM()

    def create_hmm(self, state_symbols, emission_symbols, priors=None, init_prob_matrix=True):
        """
        create an HMM model with the state symbols and emission symbols,
        :param state_symbols: list of state names
        :param emission_symbols: list of output values for the emissions
        :param priors: prior probabilities of the states
        :param init_prob_matrix: if True initialize the transition and emission
        probability matrices
        :return:
        """
        assert len(state_symbols) == self.__num_states, "number of states does not match with num_states"
        assert len(emission_symbols) == self.__num_emissions, "number of emissions does not match " \
                                                              "with num_emissions"
        if not priors:
            priors = [(1 / self.__num_states)] * self.__num_states
        for state_symbol,p in zip(state_symbols, priors):
            self.hmm_model.add_state(state_symbol, p)

        for emission_symbol in emission_symbols:
            self.hmm_model.add_emission(emission_symbol)
        if not init_prob_matrix:
            return

        # random initialize a valid transition matrix
        transition_matrix = np.random.rand(self.__num_states, self.__num_states)
        transition_matrix = (transition_matrix.T / np.sum(transition_matrix, axis=1)).T

        # random initialize a valid emission matrix
        emission_matrix = np.random.rand(self.__num_states, self.__num_emissions)
        emission_matrix = (emission_matrix.T / np.sum(emission_matrix, axis=1)).T

        self.hmm_model.add_transition_prob(transition_matrix)
        self.hmm_model.add_emission_prob(emission_matrix)
