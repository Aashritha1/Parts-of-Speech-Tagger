# coding=utf-8
from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        alpha[:, 0] = self.pi * self.B[:, self.obs_dict[Osequence[0]]]
        for each in range(1, L):
            alpha[:, each] = self.B[:, self.obs_dict[Osequence[each]]] * np.dot(self.A.T, alpha[:, each - 1])
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        beta[:, L - 1] = 1
        for each in reversed(range(0, L - 1)):
            beta[:, each] = np.dot(self.A, (self.B[:, self.obs_dict[Osequence[each + 1]]] * beta[:, each + 1]))
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        alphas = self.forward(Osequence)
        prob = np.sum(alphas[:, alphas.shape[1] - 1])
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        prob = (self.forward(Osequence) * self.backward(Osequence)) / self.sequence_prob(Osequence)
        ###################################################
        return prob

    # TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        for each in range(L - 1):
            prob[:, :, each] = (self.A * np.dot(self.forward(Osequence)[:, each].reshape(S, 1),
                                                (self.B[:, self.obs_dict[Osequence[each + 1]]] *
                                                 self.backward(Osequence)[:, each + 1]).reshape(S, 1).T)) / \
                               self.sequence_prob(Osequence)
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        S = len(self.pi)
        L = len(Osequence)
        deltas = np.zeros((S, L))
        indices = np.zeros((S, L))
        deltas[:, 0] = self.pi * self.B[:, self.obs_dict[Osequence[0]]]
        for each in range(1, L):
            for every in range(S):
                deltas[every, each] = self.B[every, self.obs_dict[Osequence[each]]] * np.max(self.A[:, every] * deltas[:, each - 1])
                indices[every, each] = np.argmax(self.A[:, every] * deltas[:, each - 1])
        ind = np.argmax(deltas[:, L - 1])
        path.append(str(list(self.state_dict.keys())[list(self.state_dict.values()).index(ind)]))
        for each in reversed(range(1, L)):
            ind = indices[int(ind), each]
            path.append(str(list(self.state_dict.keys())[list(self.state_dict.values()).index(ind)]))
        path.reverse()
        ###################################################
        return path
