import numpy as np

from util import accuracy
from hmm import HMM


# TODO:
def model_training(train_data, tags):
    """
    Train HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - tags: (1*num_tags) a list of POS tags

    Returns:
    - model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
    """
    model = None
    ###################################################
    num_state = len(tags)
    state_dict = {}
    obs_dict = {}
    num_obs_symbol = 0
    uni_words = set()
    pi = np.zeros(num_state)
    for each in range(len(tags)):
        state_dict[tags[each]] = each
    for each in train_data:
        for every in each.words:
            if every in uni_words:
                continue
            else:
                num_obs_symbol = num_obs_symbol + 1
                uni_words.add(every)
    values = np.array(list(uni_words))
    for each in range(num_obs_symbol):
        obs_dict[values[each]] = each
    A = np.zeros((num_state, num_state))
    B = np.zeros((num_state, num_obs_symbol))
    for each in train_data:
        for ind, val in enumerate(each.words):
            prev_state = state_dict[each.tags[ind - 1]]
            state = state_dict[each.tags[ind]]
            pi[state] += 1
            B[state][obs_dict[val]] += 1
            if ind > 0:
                A[prev_state][state] += 1
    pi = np.true_divide(pi, np.sum(pi))
    totalA = np.sum(A, axis=0)
    totalB = np.sum(B, axis=0)
    A = np.true_divide(A, totalA)
    B = np.true_divide(B, totalB)
    model = HMM(pi, A, B, obs_dict, state_dict)
    ###################################################
    return model


# TODO:
def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - model: an object of HMM class

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ###################################################
    col = np.ones((len(model.pi), 1)) * 0.000001
    ind = len(model.obs_dict)
    for each in test_data:
        keys = list(model.obs_dict.keys())
        new_words = list(set(each.words)-set(keys))
        for every in new_words:
            model.obs_dict[every] = ind
            model.B = np.append(model.B, col, axis=1)
            ind = ind + 1
        Osequence = np.array(each.words)
        tagging.append(model.viterbi(Osequence))
    ###################################################
    return tagging
