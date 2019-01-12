#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras.backend as K

def prediction_layer(x):
    # x.shape = (?,6040,5)
    x_cumsum = K.cumsum(x, axis=2)
    # x_cumsum.shape = (?,6040,5)

    output = K.softmax(x_cumsum)
    # output = (?,6040,5)
    return output


def prediction_output_shape(input_shape):
    return input_shape


def d_layer(x):
    return K.sum(x, axis=1)


def d_output_shape(input_shape):
    return (input_shape[0],)


def D_layer(x):
    return K.sum(x, axis=1)


def D_output_shape(input_shape):
    return (input_shape[0],)


def rating_cost_lambda_func(args):
    alpha = 1.
    std = 0.01
    """
    """
    pred_score,true_ratings,input_masks,output_masks,D,d = args
    pred_score_cum = K.cumsum(pred_score, axis=2)

    prob_item_ratings = K.softmax(pred_score_cum)
    accu_prob_1N = K.cumsum(prob_item_ratings, axis=2)
    accu_prob_N1 = K.cumsum(prob_item_ratings[:, :, ::-1], axis=2)[:, :, ::-1]
    mask1N = K.cumsum(true_ratings[:, :, ::-1], axis=2)[:, :, ::-1]
    maskN1 = K.cumsum(true_ratings, axis=2)
    cost_ordinal_1N = -K.sum((K.log(prob_item_ratings) - K.log(accu_prob_1N)) * mask1N, axis=2)
    cost_ordinal_N1 = -K.sum((K.log(prob_item_ratings) - K.log(accu_prob_N1)) * maskN1, axis=2)
    cost_ordinal = cost_ordinal_1N + cost_ordinal_N1
    nll_item_ratings = K.sum(-(true_ratings * K.log(prob_item_ratings)),axis=2)
    nll = std * K.sum(nll_item_ratings,axis=1) * 1.0 * D / (D - d + 1e-6) + alpha * K.sum(cost_ordinal,axis=1) * 1.0 * D / (D - d + 1e-6)
    cost = K.mean(nll)
    cost = K.expand_dims(cost, 0)

    return cost

if __name__ == "__main__":
    pass
