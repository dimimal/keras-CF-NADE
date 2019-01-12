#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import os
from data_gen import DataSet
from nade import NADE
from keras.models import load_model, model_from_json
from keras.layers import Input, Dropout, Lambda, add, Activation
from keras import backend as K
from keras.models import Model
from keras.callbacks import Callback
import keras.regularizers
from keras.optimizers import Adam
from rmse_module import RMSE_eval
import argparse
import pandas as pd
import keras.backend as K
from utils import *

def model(input_dim0, input_dim1):
    """Loads the CF-NADE model

        returns: The keras CF-NADE model
    """

    hidden_dim = 250

    input_layer = Input(
            shape=(input_dim0, input_dim1),
            name='input_ratings')

    output_ratings = Input(
            shape=(input_dim0, input_dim1),
            name='output_ratings')

    input_masks = Input(
            shape=(input_dim0,),
            name='input_masks')

    output_masks = Input(
            shape=(input_dim0,),
            name='output_masks')

    nade_layer = Dropout(0.0)(input_layer)
    nade_layer = NADE(
            hidden_dim=hidden_dim,
            activation='tanh',
            init='uniform',
            bias=True,
            W_regularizer=keras.regularizers.l2(0.02),
            V_regularizer=keras.regularizers.l2(0.02),
            b_regularizer=keras.regularizers.l2(0.02),
            c_regularizer=keras.regularizers.l2(0.02))(nade_layer)

    nade_layer = NADE(
            hidden_dim=hidden_dim,
            activation='tanh',
            init='uniform',
            bias=True,
            W_regularizer=keras.regularizers.l2(0.02),
            V_regularizer=keras.regularizers.l2(0.02),
            b_regularizer=keras.regularizers.l2(0.02),
            c_regularizer=keras.regularizers.l2(0.02))(nade_layer)

    predicted_ratings = Lambda(
            prediction_layer,
            output_shape=prediction_output_shape,
            name='predicted_ratings')(nade_layer)

    d = Lambda(
            d_layer,
            output_shape=d_output_shape,
            name='d')(input_masks)

    sum_masks = add([input_masks, output_masks])

    D = Lambda(
            D_layer,
            output_shape=D_output_shape,
            name='D')(sum_masks)

    loss_out = Lambda(
            rating_cost_lambda_func,
            output_shape=(1,),
            name='nade_loss')([nade_layer, output_ratings, input_masks, output_masks, D, d])

    cf_nade_model = Model(
            inputs=[input_layer, output_ratings, input_masks, output_masks],
            outputs=[loss_out, predicted_ratings])
    cf_nade_model.summary()

    adam = Adam(
            lr=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8)

    cf_nade_model.compile(
            loss={'nade_loss': lambda y_true, y_pred: y_pred},
            optimizer=adam)

    return cf_nade_model


def loadMovieLens():
    """TODO: Load and give predictions from movieLens
    dataset

    :arg1: TODO
    :returns: TODO

    """
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'test.csv'))
    print(data.head())
    print(data.columns)
    # defunct = pd.merge(data['ID'], data['user'], on='ID')
    # print(defunct.shape)
    return data


def parser():
    """TODO: Define the argparser

    :returns: Argparser object

    """

    argparser = argparse.ArgumentParser(
            description='Evaluate keras-CF-Nade model')

    argparser.add_argument(
        '-w',
        '--weights',
        help='path to pretrained weights')

    argparser.add_argument(
        '-i',
        '--input',
        help='path to the csv file with the movie ids')

    argparser.add_argument(
            '-n',
            action='store_true',
            help='specify this flag to load netflix test file')

    argparser.add_argument(
            '-m',
            action='store_true',
            help='specify this flag to load movielens test file')

    return argparser


def __main__(args):
    """Main function
    """

    if args.n is None and args.m is None:
        raise "You have to specify either of 2 flags -m or -n"

    # cf_nade_model = model()
    if args.n:
        # data = loadNetflix();
        pass
    elif args.m:
        assert '.weights' in args.weights, 'error loading weights file'
        weights = args.weights
        data = loadMovieLens()
    else:
        print('Unknown command ', sys.argv[1])

    print(len(data['user'].unique()))

    """OXxx
    """
    batch_size = 64
    num_users = len(data['user'].unique())
    num_items = len(data['ID'].unique())
    data_sample = 1.0
    input_dim0 = num_users
    input_dim1 = 5
    std = 0.0
    alpha = 1.0
    print('Loading data...')

    """
    train_file_list = sorted(glob.glob(os.path.join('data', 'train_set', 'part*')))
    val_file_list = sorted(glob.glob(os.path.join('data', 'val_set', 'part*')))
    test_file_list = sorted(glob.glob(os.path.join('data', 'test_set', 'part*')))
    train_file_list = [dfile for dfile in train_file_list if os.stat(dfile).st_size != 0]
    val_file_list = [dfile for dfile in val_file_list if os.stat(dfile).st_size != 0]
    test_file_list = [dfile for dfile in test_file_list if os.stat(dfile).st_size != 0]
    random.shuffle(train_file_list)
    random.shuffle(val_file_list)
    random.shuffle(test_file_list)
    train_file_list = train_file_list[:max(int(len(train_file_list) * data_sample),1)]

    train_set = DataSet(
            train_file_list,
            num_users=num_users,
            num_items=num_items,
            batch_size=batch_size,
            mode=0)

    val_set = DataSet(
            val_file_list,
            num_users=num_users,
            num_items=num_items,
            batch_size=batch_size,
            mode=1)

    test_set = DataSet(
            test_file_list,
            num_users=num_users,
            num_items=num_items,
            batch_size=batch_size,
            mode=2)
    """
    # cf_nade_model = keras.models.load_model('wholeModel.h5', custom_objects={NADE: 'NADE'})
    cf_nade_model = model(input_dim0, input_dim1)
    cf_nade_model.load_weights('model_weights.weights')
    y_pred = cf_nade_model.predict(data.user, batch_size=batch_size)
    print(y_pred.shape())

    print('Training...')

    cf_nade_model.fit_generator(
            train_set.generate(),
            steps_per_epoch=(train_set.get_corpus_size()//batch_size),
            epochs=30,
            validation_data=val_set.generate(),
            validation_steps=(val_set.get_corpus_size()//batch_size),
            shuffle=True,
            callbacks=[train_set, val_set, train_rmse_callback, val_rmse_callback],
            verbose=1)

    print('Testing...')
    rmses = []
    rate_score = np.array([1, 2, 3, 4, 5], np.float32)
    new_items = new_items

    squared_error = []
    n_samples = []
    for i, batch in enumerate(test_set.generate(max_iters=1)):
        inp_r = batch[0]['input_ratings']
        out_r = batch[0]['output_ratings']
        inp_m = batch[0]['input_masks']
        out_m = batch[0]['output_masks']

        pred_batch = cf_nade_model.predict(batch[0])[1]
        true_r = out_r.argmax(axis=2) + 1
        pred_r = (pred_batch * rate_score[np.newaxis, np.newaxis, :]).sum(axis=2)

        pred_r[:, new_items] = 3

        mask = out_r.sum(axis=2)

        '''
        if i == 0:
                print [true_r[0][j] for j in np.nonzero(true_r[0]* mask[0])[0]]
                print [pred_r[0][j] for j in np.nonzero(pred_r[0]* mask[0])[0]]
        '''

        se = np.sum(np.square(true_r - pred_r) * mask)
        n = np.sum(mask)
        squared_error.append(se)
        n_samples.append(n)

    total_squared_error = np.array(squared_error).sum()
    total_n_samples = np.array(n_samples).sum()
    rmse = np.sqrt(total_squared_error / (total_n_samples * 1.0 + 1e-8))


if __name__ == "__main__":
    arguments, unknown = parser().parse_known_args()
    __main__(arguments)
