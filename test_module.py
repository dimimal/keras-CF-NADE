import glob
import os
import random
import numpy as np
from data_gen import DataSet
from nade import NADE

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Lambda, add
from keras import backend as K
from keras.models import Model
from keras.callbacks import Callback
import keras.regularizers
from keras.optimizers import Adam

import tensorflow as tf


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

    return K.sum(x,axis=1)


def d_output_shape(input_shape):
    return (input_shape[0],)



def D_layer(x):
    return K.sum(x,axis=1)

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


class RMSE_eval(Callback):
    def __init__(
            self,
            data_set,
            new_items,
            training_set):

        self.data_set = data_set
        self.rmses = []
        self.rate_score = np.array([1, 2, 3, 4, 5], np.float32)
        self.new_items = new_items
        self.training_set = training_set

        def eval_rmse(self):
            squared_error = []
            n_samples = []
            for i,batch in enumerate(self.data_set.generate(max_iters=1)):
                inp_r = batch[0]['input_ratings']
                out_r = batch[0]['output_ratings']
                inp_m = batch[0]['input_masks']
                out_m = batch[0]['output_masks']

                pred_batch = self.model.predict(batch[0])[1]
                true_r = out_r.argmax(axis=2) + 1
                pred_r = (pred_batch * self.rate_score[np.newaxis, np.newaxis, :]).sum(axis=2)

                pred_r[:, self.new_items] = 3

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

            return rmse

        def on_epoch_end(self, epoch, logs={}):
                score = self.eval_rmse()
                if self.training_set:
                        print "training set RMSE for epoch %d is %f"%(epoch, score)
                else:
                        print "validation set RMSE for epoch %d is %f"%(epoch, score)

                self.rmses.append(score)


if __name__ == '__main__':
    batch_size = 64
    num_users = 6040
    num_items = 3706
    data_sample = 1.0
    input_dim0 = 6040
    input_dim1 = 5
    hidden_dim = 250
    std = 0.0
    alpha = 1.0
    print('Loading data...')



    train_file_list = sorted(glob.glob(os.path.join(('data/train_set'), 'part*')))
    val_file_list = sorted(glob.glob(os.path.join(('data/val_set/'), 'part*')))
    test_file_list = sorted(glob.glob(os.path.join(('data/test_set/'), 'part*')))
    train_file_list = [dfile for dfile in train_file_list if os.stat(dfile).st_size != 0]
    val_file_list = [dfile for dfile in val_file_list if os.stat(dfile).st_size != 0]
    test_file_list = [dfile for dfile in test_file_list if os.stat(dfile).st_size != 0]
    random.shuffle(train_file_list)
    random.shuffle(val_file_list)
    random.shuffle(test_file_list)
    train_file_list = train_file_list[:max(int(len(train_file_list) * data_sample),1)]



    train_set = DataSet(train_file_list,
            num_users=num_users,
            num_items=num_items,
            batch_size=batch_size,
            mode=0)
    val_set = DataSet(val_file_list,
            num_users=num_users,
            num_items=num_items,
            batch_size=batch_size,
            mode=1)
    test_set = DataSet(test_file_list,
            num_users=num_users,
            num_items=num_items,
            batch_size=batch_size,
            mode=2)



    rating_freq = np.zeros((6040, 5))
    init_b = np.zeros((6040, 5))
    for batch in val_set.generate(max_iters=1):
            inp_r = batch[0]['input_ratings']
            out_r = batch[0]['output_ratings']
            inp_m = batch[0]['input_masks']
            out_m = batch[0]['output_masks']

            rating_freq += inp_r.sum(axis=0)



    log_rating_freq = np.log(rating_freq + 1e-8)
    log_rating_freq_diff = np.diff(log_rating_freq, axis=1)
    init_b[:, 1:] = log_rating_freq_diff
    init_b[:, 0] = log_rating_freq[:, 0]

    new_items = np.where(rating_freq.sum(axis=1) == 0)[0]

    input_layer = Input(shape=(input_dim0,input_dim1),
            name='input_ratings')
    output_ratings = Input(shape=(input_dim0,input_dim1),
            name='output_ratings')
    input_masks = Input(shape=(input_dim0,),
            name='input_masks')
    output_masks = Input(shape=(input_dim0,),
            name='output_masks')

    nade_layer = Dropout(0.0)(input_layer)
    nade_layer = NADE(hidden_dim=hidden_dim,
            activation='tanh',
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
    D = Lambda(D_layer,
            output_shape=D_output_shape,
            name='D')(sum_masks)


    loss_out = Lambda(rating_cost_lambda_func,
            output_shape=(1,),
            name='nade_loss')([nade_layer,output_ratings,input_masks,output_masks,D,d])


    cf_nade_model = Model(inputs=[input_layer,output_ratings,input_masks,output_masks],
            outputs=[loss_out,predicted_ratings])
    cf_nade_model.summary()


    adam = Adam(lr=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8)

    cf_nade_model.compile(loss={'nade_loss': lambda y_true, y_pred: y_pred},
            optimizer=adam)

    train_rmse_callback = RMSE_eval(data_set=train_set,
            new_items=new_items,
            training_set=True)
    val_rmse_callback = RMSE_eval(data_set=val_set,
            new_items=new_items,
            training_set=False)

    print 'Training...'
    cf_nade_model.fit_generator(
            train_set.generate(),
            steps_per_epoch=(train_set.get_corpus_size()//batch_size),
            epochs=30,
            validation_data=val_set.generate(),
            validation_steps=(val_set.get_corpus_size()//batch_size),
            shuffle=True,
            callbacks=[train_set,val_set,train_rmse_callback,val_rmse_callback],
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
    print("test set RMSE is %f" % (rmse))
