from __future__ import print_function
import numpy as np
import tensorflow as tf
import random as rn

### We modified Pahikkala et al. (2014) source code for cross-val process ###

import os
os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(1)
rn.seed(1)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from tensorflow import keras
from tensorflow.keras import backend as K
tf.set_random_seed(0)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


from itertools import product

import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, GRU
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import optimizers, layers


import sys, pickle, os
import math, json, time
import decimal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from random import shuffle
from copy import deepcopy
from sklearn import preprocessing


TABSY = "\t"
figdir = "figures/"
os.system("mkdir -p %s" % figdir)

def cindex_score(y_true, y_pred):
    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f) #select


def build_combined_onehot(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    XDinput = Input(shape=(FLAGS.max_smi_len, FLAGS.charsmiset_size))
    XTinput = Input(shape=(FLAGS.max_seq_len, FLAGS.charseqset_size))


    encode_smiles= Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles) #pool_size=pool_length[i]


    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)



    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein])
    #encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1) #merge.Add()([encode_smiles, encode_protein])

    # Fully connected 
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)


    predictions = Dense(1, kernel_initializer='normal')(FC2) 

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])
    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score]) #, metrics=['cindex_score']
    

    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_combined_onehot.png')
    return interactionModel





def build_combined_categorical(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
   
    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32') ### Buralar flagdan gelmeliii
    XTinput = Input(shape=(FLAGS.max_seq_len,), dtype='int32')

    ### SMI_EMB_DINMS  FLAGS GELMELII 
    encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size+1, output_dim=128, input_length=FLAGS.max_smi_len)(XDinput) 
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)


    encode_protein = Embedding(input_dim=FLAGS.charseqset_size+1, output_dim=128, input_length=FLAGS.max_seq_len)(XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)


    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1) #merge.Add()([encode_smiles, encode_protein])

    # Fully connected 
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)


    # And add a logistic regression on top
    predictions = Dense(1, kernel_initializer='normal')(FC2) #OR no activation, rght now it's between 0-1, do I want this??? activation='sigmoid'

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])

    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score]) #, metrics=['cindex_score']
    print(interactionModel.summary())

    plot_model(interactionModel, to_file='figures/build_combined_categorical.png')

    return interactionModel



def build_single_drug(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
   
    interactionModel = Sequential()
    XTmodel = Sequential()
    XTmodel.add(Activation('linear', input_shape=(FLAGS.target_count,)))


    encode_smiles = Sequential()
    encode_smiles.add(Embedding(input_dim=FLAGS.charsmiset_size+1, output_dim=128, input_length=FLAGS.max_smi_len)) 
    encode_smiles.add(Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)) #input_shape=(MAX_SMI_LEN, SMI_EMBEDDING_DIMS)
    encode_smiles.add(Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1))
    encode_smiles.add(Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1))
    encode_smiles.add(GlobalMaxPooling1D())


    interactionModel.add(Merge([encode_smiles, XTmodel], mode='concat', concat_axis=1))
    #interactionModel.add(layers.merge.Concatenate([XDmodel, XTmodel]))

    # Fully connected 
    interactionModel.add(Dense(1024, activation='relu')) #1024
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(1024, activation='relu')) #1024
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(512, activation='relu')) 


    interactionModel.add(Dense(1, kernel_initializer='normal'))
    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score])

    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_single_drug.png')

    return interactionModel


def build_single_prot(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
   
    interactionModel = Sequential()
    XDmodel = Sequential()
    XDmodel.add(Activation('linear', input_shape=(FLAGS.drugcount,)))


    XTmodel1 = Sequential()
    XTmodel1.add(Embedding(input_dim=FLAGS.charseqset_size+1, output_dim=128,  input_length=FLAGS.max_seq_len))
    XTmodel1.add(Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)) #input_shape=(MAX_SEQ_LEN, SEQ_EMBEDDING_DIMS)
    XTmodel1.add(Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1))
    XTmodel1.add(Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1))
    XTmodel1.add(GlobalMaxPooling1D())


    interactionModel.add(Merge([XDmodel, XTmodel1], mode='concat', concat_axis=1))

    # Fully connected 
    interactionModel.add(Dense(1024, activation='relu'))
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(1024, activation='relu'))
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(512, activation='relu'))

    interactionModel.add(Dense(1, kernel_initializer='normal'))
    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score])

    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_single_protein.png')

    return interactionModel

def build_baseline(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    interactionModel = Sequential()

    XDmodel = Sequential()
    XDmodel.add(Dense(1, activation='linear', input_shape=(FLAGS.drug_count, )))

    XTmodel = Sequential()
    XTmodel.add(Dense(1, activation='linear', input_shape=(FLAGS.target_count,)))


    interactionModel.add(Merge([XDmodel, XTmodel], mode='concat', concat_axis=1))

    # Fully connected 
    interactionModel.add(Dense(1024, activation='relu'))
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(1024, activation='relu'))
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(512, activation='relu'))

    interactionModel.add(Dense(1, kernel_initializer='normal'))
    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score])

    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_baseline.png')

    return interactionModel


import argparse
def build_adapter(data, args):
    func_name = args["func_name"]
    func_args = args["func_args"]
    print("build_adapter", func_name, func_args)
    func = globals()[func_name]

    FLAGS = argparse.ArgumentParser()
    FLAGS.drug_count = data.drug_count
    FLAGS.target_count =  data.target_count
    FLAGS.max_smi_len = data.drug_max_seqlen
    FLAGS.max_seq_len = data.target_max_seqlen
    FLAGS.charsmiset_size = data.drug_charset_size
    FLAGS.charseqset_size = data.target_charset_size
    NUM_FILTERS = func_args["num_window"]
    FILTER_LENGTH1 = func_args["smi_window_length"]
    FILTER_LENGTH2 = func_args["seq_window_length"]

    return func(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2)
