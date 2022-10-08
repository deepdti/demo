from __future__ import print_function
import os,json,argparse
import sys
sys.path.append(os.path.dirname(os.getcwd()))

print("syspath", sys.path)

import util.app as AppUtil
from util.log import logger


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


def plotLoss(history, batchind, epochind, param3ind, foldind):

    figname = "b"+str(batchind) + "_e" + str(epochind) + "_" + str(param3ind) + "_"  + str( foldind) + "_" + str(time.time()) 
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
	#plt.legend(['trainloss', 'valloss', 'cindex', 'valcindex'], loc='upper left')
    plt.legend(['trainloss', 'valloss'], loc='upper left')
    plt.savefig("figures/"+figname +".png" , dpi=None, facecolor='w', edgecolor='w', orientation='portrait', 
                    format=None,transparent=False, bbox_inches=None, pad_inches=0.1)
    plt.close()


    ## PLOT CINDEX
    plt.figure()
    plt.title('model concordance index')
    plt.ylabel('cindex')
    plt.xlabel('epoch')
    plt.plot(history.history['cindex_score'])
    plt.plot(history.history['val_cindex_score'])
    plt.legend(['traincindex', 'valcindex'], loc='upper left')
    plt.savefig("figures/"+figname + "_acc.png" , dpi=None, facecolor='w', edgecolor='w', orientation='portrait', 
                        format=None,transparent=False, bbox_inches=None, pad_inches=0.1)
    plt.close()


def prepare_interaction_pairs(XD, XT,  Y, rows, cols):
    drugs = []
    targets = []
    targetscls = []
    affinity=[] 
        
    for pair_ind in range(len(rows)):

            drug = XD[rows[pair_ind]]
            drugs.append(drug)

            target=XT[cols[pair_ind]]
            targets.append(target)

            affinity.append(Y[rows[pair_ind],cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)

    return drug_data,target_data,  affinity


def general_nfold_cv(XD, XT,  Y, label_row_inds, label_col_inds, prfmeasure, runmethod, FLAGS, labeled_sets, val_sets): ## BURAYA DA FLAGS LAZIM????
    
    paramset1 = FLAGS.num_windows                              #[32]#[32,  512] #[32, 128]  # filter numbers
    paramset2 = FLAGS.smi_window_lengths                               #[4, 8]#[4,  32] #[4,  8] #filter length smi
    paramset3 = FLAGS.seq_window_lengths                               #[8, 12]#[64,  256] #[64, 192]#[8, 192, 384]
    epoch = FLAGS.num_epoch                                 #100
    batchsz = FLAGS.batch_size                             #256

    logger.info("---Parameter Search-----")

    w = len(val_sets)
    h = len(paramset1) * len(paramset2) * len(paramset3)

    all_predictions = [[0 for x in range(w)] for y in range(h)] 
    all_losses = [[0 for x in range(w)] for y in range(h)] 
    print(all_predictions)

    # 这里还是五组
    for foldind in range(len(val_sets)):
        valinds = val_sets[foldind]
        labeledinds = labeled_sets[foldind]

        Y_train = np.mat(np.copy(Y))

        params = {}
        
        # 下面马上就会修改了，所以这里两句看起没啥意义
        XD_train = XD
        XT_train = XT

        # labeledinds(数据索引) -> label_row_inds/label_col_inds(D和T的索引) -> D和T的编码数据
        trrows = label_row_inds[labeledinds]
        trcols = label_col_inds[labeledinds]

        # 拿出D和T的编码，但其实没用，所以也没有意义。。。
        XD_train = XD[trrows]
        XT_train = XT[trcols]

        # 这里才是真正的准备数据
        train_drugs, train_prots,  train_Y = prepare_interaction_pairs(XD, XT, Y, trrows, trcols)
        
        terows = label_row_inds[valinds]
        tecols = label_col_inds[valinds]
        #print("terows", str(terows), str(len(terows)))
        #print("tecols", str(tecols), str(len(tecols)))

        val_drugs, val_prots,  val_Y = prepare_interaction_pairs(XD, XT,  Y, terows, tecols)


        pointer = 0
       
        for param1ind in range(len(paramset1)): #hidden neurons
            param1value = paramset1[param1ind]
            for param2ind in range(len(paramset2)): #learning rate
                param2value = paramset2[param2ind]

                for param3ind in range(len(paramset3)):
                    param3value = paramset3[param3ind]

                    gridmodel = runmethod(FLAGS, param1value, param2value, param3value)
                    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
                    gridres = gridmodel.fit(([np.array(train_drugs),np.array(train_prots) ]), np.array(train_Y), batch_size=batchsz, epochs=epoch, 
                            validation_data=( ([np.array(val_drugs), np.array(val_prots) ]), np.array(val_Y)),  shuffle=False, callbacks=[es] ) 


                    predicted_labels = gridmodel.predict([np.array(val_drugs), np.array(val_prots) ])
                    loss, rperf2 = gridmodel.evaluate(([np.array(val_drugs),np.array(val_prots) ]), np.array(val_Y), verbose=0)
                    rperf = prfmeasure(val_Y, predicted_labels)
                    rperf = rperf[0]


                    logger.info("P1 = %d,  P2 = %d, P3 = %d, Fold = %d, CI-i = %f, CI-ii = %f, MSE = %f" % 
                    (param1ind, param2ind, param3ind, foldind, rperf, rperf2, loss))

                    plotLoss(gridres, param1ind, param2ind, param3ind, foldind)

                    all_predictions[pointer][foldind] =rperf #TODO FOR EACH VAL SET allpredictions[pointer][foldind]
                    all_losses[pointer][foldind]= loss

                    pointer +=1

    bestperf = -float('Inf')
    bestpointer = None


    best_param_list = []
    ##Take average according to folds, then chooose best params
    pointer = 0
    for param1ind in range(len(paramset1)):
            for param2ind in range(len(paramset2)):
                for param3ind in range(len(paramset3)):
                
                    avgperf = 0.
                    for foldind in range(len(val_sets)):
                        foldperf = all_predictions[pointer][foldind]
                        avgperf += foldperf
                    avgperf /= len(val_sets)
                    #print(epoch, batchsz, avgperf)
                    if avgperf > bestperf:
                        bestperf = avgperf
                        bestpointer = pointer
                        best_param_list = [param1ind, param2ind, param3ind]

                    pointer +=1
        
    return  bestpointer, best_param_list, bestperf, all_predictions, all_losses


def nfold_1_2_3_setting_sample(data, runmethod, measure, FLAGS):
    XD = data["XD"]
    XT = data["XT"]
    Y = data["Y"]
    label_row_inds = data["label_row_inds"]
    label_col_inds = data["label_col_inds"]
    test_set = data["test_set"]
    outer_train_sets = data["outer_train_sets"]

    bestparamlist = []
    foldinds = len(outer_train_sets)

    test_sets = []
    ## TRAIN AND VAL
    val_sets = []
    train_sets = []

    logger.info('Start training')
    # 一共有五组train，4个train轮流做validation(交叉验证)，test一直不变
    # 最终生成五组数据
    for val_foldind in range(foldinds):
        val_fold = outer_train_sets[val_foldind]
        val_sets.append(val_fold)
        otherfolds = deepcopy(outer_train_sets)
        otherfolds.pop(val_foldind)
        otherfoldsinds = [item for sublist in otherfolds for item in sublist]
        train_sets.append(otherfoldsinds)
        test_sets.append(test_set)
        print("val set", str(len(val_fold)))
        print("train set", str(len(otherfoldsinds)))

    bestparamind, best_param_list, bestperf, all_predictions_not_need, losses_not_need = general_nfold_cv(
        XD, XT,  Y, label_row_inds, label_col_inds, 
        measure, runmethod, FLAGS, train_sets, val_sets
    )
   
    #print("Test Set len", str(len(test_set)))
    #print("Outer Train Set len", str(len(outer_train_sets)))
    bestparam, best_param_list, bestperf, all_predictions, all_losses = general_nfold_cv(
        XD, XT,  Y, label_row_inds, label_col_inds, 
        measure, runmethod, FLAGS, train_sets, test_sets
    )
    
    testperf = all_predictions[bestparamind]##pointer pos 

    logger.info("---FINAL RESULTS-----")
    logger.info("best param index = %s,  best param = %.5f" % 
            (bestparamind, bestparam))


    testperfs = []
    testloss= []

    avgperf = 0.

    for test_foldind in range(len(test_sets)):
        foldperf = all_predictions[bestparamind][test_foldind]
        foldloss = all_losses[bestparamind][test_foldind]
        testperfs.append(foldperf)
        testloss.append(foldloss)
        avgperf += foldperf

    avgperf = avgperf / len(test_sets)
    avgloss = np.mean(testloss)
    teststd = np.std(testperfs)

    logger.info("Test Performance CI")
    logger.info(testperfs)
    logger.info("Test Performance MSE")
    logger.info(testloss)
    return avgperf, avgloss, teststd



from arguments import argparser
if __name__ == "__main__":
    FLAGS = argparser()
    
    from convertor.deepdta_datahelper import run as parse_data
    data = parse_data(
        FLAGS.dataset_path, FLAGS.problem_type, 
        FLAGS.max_seq_len, FLAGS.max_smi_len,
        FLAGS.is_log
    )

    # TODO: 用opts替换FLAGS
    FLAGS.charseqset_size = data["charseqset_size"]
    FLAGS.charsmiset_size = data["charsmiset_size"]

    from model import deepdta
    model_builder = deepdta.build_combined_categorical

    from evaluator.deepdta_emetrics import get_aupr, get_cindex, get_rm2
    evaluator = get_cindex
    logger.info("FLAGS:%s", FLAGS)

    S1_avgperf, S1_avgloss, S1_teststd = nfold_1_2_3_setting_sample(
        data, model_builder, evaluator, FLAGS
    )
    logger.info(
        "avg_perf = %.5f,  avg_mse = %.5f, std = %.5f, opts:%s" %  
        (S1_avgperf, S1_avgloss, S1_teststd)
    )
    

