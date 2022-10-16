import os, sys
print("syspath", os.getcwd())
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd


# import keras modules
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Embedding, Lambda
from tensorflow.keras.layers import Convolution1D, GlobalMaxPooling1D, SpatialDropout1D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2,l1
from tensorflow.keras.preprocessing import sequence


from sklearn.metrics import precision_recall_curve, auc, roc_curve

class Drug_Target_Prediction(object):
    
    def PLayer(self, size, filters, activation, initializer, regularizer_param):
        def f(input):
            # model_p = Convolution1D(filters=filters, kernel_size=size, padding='valid', activity_regularizer=l2(regularizer_param), kernel_initializer=initializer, kernel_regularizer=l2(regularizer_param))(input)
            model_p = Convolution1D(filters=filters, kernel_size=size, padding='same', kernel_initializer=initializer, kernel_regularizer=l2(regularizer_param))(input)
            model_p = BatchNormalization()(model_p)
            model_p = Activation(activation)(model_p)
            return GlobalMaxPooling1D()(model_p)
        return f

    def modelv(self, dropout, drug_layers, protein_strides, filters, fc_layers, prot_vec=False, prot_len=2500,
               activation='relu', protein_layers=None, initializer="glorot_normal", drug_len=2048, drug_vec="ECFP4"):
        def return_tuple(value):
            if type(value) is int:
               return [value]
            else:
               return tuple(value)

        regularizer_param = 0.001
        input_d = Input(shape=(drug_len,))
        input_p = Input(shape=(prot_len,))
        params_dic = {"kernel_initializer": initializer,
                      # "activity_regularizer": l2(regularizer_param),
                      "kernel_regularizer": l2(regularizer_param),
        }
        input_layer_d = input_d
        if drug_layers is not None:
            drug_layers = return_tuple(drug_layers)
            for layer_size in drug_layers:
                model_d = Dense(layer_size, **params_dic)(input_layer_d)
                model_d = BatchNormalization()(model_d)
                model_d = Activation(activation)(model_d)
                model_d = Dropout(dropout)(model_d)
                input_layer_d = model_d

        if prot_vec == "Convolution":
            model_p = Embedding(26,20, embeddings_initializer=initializer,embeddings_regularizer=l2(regularizer_param))(input_p)
            model_p = SpatialDropout1D(0.2)(model_p)
            model_ps = [self.PLayer(stride_size, filters, activation, initializer, regularizer_param)(model_p) for stride_size in protein_strides]
            if len(model_ps)!=1:
                model_p = Concatenate(axis=1)(model_ps)
            else:
                model_p = model_ps[0]
        else:
            model_p = input_p

        if protein_layers:
            input_layer_p = model_p
            protein_layers = return_tuple(protein_layers)
            for protein_layer in protein_layers:
                model_p = Dense(protein_layer, **params_dic)(input_layer_p)
                model_p = BatchNormalization()(model_p)
                model_p = Activation(activation)(model_p)
                model_p = Dropout(dropout)(model_p)
                input_layer_p = model_p

        model_t = Concatenate(axis=1)([model_d,model_p])
        #input_dim = filters*len(protein_strides) + drug_layers[-1]
        if fc_layers is not None:
            fc_layers = return_tuple(fc_layers)
            for fc_layer in fc_layers:
                model_t = Dense(units=fc_layer,#, input_dim=input_dim,
                                **params_dic)(model_t)
                model_t = BatchNormalization()(model_t)
                model_t = Activation(activation)(model_t)
                # model_t = Dropout(dropout)(model_t)
                input_dim = fc_layer
        model_t = Dense(1, activation='tanh', activity_regularizer=l2(regularizer_param),**params_dic)(model_t)
        model_t = Lambda(lambda x: (x+1.)/2.)(model_t)

        model_f = Model(inputs=[input_d, input_p], outputs = model_t)

        return model_f

    def __init__(self, dropout=0.2, drug_layers=(1024,512), protein_windows = (10,15,20,25), filters=64,
                 learning_rate=1e-3, decay=0.0, fc_layers=None, prot_vec=None, prot_len=2500, activation="relu",
                 drug_len=2048, drug_vec="ECFP4", protein_layers=None):
        self.__dropout = dropout
        self.__drugs_layer = drug_layers
        self.__protein_strides = protein_windows
        self.__filters = filters
        self.__fc_layers = fc_layers
        self.__learning_rate = learning_rate
        self.__prot_vec = prot_vec
        self.__prot_len = prot_len
        self.__drug_vec = drug_vec
        self.__drug_len = drug_len
        self.__activation = activation
        self.__protein_layers = protein_layers
        self.__decay = decay
        self.__model_t = self.modelv(self.__dropout, self.__drugs_layer, self.__protein_strides,
                                     self.__filters, self.__fc_layers, prot_vec=self.__prot_vec,
                                     prot_len=self.__prot_len, activation=self.__activation,
                                     protein_layers=self.__protein_layers, drug_vec=self.__drug_vec,
                                     drug_len=self.__drug_len)

        opt = Adam(lr=learning_rate, decay=self.__decay)
        self.__model_t.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        with tf.compat.v1.keras.backend.get_session() as sess:
            run_args = tf.compat.v1.global_variables_initializer()
            #welcome = tf.constant('Welcome to tensorflow world!')
            sess.run(run_args)

    def fit(self, drug_feature, protein_feature, label, n_epoch=10, batch_size=32):
        for _ in range(n_epoch):
            history = self.__model_t.fit([drug_feature,protein_feature],label, epochs=_+1, batch_size=batch_size, shuffle=True, verbose=1,initial_epoch=_)
        return self.__model_t
    
    def summary(self):
        self.__model_t.summary()
    
    def validation(self, drug_feature, protein_feature, label, output_file=None, n_epoch=10, batch_size=32, **kwargs):
        if output_file:
            param_tuple = pd.MultiIndex.from_tuples([("parameter", param) for param in ["window_sizes", "drug_layers", "fc_layers", "learning_rate"]])
            result_df = pd.DataFrame(data = [[self.__protein_strides, self.__drugs_layer, self.__fc_layers, self.__learning_rate]]*n_epoch, columns=param_tuple)
            result_df["epoch"] = range(1,n_epoch+1)
        result_dic = {dataset: {"AUC":[], "AUPR": [], "opt_threshold(AUPR)":[], "opt_threshold(AUC)":[] }for dataset in kwargs}

        for _ in range(n_epoch):
            history = self.__model_t.fit([drug_feature,protein_feature],label,
                                         epochs=_+1, batch_size=batch_size, shuffle=True, verbose=1, initial_epoch=_)
            for dataset in kwargs:
                print("\tPredction of " + dataset)
                test_p = kwargs[dataset]["protein_feature"]
                test_d = kwargs[dataset]["drug_feature"]
                test_label = kwargs[dataset]["label"]
                prediction = self.__model_t.predict([test_d,test_p])
                fpr, tpr, thresholds_AUC = roc_curve(test_label, prediction)
                AUC = auc(fpr, tpr)
                precision, recall, thresholds = precision_recall_curve(test_label,prediction)
                distance = (1-fpr)**2+(1-tpr)**2
                EERs = (1-recall)/(1-precision)
                positive = sum(test_label)
                negative = test_label.shape[0]-positive
                ratio = negative/positive
                opt_t_AUC = thresholds_AUC[np.argmin(distance)]
                opt_t_AUPR = thresholds[np.argmin(np.abs(EERs-ratio))]
                AUPR = auc(recall,precision)
                print("\tArea Under ROC Curve(AUC): %0.3f" % AUC)
                print("\tArea Under PR Curve(AUPR): %0.3f" % AUPR)
                print("\tOptimal threshold(AUC)   : %0.3f " % opt_t_AUC)
                print("\tOptimal threshold(AUPR)  : %0.3f" % opt_t_AUPR)
                print("=================================================")
                result_dic[dataset]["AUC"].append(AUC)
                result_dic[dataset]["AUPR"].append(AUPR)
                result_dic[dataset]["opt_threshold(AUC)"].append(opt_t_AUC)
                result_dic[dataset]["opt_threshold(AUPR)"].append(opt_t_AUPR)
        if output_file:
            for dataset in kwargs:
                result_df[dataset, "AUC"] = result_dic[dataset]["AUC"]
                result_df[dataset, "AUPR"] = result_dic[dataset]["AUPR"]
                result_df[dataset, "opt_threshold(AUC)"] = result_dic[dataset]["opt_threshold(AUC)"]
                result_df[dataset, "opt_threshold(AUPR)"] = result_dic[dataset]["opt_threshold(AUPR)"]
            print("save to " + output_file)
            print(result_df)
            result_df.to_csv(output_file, index=False)

    def predict(self, **kwargs):
        results_dic = {}
        for dataset in kwargs:
            result_dic = {}
            test_p = kwargs[dataset]["protein_feature"]
            test_d = kwargs[dataset]["drug_feature"]
            result_dic["label"] = kwargs[dataset]["label"]
            result_dic["predicted"] = self.__model_t.predict([test_d, test_p])
            results_dic[dataset] = result_dic
        return results_dic
    
    def save(self, output_file):
        self.__model_t.save(output_file)

    def get_rawmodel(self):
        return self.__model_t


def build_model(**kwargs):
    return Drug_Target_Prediction(**kwargs)


class ModelAdapter(Drug_Target_Prediction):
    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose='auto',
            callbacks=None,
            validation_split=0.0, validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, 
            initial_epoch=0, steps_per_epoch=None,
            validation_steps=None, validation_batch_size=None, validation_freq=1,
            max_queue_size=10, workers=1, use_multiprocessing=False):
        for _ in range(epochs):
            history = self.get_rawmodel().fit(
                x, y, 
                epochs=_+1, batch_size=batch_size, shuffle=True, verbose=1,
                initial_epoch=_
            )
        return history

    # TODO:有需要其参数时再加
    def predict(self, x):
        return self.get_rawmodel().predict(x)

def build_adapter(data, args):
    model_params = {
        "protein_windows": args['window_sizes'],
        "protein_layers": args['protein_layers'],
        "drug_layers": args['drug_layers'],
        "fc_layers": args['fc_layers'],
        "learning_rate": args['learning_rate'],
        "decay": args['decay'],
        "activation": args['activation'],
        "filters": args['n_filters'],
        "dropout": args['dropout'],

        "prot_len": data.target_max_seqlen,
        "drug_len": data.drug_max_seqlen,
        "prot_vec": data.get_extra_attr('prot_vec'),
        "drug_vec": data.get_extra_attr('drug_vec'),
    }
    print("build model", model_params)
    return ModelAdapter(**model_params)