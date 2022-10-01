import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
import pandas as pd


# import keras modules
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Embedding, Lambda
from tensorflow.keras.layers import Convolution1D, GlobalMaxPooling1D, SpatialDropout1D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2,l1
from tensorflow.keras.preprocessing import sequence


from sklearn.metrics import precision_recall_curve, auc, roc_curve


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="""
    This Python script is used to train, validate, test deep learning model for prediction of drug-target interaction (DTI)\n
    Deep learning model will be built by Keras with tensorflow.\n
    You can set almost hyper-parameters as you want, See below parameter description\n
    DTI, drug and protein data must be written as csv file format. And feature should be tab-delimited format for script to parse data.\n
    Basically, this script builds convolutional neural network on sequence.\n
    If you don't want convolutional neural network but traditional dense layers on provide protein feature, specify type of feature and feature length.\n
    \n
    requirement\n
    ============================\n
    tensorflow > 1.0\n
    keras > 2.0\n
    numpy\n
    pandas\n
    scikit-learn\n
    ============================\n
    \n
    contact : dlsrnsladlek@gist.ac.kr\n
    """)
    # train_params
    parser.add_argument("dti_dir", help="Training DTI information [drug, target, label]")
    parser.add_argument("drug_dir", help="Training drug information [drug, SMILES,[feature_name, ..]]")
    parser.add_argument("protein_dir", help="Training protein information [protein, seq, [feature_name]]")
    # test_params
    parser.add_argument("--test-name", '-n', help="Name of test data sets", nargs="*")
    parser.add_argument("--test-dti-dir", "-i", help="Test dti [drug, target, [label]]", nargs="*")
    parser.add_argument("--test-drug-dir", "-d", help="Test drug information [drug, SMILES,[feature_name, ..]]", nargs="*")
    parser.add_argument("--test-protein-dir", '-t', help="Test Protein information [protein, seq, [feature_name]]", nargs="*")
    parser.add_argument("--with-label", "-W", help="Existence of label information in test DTI", action="store_true")
    # structure_params
    parser.add_argument("--window-sizes", '-w', help="Window sizes for model (only works for Convolution)", default=[10, 15, 20, 25, 30], nargs="*", type=int)
    parser.add_argument("--protein-layers","-p", help="Dense layers for protein", default=[128, 64], nargs="*", type=int)
    parser.add_argument("--drug-layers", '-c', help="Dense layers for drugs", default=[128], nargs="*", type=int)
    parser.add_argument("--fc-layers", '-f', help="Dense layers for concatenated layers of drug and target layer", default=[256], nargs="*", type=int)
    # training_params
    parser.add_argument("--learning-rate", '-r', help="Learning late for training", default=1e-4, type=float)
    parser.add_argument("--n-epoch", '-e', help="The number of epochs for training or validation", type=int, default=15)
    # type_params
    parser.add_argument("--prot-vec", "-v", help="Type of protein feature, if Convolution, it will execute conlvolution on sequeunce", type=str, default="Convolution")
    parser.add_argument("--prot-len", "-l", help="Protein vector length", default=2500, type=int)
    parser.add_argument("--drug-vec", "-V", help="Type of drug feature", type=str, default="morgan_fp_r2")
    parser.add_argument("--drug-len", "-L", help="Drug vector length", default=2048, type=int)
    # the other hyper-parameters
    parser.add_argument("--activation", "-a", help='Activation function of model', type=str, default='elu')
    parser.add_argument("--dropout", "-D", help="Dropout ratio", default=0.2, type=float)
    parser.add_argument("--n-filters", "-F", help="Number of filters for convolution layer, only works for Convolution", default=64, type=int)
    parser.add_argument("--batch-size", "-b", help="Batch size", default=32, type=int)
    parser.add_argument("--decay", "-y", help="Learning rate decay", default=1e-4, type=float)
    # mode_params
    parser.add_argument("--validation", help="Excute validation with independent data, will give AUC and AUPR (No prediction result)", action="store_true", default=False)
    parser.add_argument("--predict", help="Predict interactions of independent test set", action="store_true", default=False)
    # output_params
    parser.add_argument("--save-model", "-m", help="save model", type=str)
    parser.add_argument("--output", "-o", help="Prediction output", type=str)

    args = parser.parse_args()
    # train data
    train_dic = {
        "dti_dir": args.dti_dir,
        "drug_dir": args.drug_dir,
        "protein_dir": args.protein_dir,
        "with_label": True
    }
    # create dictionary of test_data
    test_names = args.test_name
    tests = args.test_dti_dir
    test_proteins = args.test_protein_dir
    test_drugs = args.test_drug_dir
    if test_names is None:
        test_sets = []
    else:
        test_sets = zip(test_names, tests, test_drugs, test_proteins)
    output_file = args.output
    # model_structure variables
    drug_layers = args.drug_layers
    window_sizes = args.window_sizes
    if window_sizes==0:
        window_sizes = None
    protein_layers = args.protein_layers
    fc_layers = args.fc_layers
    # training parameter
    train_params = {
        "n_epoch": args.n_epoch,
        "batch_size": args.batch_size,
    }
    # type parameter
    type_params = {
        "prot_vec": args.prot_vec,
        "prot_len": args.prot_len,
        "drug_vec": args.drug_vec,
        "drug_len": args.drug_len,
    }
    # model parameter
    model_params = {
        "drug_layers": drug_layers,
        "protein_windows": window_sizes,
        "protein_layers": protein_layers,
        "fc_layers": fc_layers,
        "learning_rate": args.learning_rate,
        "decay": args.decay,
        "activation": args.activation,
        "filters": args.n_filters,
        "dropout": args.dropout
    }

    model_params.update(type_params)
    print("\tmodel parameters summary\t")
    print("=====================================================")
    for key in model_params.keys():
        print("{:20s} : {:10s}".format(key, str(model_params[key])))
    print("=====================================================")

    from model import deepconvdti
    dti_prediction_model = deepconvdti.Drug_Target_Prediction(**model_params)
    dti_prediction_model.summary()

    # read and parse training and test data
    from convertor.deepconvdti import run as parse_data
    train_dic.update(type_params)
    train_dic = parse_data(**train_dic)
    test_dic = {test_name: parse_data(test_dti, test_drug, test_protein, with_label=True, **type_params)
                for test_name, test_dti, test_drug, test_protein in test_sets}

    # prediction mode
    if args.predict:
        print("prediction")
        train_dic.update(train_params)
        dti_prediction_model.fit(**train_dic)
        test_predicted = dti_prediction_model.predict(**test_dic)
        result_df = pd.DataFrame()
        result_columns = []
        for dataset in test_predicted:
            temp_df = pd.DataFrame()
            value = test_predicted[dataset]["predicted"]
            value = np.squeeze(value)
            print(dataset+str(value.shape))
            temp_df[dataset,'predicted'] = value
            temp_df[dataset, 'label'] = np.squeeze(test_predicted[dataset]['label'])
            result_df = pd.concat([result_df, temp_df], ignore_index=True, axis=1)
            result_columns.append((dataset, "predicted"))
            result_columns.append((dataset, "label"))
        result_df.columns = pd.MultiIndex.from_tuples(result_columns)
        print("save to %s"%output_file)
        result_df.to_csv(output_file, index=False)
    # validation mode
    if args.validation:
        validation_params = {}
        validation_params.update(train_params)
        validation_params["output_file"] = output_file
        print("\tvalidation summary\t")
        print("=====================================================")
        for key in validation_params.keys():
            print("{:20s} : {:10s}".format(key, str(validation_params[key])))
        print("=====================================================")
        validation_params.update(train_dic)
        validation_params.update(test_dic)
        dti_prediction_model.validation(**validation_params)

    # save trained model
    if args.save_model:
        dti_prediction_model.save(args.save_model)
    exit()
