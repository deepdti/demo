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

seq_rdic = ['A','I','L','V','F','W','Y','N','C','Q','M','S','T','D','E','R','H','K','G','P','O','U','X','B','Z']
seq_dic = {w: i+1 for i,w in enumerate(seq_rdic)}


def encodeSeq(seq, seq_dic):
    if pd.isnull(seq):
        return [0]
    else:
        return [seq_dic[aa] for aa in seq]

def run(dti_dir, drug_dir, protein_dir, with_label=True,
        prot_len=2500, prot_vec="Convolution",
        drug_vec="Convolution", drug_len=2048):

    print("Parsing {0} , {1}, {2} with length {3}, type {4}".format(*[dti_dir ,drug_dir, protein_dir, prot_len, prot_vec]))

    protein_col = "Protein_ID"
    drug_col = "Compound_ID"
    col_names = [protein_col, drug_col]
    if with_label:
        label_col = "Label"
        col_names += [label_col]
    dti_df = pd.read_csv(dti_dir)
    drug_df = pd.read_csv(drug_dir, index_col="Compound_ID")
    protein_df = pd.read_csv(protein_dir, index_col="Protein_ID")

    if prot_vec == "Convolution":
        protein_df["encoded_sequence"] = protein_df.Sequence.map(lambda a: encodeSeq(a, seq_dic))
    dti_df = pd.merge(dti_df, protein_df, left_on=protein_col, right_index=True)
    dti_df = pd.merge(dti_df, drug_df, left_on=drug_col, right_index=True)
    drug_feature = np.stack(dti_df[drug_vec].map(lambda fp: fp.split("\t")))
    if prot_vec=="Convolution":
        protein_feature = sequence.pad_sequences(dti_df["encoded_sequence"].values, prot_len)
    else:
        protein_feature = np.stack(dti_df[prot_vec].map(lambda fp: fp.split("\t")))
    if with_label:
        label = dti_df[label_col].values
        print("\tPositive data : %d" %(sum(dti_df[label_col])))
        print("\tNegative data : %d" %(dti_df.shape[0] - sum(dti_df[label_col])))
        return {"protein_feature": protein_feature, "drug_feature": drug_feature, "label": label}
    else:
        return {"protein_feature": protein_feature, "drug_feature": drug_feature}


from inputer.base import BaseDataMgr, BaseDataObj
class DataObj(BaseDataObj):
    def __init__(self):
        super(DataObj, self).__init__()
        self.train_data = None
        self.test_data = None

    def get_train_x(self):
        return [self.train_data['drug_feature'], self.train_data['protein_feature']]

    def get_train_y(self):
        return self.train_data["label"]

    def get_val_data(self):
        return None

    def get_test_x(self):
        return [self.test_data['drug_feature'], self.test_data['protein_feature']]

    def get_test_y(self):
        return self.test_data["label"]


class DataMgr(BaseDataMgr):
    def __init__(self, 
        dti_dir, drug_dir, protein_dir, with_label,
        test_dti_dir, test_drug_dir, test_protein_dir,
        drug_vec="Convolution", drug_len=2048,
        prot_vec="Convolution", prot_len=2500):

        self.drug_len = drug_len
        self.prot_len = prot_len
        self.drug_vec = drug_vec
        self.prot_vec = prot_vec
        self._train_data = run(
            dti_dir, drug_dir, protein_dir, with_label,
            prot_len, prot_vec, drug_vec, drug_len
        )
        self._test_data = run(
            test_dti_dir, test_drug_dir, test_protein_dir, with_label,
            prot_len, prot_vec, drug_vec, drug_len
        )

    def get_testcase_num(self):
        return 1

    def get_testcase_data(self, testcase_idx):
        data = DataObj()
        data.drug_max_seqlen = self.drug_len
        data.target_max_seqlen = self.prot_len
        data.drug_charset_size = len(seq_rdic)
        data.target_charset_size = len(seq_rdic)

        data.drug_count = self._train_data['drug_feature'].shape[0]
        data.target_count = self._train_data['protein_feature'].shape[0]

        data.train_data = self._train_data
        data.test_data = self._test_data
        data.set_extra_attr('drug_vec', self.drug_vec)
        data.set_extra_attr('prot_vec', self.prot_vec)

        return data


