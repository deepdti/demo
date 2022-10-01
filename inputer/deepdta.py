import sys, re, math, time
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import collections
from collections import OrderedDict
from matplotlib.pyplot import cm
#from keras.preprocessing.sequence import pad_sequences
from copy import deepcopy


## ######################## ##
#
#  Define CHARSET, CHARLEN
#
## ######################## ## 

# CHARPROTSET = { 'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, \
#             'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, \
#             'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20, \
#             'O': 20, 'U': 20,
#             'B': (2, 11),
#             'Z': (3, 13),
#             'J': (7, 9) }
# CHARPROTLEN = 21

CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }

CHARPROTLEN = 25

CHARCANSMISET = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, 
			 ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12, 
			 "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18, 
			 "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
			 "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30, 
			 "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36, 
			 "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42, 
			 "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48, 
			 "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54, 
			 "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
			 "t": 61, "y": 62}

CHARCANSMILEN = 62

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64


## ######################## ##
#
#  Encoding Helpers
#
## ######################## ## 

#  Y = -(np.log10(Y/(math.pow(math.e,9))))

def one_hot_smiles(line, MAX_SMI_LEN, smi_ch_ind):
	X = np.zeros((MAX_SMI_LEN, len(smi_ch_ind))) #+1

	for i, ch in enumerate(line[:MAX_SMI_LEN]):
		X[i, (smi_ch_ind[ch]-1)] = 1 

	return X #.tolist()

def one_hot_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
	X = np.zeros((MAX_SEQ_LEN, len(smi_ch_ind))) 
	for i, ch in enumerate(line[:MAX_SEQ_LEN]):
		X[i, (smi_ch_ind[ch])-1] = 1

	return X #.tolist()


def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
	X = np.zeros(MAX_SMI_LEN)
	for i, ch in enumerate(line[:MAX_SMI_LEN]): #	x, smi_ch_ind, y
		X[i] = smi_ch_ind[ch]

	return X #.tolist()

def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
	X = np.zeros(MAX_SEQ_LEN)

	for i, ch in enumerate(line[:MAX_SEQ_LEN]):
		X[i] = smi_ch_ind[ch]

	return X #.tolist()

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

## ######################## ##
#
#  DATASET Class
#
## ######################## ## 
# works for large dataset
from inputer.base import BaseDataMgr, BaseDataObj
class DataObj(BaseDataObj):
    def __init__(self):
        super(DataObj, self).__init__()
        self.train_drugs = None
        self.train_prots = None
        self.train_Y = None
        self.val_drugs = None
        self.val_prots = None
        self.val_Y = None

    def get_train_x(self):
        return ([np.array(self.train_drugs),np.array(self.train_prots)])

    def get_train_y(self):
        return np.array(self.train_Y)

    def get_val_data(self):
        return (([np.array(self.val_drugs), np.array(self.val_prots) ]), np.array(self.val_Y))

    def get_test_x(self):
        return [np.array(self.val_drugs), np.array(self.val_prots)]

    def get_test_y(self):
        return np.array(self.val_Y)
    

class DataMgr(BaseDataMgr):
  def __init__(self, dataset_path, problem_type, target_max_seqlen, drug_max_seqlen,
               is_log=0,need_shuffle=False):
    self.dataset_path = dataset_path
    self.problem_type = problem_type
    self.SEQLEN = target_max_seqlen
    self.SMILEN = drug_max_seqlen
    #self.NCLASSES = n_classes
    self.charseqset = CHARPROTSET
    self.charseqset_size = CHARPROTLEN

    self.charsmiset = CHARISOSMISET ###HERE CAN BE EDITED
    self.charsmiset_size = CHARISOSMILEN
    self.PROBLEMSET = problem_type
    self._raw_data = None
    self.is_log = is_log

  def read_sets(self): ### fpath should be the dataset folder /kiba/ or /davis/
    fpath = self.dataset_path
    setting_no = self.problem_type
    print("Reading %s start" % fpath)

    test_fold = json.load(open(fpath + "folds/test_fold_setting" + str(setting_no)+".txt"))
    train_folds = json.load(open(fpath + "folds/train_fold_setting" + str(setting_no)+".txt"))
    
    return test_fold, train_folds

  def parse_data(self, with_label=True): 
    fpath = self.dataset_path	
    print("Read %s start" % fpath)

    # 这里是OrderedDict，所以是按照配置的顺序来的
    ligands = json.load(open(fpath+"ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath+"proteins.txt"), object_pairs_hook=OrderedDict)

    Y = pickle.load(open(fpath + "Y","rb"), encoding='latin1') ### TODO: read from raw
    if self.is_log:
        Y = -(np.log10(Y/(math.pow(10,9))))

    XD = []
    XT = []

    if with_label:
        for d in ligands.keys():
            XD.append(label_smiles(ligands[d], self.SMILEN, self.charsmiset))

        for t in proteins.keys():
            XT.append(label_sequence(proteins[t], self.SEQLEN, self.charseqset))
    else:
        for d in ligands.keys():
            XD.append(one_hot_smiles(ligands[d], self.SMILEN, self.charsmiset))

        for t in proteins.keys():
            XT.append(one_hot_sequence(proteins[t], self.SEQLEN, self.charseqset))
  
    return XD, XT, Y

  def get_testcase_num(self):
    if not self._raw_data:
      self.init_data()
    
    outer_train_sets = self._raw_data["outer_train_sets"]
    return len(outer_train_sets)

  def get_testcase_data(self, testcase_idx):
    if not self._raw_data:
      self.init_data()
    #testcase_num = self.get_testcase_num()
    #assert(testcase_idx < testcase_num, "error idx")
    
    raw_data = self._raw_data
    XD = raw_data["XD"]
    XT = raw_data["XT"]
    Y = raw_data["Y"]
    label_row_inds = raw_data["label_row_inds"]
    label_col_inds = raw_data["label_col_inds"]
    test_set = raw_data["test_set"]
    outer_train_sets = raw_data["outer_train_sets"]
    
    # 验证数据
    val_fold = outer_train_sets[testcase_idx]
    
    # 剔除验证数据，剩下的做训练数据
    otherfolds = deepcopy(outer_train_sets)
    otherfolds.pop(testcase_idx)
    train_sets = [item for sublist in otherfolds for item in sublist]

    # TODO:算出model相关函数的参数
    valinds = val_fold
    labeledinds = train_sets

    # labeledinds(数据索引) -> label_row_inds/label_col_inds(D和T的索引) -> D和T的编码数据
    trrows = label_row_inds[labeledinds]
    trcols = label_col_inds[labeledinds]
    train_drugs, train_prots, train_Y = prepare_interaction_pairs(XD, XT, Y, trrows, trcols)
        
    terows = label_row_inds[valinds]
    tecols = label_col_inds[valinds]
    val_drugs, val_prots, val_Y = prepare_interaction_pairs(XD, XT,  Y, terows, tecols)

    data = DataObj()
    data.drug_max_seqlen = self.SMILEN
    data.target_max_seqlen = self.SEQLEN
    data.drug_charset_size = self.charsmiset_size
    data.target_charset_size = self.charseqset_size
    data.drug_count = raw_data['drug_count']
    data.target_count = raw_data['target_count']

    data.train_drugs = train_drugs
    data.train_prots = train_prots
    data.train_Y = train_Y
    data.val_drugs = val_drugs
    data.val_prots = val_prots
    data.val_Y = val_Y

    return data

  def init_data(self):
    print("-- init data")
    XD, XT, Y = self.parse_data()
    XD = np.asarray(XD)
    XT = np.asarray(XT)
    Y = np.asarray(Y)

    drugcount = XD.shape[0]
    targetcount = XT.shape[0]
    #basically finds the point address of affinity [x,y]
    label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)
    test_set, outer_train_sets = self.read_sets() 

    self._raw_data = {
      "XD": XD,
      "XT": XT,
      "Y": Y,
      "drug_count": drugcount,
      "target_count": targetcount,
      "label_row_inds": label_row_inds,
      "label_col_inds": label_col_inds,
      "test_set": test_set,
      "outer_train_sets": outer_train_sets,
    }

