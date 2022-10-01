import json
import pickle
import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_fpath")
    args = parser.parse_args()

    setting_no = 1
    input_fpath = args.input_fpath
    test_fold = json.load(open(input_fpath + "folds/test_fold_setting" + str(setting_no)+".txt"))
    train_folds = json.load(open(input_fpath + "folds/train_fold_setting" + str(setting_no)+".txt"))
    print("test:", len(test_fold))
    print("train:", len(train_folds), len(train_folds[0]))
