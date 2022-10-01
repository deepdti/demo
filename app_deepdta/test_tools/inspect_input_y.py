import json
import pickle
import argparse
import numpy as np

def show_np_attrs(Y):
    print("row:%s, col:%s" % (len(Y), len(Y[0])))
    Y = np.asarray(Y)

    # https://www.cnblogs.com/massquantity/p/8908859.html where这里的用法代表不是Nan的坐标
    label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)
    print("label_row_inds:%s, len:%s" % (label_row_inds, len(label_row_inds)))
    print("label_col_inds:%s, len:%s" % (label_col_inds, len(label_col_inds)))

def conv_json(Y, output_fpath):

    json.dump(Y, open(output_fpath, "w"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_fpath")
    args = parser.parse_args()

    fpath = args.input_fpath + '/Y'
    Y = pickle.load(open(fpath,"rb"), encoding='latin1').tolist()
    print(type(Y))

    # conv_json(Y, "./Y.json")
    show_np_attrs(Y)

