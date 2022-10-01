import os,json,argparse
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from copy import deepcopy

import config
import importlib

def _get_func(module_path, func_name):
    m = importlib.import_module(module_path)
    return getattr(m, func_name)

def new_inputer(input_cfg):
    loader_cfg = input_cfg["loader"]    
    func = _get_func("inputer.%s" % loader_cfg[0], loader_cfg[1])
    return func(**input_cfg["args"])


def new_model_obj(data, model_cfg):
    loader_cfg = model_cfg["loader"]    
    func = _get_func("model.%s" % loader_cfg[0], loader_cfg[1])
    return func(data, model_cfg["args"])


def run_model(model_obj, data, run_cfg):
    # train
    history = model_obj.fit(
        data.get_train_x(),
        data.get_train_y(),
        epochs = run_cfg["epochs"],
        validation_data = data.get_val_data(),                
    )
    
    # predict
    pre_y = model_obj.predict(data.get_test_x())
    
    # evaluate
    # use evaluator(testcase_id, history, data.get_test_y())
        

def main():
    for input_cfg in config.INPUT_LIST:
        inputer = new_inputer(input_cfg)
        print("inputer:", input_cfg["name"], input_cfg)
        print("total data num:%s" % inputer.get_testcase_num())
        for data_id in range(inputer.get_testcase_num()):
            print("data:%s" % data_id)
            data = inputer.get_testcase_data(data_id)
            print(
                data, 
                "cnt:", data.drug_count, data.target_count,
                "len:", data.drug_max_seqlen, data.target_max_seqlen,
                "csz:", data.drug_charset_size, data.target_charset_size
            )
            for model_id, model_cfg in enumerate(config.MODEL_LIST):
                print("model:%s" % model_id, model_cfg)
                model_obj = new_model_obj(data, model_cfg)
                run_model(model_obj, data, config.MODEL_RUN)


if __name__ == "__main__":
    main()
