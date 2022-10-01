from util.log import logger
import os
import importlib

def get_inputer(module_path, func_name):
    m = importlib.import_module("inputer.%s" % module_path)
    return m[func_name]

def get_model_builder(module_path, func_name):
    m = importlib.import_module("inputer.%s" % module_path)
    return m[func_name]


def conv_data(module_path, args={}):
    try:
        m = importlib.import_module("convertor." + module_path)
    except:
        logger.error("load convertor fail:%s", module_path)
        raise
    
    logger.info("run convertor.begin:%s", module_path)
    try:
        data = m.run(args)
    except:
        logger.error("run convertor.fail:%s", module_path)
        raise
    logger.info("run convertor.done:%s", module_path)
    return data