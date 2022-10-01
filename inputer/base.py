class BaseDataMgr(object):
    def __init__(self):
        pass

    def get_testcase_num(self):
        raise Exception("this func need implementation")

    def get_testcase_data(self):
        raise Exception("this func need implementation")


class BaseDataObj(object):
    def __init__(self):
        self.drug_max_seqlen = 0 # 药物编码的长度上限
        self.target_max_seqlen = 0 # 靶标编码的长度上限
        self.drug_charset_size = 0 # 药物编码的种类
        self.target_charset_size = 0 # 靶标编码的种类
        self.drug_count = 0 # 药物个数
        self.target_count = 0 # 靶标个数

        # 额外非统一属性，用特殊模型需求
        # 比如deepconvdti的drug_vec和prot_vec
        self._extra_attrs = {}

    def get_extra_attr(self, key):
        return self._extra_attrs.get(key)

    def set_extra_attr(self, key, val):
        self._extra_attrs[key] = val

    def get_train_x(self):
        raise Exception("this func need implementation")

    def get_train_y(self):
        raise Exception("this func need implementation")

    def get_val_data(self):
        raise Exception("this func need implementation")

    def get_test_x(self):
        raise Exception("this func need implementation")

    def get_test_y(self):
        raise Exception("this func need implementation")

