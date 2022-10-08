from copy import deepcopy

INPUT_LIST = [
    {   
        "name": "deepdta",
        "loader" : ["deepdta", "DataMgr"],
        "args" : {
            "dataset_path": "../app_deepdta/data/davis/",
            "problem_type": 1,
            "drug_max_seqlen": 1000,
            "target_max_seqlen": 100,
            "is_log": 1,
        }
    },
    {   
        "name": "deepconvdti",
        "loader" : ["deepconvdti", "DataMgr"],
        "args" : {
            "dti_dir": "../app_deepconvdti/data/training_dataset/training_dti.csv",
            "drug_dir": "../app_deepconvdti/data/training_dataset/training_compound.csv",
            "protein_dir": "../app_deepconvdti/data/training_dataset/training_protein.csv",
            "with_label": True,

            #"test_names": "validation_dataset",
            "test_dti_dir": "../app_deepconvdti/data/validation_dataset/validation_dti.csv",
            "test_drug_dir": "../app_deepconvdti/data/validation_dataset/validation_compound.csv",
            "test_protein_dir": "../app_deepconvdti/data/validation_dataset/validation_protein.csv",

            "drug_vec": "morgan_fp_r2",
            "drug_len": 2048,
            "prot_vec": "Convolution",
            "prot_len": 2500,
        }
    }
]

### MODEL LIST CONFIG(with fit args)
MODEL_LIST = []

# add deepdta models
from tensorflow.keras.callbacks import EarlyStopping
deepdta_num_windows = [1]
deepdta_seq_window_lengths =[8, 12]
deepdta_smi_window_lengths =[4, 8]
deepdta_base_model_obj = {
    "loader": ["deepdta", "build_adapter"],
    "args": {
        "func_name": "build_combined_categorical",
    },
    "fit_args": {
        "epochs": 2,
        "batch_size": 512,
        "shuffle": False,
        "callbacks": [
            EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        ]
    }
}
for num_window in deepdta_num_windows:
    for seq_window_length in deepdta_seq_window_lengths:
        for smi_window_length in deepdta_smi_window_lengths:
            model_obj = deepcopy(deepdta_base_model_obj)
            model_obj["args"]["func_args"] = {
                "num_window": num_window,
                "smi_window_length": smi_window_length,
                "seq_window_length": seq_window_length,
            }
            MODEL_LIST.append(model_obj)

# add convdeepdti model
deepconvdti_base_model_obj = {
    "loader": ["deepconvdti", "build_adapter"],
    "args": {
        "window_sizes": [10,15,20,25,30],
        "protein_layers": [128],
        "drug_layers": [512,128],
        "fc_layers": [128],
        "learning_rate": 0.0001,
        "decay": 0.0001,
        "activation": "elu",
        "n_filters": 128,
        "dropout": 0,
    },
    "fit_args": {
        "epochs": 2,
        "batch_size": 8,
        "shuffle": False,
    }
}
MODEL_LIST.append(deepconvdti_base_model_obj)

EVALUATOR = {
}