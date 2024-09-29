import yaml
import os

Data = {}


Data["FEWSHOT"] = {}
Data["FEWSHOT"]["NWAY"] = 5
Data["FEWSHOT"]["KSHOT"] = 1
Data["FEWSHOT"]["TRAIN_QUERY_SHOT"] = 5
Data["FEWSHOT"]["TEST_QUERY_SHOT"] = 5
Data["FEWSHOT"]["TRAIL"] = 1000

Data["DATASET"] = "sketchy"
Data["DATA_ROOT"] = "/home/zhengwei/my_data/datasets"

Data["DATA"] = {}
Data["DATA"]["TRAIN"] = {}
Data["DATA"]["TRAIN"]["DATASET_ROOTS"] = ["/home/zhengwei/my_data/datasets"]
Data["DATA"]["TRAIN"]["DATASET_NAMES"] = ["sketchy"]
Data["DATA"]["TRAIN"]["IS_EPISODIC"] = False

Data["DATA"]["VALID"] = {}
Data["DATA"]["VALID"]["DATASET_ROOTS"] = ["/home/zhengwei/my_data/datasets"]
Data["DATA"]["VALID"]["DATASET_NAMES"] = ["sketchy"]

Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"] = {}
Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["NUM_WAYS"] = 5
Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"] = 5
Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY"] = 15
Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["MAX_NUM_QUERY"] = 15
Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["USE_DAG_HIERARCHY"] = False
Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["MIN_EXAMPLES_IN_CLASS"] = Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"]+Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY"]

Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["NUM_TASKS_PER_EPOCH"] = 600
Data["DATA"]["VALID"]["BATCH_SIZE"] = 8


Data["AUG"] = {}
Data["AUG"]["MEAN"] = [0.4712, 0.4499, 0.4031]
Data["AUG"]["STD"] = [0.2726, 0.2634, 0.2794]

Data["OUTPUT"] = "./nohup_logs/base_sketchy"

Data["MODEL"] = {}
Data["MODEL"]["TYPE"] = "CE"
Data["MODEL"]["CLASSIFIER"] = "finetune"
Data["MODEL"]["NAME"] = "vit_CE"


Data["MODEL"]["BACKBONE"] = 'vit'

Data["DATA"]["IMG_SIZE"] = 224
Data["DATA"]["NUM_WORKERS"] = 8
Data["GPU_ID"] = 0
Data["TRAIN"] = {}
Data["TRAIN"]["EPOCHS"] = 60

Data["DATA"]["TRAIN"]["BATCH_SIZE"] = 32

Data["TRAIN"]["BASE_LR"] = 0.1*Data["DATA"]["TRAIN"]["BATCH_SIZE"]/128



if not os.path.exists('./configs/sketchy'):
   os.makedirs('./configs/sketchy')

with open('./configs/sketchy/vit.yaml', 'w') as f:
   yaml.dump(data=Data, stream=f)