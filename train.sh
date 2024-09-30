

# training CE model
# python write_yaml_CE_sketchy.py

# nohup python -u main.py --cfg configs/sketchy/vit.yaml --tag main > vitCE_base_sketchy.log 2>&1 &





#test
python write_yaml_test_sketchy_tune.py
nohup python -u main.py --cfg configs/sketchy/evaluation/finetune_sketchy_vit_CE.yaml --tag test > vitCE_finetune_novel_sketchy_a61w5s.log 2>&1 &











# training PN model
# python write_yaml_PN.py
# python main.py --cfg configs/PN/miniImageNet_res12.yaml --tag main

# searching for hyperparameters for finetune.
# python write_yaml_search.py
# python search_hyperparameter.py --cfg configs/search/finetune_res12_CE.yaml
