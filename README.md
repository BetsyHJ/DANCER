# DANCER
This repository contains the code used for the experiments in ["It Is Different When Items Are Older: Debiasing Recommendations When Selection Bias and User Preferences Are Dynamic"](https://arxiv.org/abs/2111.12481).

## Citation
If you use this code to produce results for your scientific publication, or if you share a copy or fork, please refer to our WSDM 2022 paper:
```
@inproceedings{huang-2022-different,
author = {Huang, Jin and Oosterhuis, Harrie and de Rijke, Maarten},
booktitle = {WSDM 2022: The Fifteenth International Conference on Web Search and Data Mining},
date-added = {2021-11-14 21:03:42 +0100},
date-modified = {2021-11-14 21:03:42 +0100},
month = {February},
publisher = {ACM},
title = {It Is Different When Items Are Older: Debiasing Recommendations When Selection Bias and User Preferences are Dynamic},
year = {2022}}
```

## Required packages
You can install conda and then create Python 3.6 Conda environment. 
Run conda create -n Dancer python=3.6 to create the environment.
Activate the environment by running conda activate Dancer. Then try to install the required packages:
```
$ pip install -r requirements.txt
```


## Reproducing Experiments
We compare time-aware and time-unaware methods to answer three research questions. 

### RQ1: Does item-age affect selection bias present in real-world logged data?
Reproducing the results of methods - MF, TMF, TTF++, and TMTF -, in observation prediction can be done with the following commands with the best hyperparameters given for each methods:
#### Random-split
```
$ cd examples
$ python run_exp.py --task OIPT --mode MF --setting random --lr 1e-5 --reg 1e-6 
$ python run_exp.py --task OIPT --mode TMF_v --setting random --lr 1e-5 --reg 1e-7
$ python run_exp.py --task OIPT --mode TF --setting random --lr 1e-5 --reg 1e-7
$ python run_exp.py --task OIPT --mode TMTF --setting random --lr 1e-4 --reg 1e-6  
```
#### Time-based split
```
$ cd examples
$ python run_exp.py --task OIPT --mode MF --setting time --lr 1e-4 --reg 1e-7 
$ python run_exp.py --task OIPT --mode TMF_v --setting time --lr 1e-4 --reg 1e-7 
$ python run_exp.py --task OIPT --mode TF --setting time --lr 1e-4 --reg 0
$ python run_exp.py --task OIPT --mode TMTF --setting time --lr 1e-4 --reg 1e-7  
```

### RQ2: Does item-age affect real-world user preferences?
Reproducing the results of methods - MF, TMF, TTF++, and TMTF -, in predicting ratings can be done with the following commands with the best hyperparameters given for each methods:
#### In the observed setting
```
$ cd examples
$ python run_exp.py --task OPPT --mode MF_v --setting naive --lr 1e-4 --reg 1e-4 
$ python run_exp.py --task OPPT --mode TMF_v --setting naive --lr 1e-4 --reg 1e-4
$ python run_exp.py --task OPPT --mode TF --setting naive --lr 1e-4 --reg 1e-5
$ python run_exp.py --task OPPT --mode TMTF --setting naive --lr 1e-4 --reg 1e-4
```
#### In the debiased setting
```
$ cd examples
$ python run_exp.py --task OPPT --mode MF_v --setting ips --lr 1e-2 --reg 1e-4
$ python run_exp.py --task OPPT --mode TMF_v --setting ips --lr 1e-2 --reg 1e-3
$ python run_exp.py --task OPPT --mode TF --setting ips --lr 1e-4 --reg 1e-6
$ python run_exp.py --task OPPT --mode TMTF --setting ips --lr 1e-4 --reg 1e-6 
```

### RQ3: Can TMF-DANCER better mitigate dynamic selection bias?
Reproducing the results of methods - MF, TMF, MF-StaticIPS, TMF-StaticIPS, MF-DANCER, and TMF-DANCER -, in predicting ratings can be done with the following commands with the best hyperparameters given for each methods:
```
$ cd examples
$ python run_exp.py --task TART --mode MF_v --setting naive --lr 1e-2 --reg 1e-4
$ python run_exp.py --task TART --mode TMF_v --setting naive --lr 1e-2 --reg 1e-4
$ python run_exp.py --task TART --mode MF_v --setting StaticIps --lr 1.0 --reg 1e-4
$ python run_exp.py --task TART --mode TMF_v --setting StaticIps --lr 1e-3 --reg 1e-1
$ python run_exp.py --task TART --mode MF_v --setting DANCER --lr 1e-4 --reg 1e-3
$ python run_exp.py --task TART --mode TMF_v --setting DANCER --lr 1e-4 --reg 1e-3
```

Moreover, results reported in the paper are the averages of 10 independent runs and can be reproduced with random seed 2012 ~ 2021 with seed setting e.g., ```--seed 2012```.

For using the simulation of MovieLens dataset in the third task, please unzip ```./data/simulation2.zip``` first.
