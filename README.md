# DANCER
This repository contains the code used for the experiments in "It Is Different When Items Are Older: Debiasing Recommendations When Selection Bias and User Preferences Are Dynamic".

## Required packages
TBD.

## Reproducing Experiments
We compare time-aware and time un-aware methods to answer three research questions. 

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
Reproducing the results of methods - MF, TMF-, MF-StaticIPS, TMF-StaticIPS, MF-DANCER, and TMF-DANCER -, in predicting ratings can be done with the following commands with the best hyperparameters given for each methods:
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

For using the simulation of MovieLens dataset in third task, please unzip ```./data/simulation2.zip```.
