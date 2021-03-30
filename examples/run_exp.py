import sys
sys.path.append('../src/')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import random

import numpy as np
from time import time, localtime, strftime
import configparser

import torch

from util.data import Dataset
from offlineExp.gru4rec import GRU4Rec
from offlineExp.tmf import TMF, TMF_variety, TMF_fast, TMF_fast_variety
from offlineExp.mf import MF, MF_v
from offlineExp.tf import TF, TMTF
# from trainer.trainer import TARS_Trainer as Trainer
from trainer.trainer import OP_Trainer 
from trainer.trainer import OPPT_Trainer

from evaluator.evaluator import OP_Evaluator, OPPT_Evaluator

# np.random.seed(2020)

def _get_conf(conf_name):
    config = configparser.ConfigParser()
    config.read("../conf/"+conf_name+".properties")
    conf=dict(config.items("default"))
    conf['mode'] = conf['mode'].lower()
    return conf

def _logging_(basis_conf, params_conf):
    now = localtime(time())
    now = strftime("%Y-%m-%d %H:%M:%S", now)
    origin_data_name = basis_conf["data.input.dataset"]
    debiasing = basis_conf["debiasing"]
    print(now + " - data: %s" % origin_data_name)
    print(now + " - task: %s" % (params_conf['task']))
    print(now + " - model: %s, debiasing: %s" % (basis_conf['mode'], debiasing))
    # print(now + " - use gpu: %s" % (basis_conf['use_gpu']))
    if ("evaluation" in basis_conf) and (basis_conf['evaluation'].lower() == 'true'):
        print(now + " - directly load well-trained model and evaluate", flush=True)
    
    if basis_conf['mode'][0] != 'b': # baselines do not have params
        print("conf : " + str(params_conf), flush=True)
    

def run_dqn():
    conf = _get_conf('ml-100k')
    # for multiple jobs in 
    args = set_hparams()
    if args.mode is not None:
        conf['mode'] = args.mode.lower()

    # init DQN
    config = load_parameters(conf['mode'])
    if args.lr is not None:
        config["learning_rate"] = args.lr
    if args.reg is not None:
        config['l2_reg'] = args.reg
    if args.seed is not None:
        config['seed'] = args.seed

    # # set random seed
    if 'seed' in config:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])

    task = 'OIPT'
    # task = 'OPPT'
    if 'task' in conf:
        task = conf['task']
    config['task'] = task
    _logging_(conf, config)
    ## loading data
    data = Dataset(conf, task=task)
    # ctr = data.train['ctr']
    
    # # add random-splitting for task 1: OIPT
    if task == 'OIPT':
        config['splitting'] = 'time'
        if 'splitting' in conf:
            config['splitting'] = conf['splitting']

    # Super simple baselines just need some statistic info without training process.
    if 'b' in conf['mode']:
        if config['task'] == 'OIPT':
            evaluator = OP_Evaluator(config, None, data)
            # evaluator.evaluate(baselines=conf['mode'], subset='neg')
            for subset in [None, 'pos', 'neg']:
                evaluator.evaluate(baselines=conf['mode'], subset=subset)
            # for subset in [None, 'pos', 'neg']:
            #     for i in range(1, 4):
            #         print("\n*-*-*-*-*- B%d -*-*-*-*-*" % i)
            #         evaluator.evaluate(baselines='b%d'%i, subset=subset)
        else:
            evaluator = OPPT_Evaluator(None, None, data)
            evaluator.evaluate(baselines=conf['mode'])
        exit(0)

    # add some fixed parameters
    config['path'] = conf['data.input.path']
    config['dataset'] = conf['data.input.dataset']
    config['epochs'] = 500
    if conf['debiasing'].lower() == 'ips':
        config['debiasing'] = True
    else:
        config['debiasing'] = False
    
    if conf['mode'].lower() == "tmf":
        MODEL = TMF
    elif conf['mode'].lower() == "tmf_v":
        MODEL = TMF_variety
    elif conf['mode'].lower() == "tmf_fast":
        MODEL = TMF_fast
    elif conf['mode'].lower() == "tmf_fast_v":
        MODEL = TMF_fast_variety
    elif conf['mode'].lower() == "tmtf":
        MODEL = TMTF
    elif conf['mode'].lower() == "tf":
        MODEL = TF
    elif conf['mode'].lower() == "mf":
        MODEL = MF
    elif conf['mode'].lower() == "mf_v":
        MODEL = MF_v
    elif conf['mode'].lower() == "gru4rec":
        MODEL = GRU4Rec
    else:
        raise NotImplementedError("Make sure 'mode' in ['GRU4Rec', 'TMF', 'TMF_fast', 'MF', 'TF', 'TMTF']!")

    # # train process
    config['mode'] = conf['mode']
    model = MODEL(config, data, debiasing=config['debiasing'])
    if ('task' in config) and (config['task']=='OPPT'):
        Trainer = OPPT_Trainer
    else:
        Trainer = OP_Trainer
    trainer = Trainer(config, model, data)
    if ("evaluation" in conf) and (conf['evaluation'].lower() == 'true'):
        print("Directly load well-trained model and evaluate")
        trainer.load_model()
        print("saved params:", model.state_dict().keys())
    else:
        model = trainer.fit()

    # evaluate process
    model.eval()
    if ('task' in config) and (config['task']=='OPPT'):
        Evaluator = OPPT_Evaluator
    else:
        Evaluator = OP_Evaluator
    evaluator = Evaluator(config, model, data)
    evaluator.evaluate()
    # evaluator.evaluate(ub='false')
    # evaluator.evaluate(ub='snips')
    # for thr in [1e-2, 1e-3, 1e-4]:
    #     evaluator.evaluate(ub='pop', threshold=thr)
    #     evaluator.evaluate(ub='unpop', threshold=thr)
    

def load_parameters(mode):
    params = {}
    config = configparser.ConfigParser()
    if 'tmf_fast_v' in mode.lower():
        mode = 'tmf_fast'
    elif 'tmf_v' in mode.lower():
        mode = 'tmf'
    elif 'mf_v' in mode.lower():
        mode = 'mf'
    elif mode[0] == 'b':
        return {}
    config.read("../conf/"+mode+".properties")
    conf=dict(config.items("hyperparameters"))
    return conf

def set_hparams():
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--seed', type=int)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--reg', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--mode', type=str, default=None)
    args = parser.parse_args()
    print("now lr is", args.lr, ", reg is", args.reg, ", seed is", args.seed, ", and mode is", args.mode, flush=True)
    return args

if __name__ == "__main__":
    run_dqn()
    print("End. " + strftime("%Y-%m-%d %H:%M:%S", localtime(time())))
# print("checkpoint")