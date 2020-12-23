import sys
sys.path.append('../src/')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
from time import time, localtime, strftime
import configparser

from util.data import Dataset
from offlineExp.gru4rec import GRU4Rec
from offlineExp.tmf import TMF
from offlineExp.tmf import TMF_fast
from offlineExp.mf import MF
from offlineExp.tf import TF
# from trainer.trainer import TARS_Trainer as Trainer
from trainer.trainer import OP_Trainer as Trainer

from evaluator.evaluator import OP_Evaluator as Evaluator

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
    print(now + " - model: %s" % (basis_conf['mode']))
    print(now + " - debiasing: %s" % (debiasing))
    # print(now + " - use gpu: %s" % (basis_conf['use_gpu']))
    if ("evaluation" in basis_conf) and (basis_conf['evaluation'].lower() == 'true'):
        print(now + " - directly load well-trained model and evaluate")
    print("conf : " + str(params_conf))
    

def run_dqn():
    conf = _get_conf('ml-100k')

    # init DQN
    config = load_parameters(conf['mode'])
    
    # tuning = 'learning_rate'.upper()
    # tuning = 'memory_size'.upper()
    # tuning = 'batch_size'.upper()
    # tuning = 'gamma'.upper()
    # tuning = 'optimizer'.upper()
    # tuning = 'replace_targetnet'.upper()
    # tuning = 'epsilon_decay_step'
    # tuning = 'lr_decay_step'
    # tuning = "state_encoder"
    # tuning = 'action_dim'.upper()
    # tuning = 'RNN_STATE_DIM'
    # print("tuning:",tuning)
    # config['SAVE_MODEL_FILE'] = conf["data.input.dataset"] + '_' + \
    #     conf["data.gen_model"] + '_' + conf["data.debiasing"] + '_' + \
    #     conf['mode'] + '_' + config["state_encoder"] + '_' + 'r01_SmoothL1_' + 'notrick_' + tuning + str(config[tuning]) + '_' 
    # config['SAVE_MODEL_FILE'] = 'sim_random_' + str(num_users) + '_' + str(action_space) + '_' + config["state_encoder"] + '_'

    _logging_(conf, config)
    # add some fixed parameters
    config['dataset'] = conf['data.input.dataset']
    config['epochs'] = 100
    if conf['debiasing'].lower() == 'ips':
        config['debiasing'] = True
    else:
        config['debiasing'] = False

    ## loading data
    data = Dataset(conf)
    # ctr = data.train['ctr']
    
    if conf['mode'].lower() == "tmf":
        MODEL = TMF
    elif conf['mode'].lower() == "tmf_fast":
        MODEL = TMF_fast
    elif conf['mode'].lower() == "tf":
        MODEL = TF
    elif conf['mode'].lower() == "mf":
        MODEL = MF
    elif conf['mode'].lower() == "gru4rec":
        MODEL = GRU4Rec
    else:
        NotImplementedError("Make sure 'mode' in ['GRU4Rec', 'TMF', 'TMF_fast', 'MF', 'TF']!")
    ## train process
    config['mode'] = conf['mode']
    model = MODEL(config, data, debiasing=config['debiasing'])
    trainer = Trainer(config, model, data)
    if ("evaluation" in conf) and (conf['evaluation'].lower() == 'true'):
        print("Directly load well-trained model and evaluate")
        trainer.load_model()
        print("saved params:", model.state_dict().keys())
    else:
        model = trainer.fit()

    # evaluate process
    model.eval()
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
    config.read("../conf/"+mode+".properties")
    conf=dict(config.items("hyperparameters"))
    return conf
    

if __name__ == "__main__":
    run_dqn()
    print("End. " + strftime("%Y-%m-%d %H:%M:%S", localtime(time())))
# print("checkpoint")