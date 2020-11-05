import sys
sys.path.append('../src/')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
from time import time, localtime, strftime
import configparser

from util.data import Dataset
from offlineExp.gru4rec import GRU4Rec
from trainer.trainer import Trainer

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
    debiasing = basis_conf["data.debiasing"]
    print(now + " - data:%s" % origin_data_name)
    print(now + " - debiasing:%s" % (debiasing))
    print(now + " - model: %s" % (basis_conf['mode']))
    print(now + " - use gpu: %s" % (basis_conf['usegpu']))
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
    
    data = Dataset(conf)
    ctr = data.train['ctr']
    print(max(ctr), min(ctr))

    ## train process
    model = GRU4Rec(config, data)
    trainer = Trainer(config, model, data)
    trainer.fit()
    # evalProcess = conf['evaluation']
    # if evalProcess.lower() == 'false':
    #     train(conf, config, sofa)
    # else:
    #     evaluate(conf, config, sofa)


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