import numpy as np
import logging 
import logging as l
import sys, os
import supervised_learning.data_generator.game_manager as gm
import supervised_learning.model_ai.trainer as trainer 
import tensorflow as tf

import global_config_supervised_learning as gc

def setupLogging():
    fileName = 'app.log'
    logPath = '.'
    path = os.path.join( logPath , fileName )

    format='%(asctime)s %(levelname)-8s - %(message)s'
    logFormatter = logging.Formatter(format)
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(logging.DEBUG)



def test_robot():
    import supervised_learning.model_ai.robot as robot 
    import game_env.game_env as g
    import supervised_learning.data_generator.game_manager  as gm

    working_folder = gc.C_save_model_current_folder 
    base_folder = gc.C_save_model_base_folder
    save_model_path = './{}/{}/savemodel/my_model'.format(base_folder, working_folder)

    print(save_model_path)
    win_rate, draw_rate, loss_rate = gm.robot_evaluate_by_path(save_model_path)

    print(win_rate, draw_rate, loss_rate)

def generate_data():
    n_example = 200
    # n_example = 170000
    gm.loop_games(n_example)

def training():
    model = trainer.train_model()


def supervised_main():

    l.info('start')
    # generate_data()
    # training()
    test_robot()


