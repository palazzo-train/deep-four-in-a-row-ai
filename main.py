import numpy as np
import logging 
import logging as l
import sys, os
import data_generator.game_manager as gm
import model_ai.trainer as trainer 
import tensorflow as tf

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




def generate_data():
    n_example = 200000
    # n_example = 1000
    gm.loop_games(n_example)

def training():
    model = trainer.train_model()

def main():
    setupLogging()

    l.info('start')
    # generate_data()
    training()


if __name__ == "__main__":
    main()
