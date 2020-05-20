import numpy as np
import logging 
import logging as l
import sys, os
import data_generator.game_manager as gm
import model_ai.model as m
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




def test_gen1():
    gm.loop_games(200)


def test1():
    model = m.train_model()

def main():
    setupLogging()

    l.info('start')
    test1()


if __name__ == "__main__":
    main()
