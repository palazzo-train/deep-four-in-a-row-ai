import numpy as np
import logging 
import logging as l
import sys, os
from game_env import GameEnv, RED, GREEN


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

def main():
    setupLogging()

    l.info('in sdd')

    g = GameEnv()

    g.move(RED, 2)
    g.move(GREEN, 2)
    g.move(RED, 1)
    g.move(GREEN, 4)


    l.info( g.print_ascii() )
    l.info(g.is_win(GREEN) )
    
    g.move(RED, 5)
    g.move(RED, 5)
    g.move(RED, 5)
    g.move(RED, 5)
    l.info( g.print_ascii() )
    l.info(g.is_win(RED) )






if __name__ == "__main__":
    main()
