import numpy as np
import logging 
import logging as l
import sys, os
from game_env import GameEnv, RED, GREEN
from random_robot_players import RobotA


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
    p_r = RobotA()
    p_g = RobotA()

    won = False
    count = 0 

    while not won:
        valid_move, won = g.move(RED,p_r.move())
        l.info( g.print_ascii() )
        if won:
            break

        valid_move, won = g.move(GREEN,p_g.move())
        l.info( g.print_ascii() )
        if won:
            break

        count += 1

        if count > 60 :
            break

    print("count {} win? {}".format(count, won))


    return

    g.move(RED, 2)
    g.move(GREEN, 2)
    g.move(RED, 1)
    w = g.move(GREEN, 4)


    l.info( g.print_ascii() )
    # l.info(g.is_win(GREEN) )
    
    g.move(RED, 5)
    g.move(RED, 5)
    w = g.move(RED, 5)
    l.info( g.print_ascii() )

    a = RobotA()
    w = g.move(RED, a.move() )
    l.info( g.print_ascii() )

    l.info( g.print_ascii() )
    # l.info(g.is_win(RED) )

    print( ( g.step_trace ) )
    print( len( g.step_trace ) )






if __name__ == "__main__":
    main()
