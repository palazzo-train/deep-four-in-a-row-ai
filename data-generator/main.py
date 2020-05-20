import numpy as np
import logging 
import logging as l
import sys, os
from game_env import GameEnv, RED, GREEN
import random_robot_players as rp
import data_preparer as dp

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


def manual_test():
    g = GameEnv()

    moves = [ 6 , 2 , 1 , 5 , 1 , 5 , 1 , 5, 1]

    isRed= True 
    for m in moves: 
        if isRed: 
            color = RED
        else:
            color = GREEN

        sc = g.test_all_moves(color)
        valid_move, won , board = g.move(color,m)
        postsc = g.test_all_moves(color)
        print('')
        print('')
        print('******** isRED {} won {} , move {} suggested {} {}'.format( isRed, won, m, sc, postsc ))
        print( g.print_ascii() )

        isRed= not isRed

def main():
    setupLogging()

    l.info('start')
    g = GameEnv()

    red_robots = rp.getRobots(RED,GREEN)
    green_robots = rp.getRobots(GREEN,RED)

    r_idx = np.random.choice( len(red_robots))
    g_idx = np.random.choice( len(green_robots))
    p_r = red_robots[r_idx]
    p_g = green_robots[g_idx]

    won = False
    count = 0
    while not won:
        print('******** count {} RED'.format(count))
        c0 = p_r.move(g)
        oc = g.test_all_moves(RED)
        dc = g.test_all_moves(GREEN)
        valid_move, won , board = g.move(RED,c0)
        l.info( g.print_ascii() )
        print('red suggested : offense {} defense {} , actual {}'.format(oc,dc,c0))
        print('won : {}, valid {}'.format(won, valid_move))
        if won:
            break

        count += 1
        print('******** count {} GREEN'.format(count))
        c0 = p_g.move(g)
        oc = g.test_all_moves(GREEN)
        dc = g.test_all_moves(RED)
        valid_move, won , board = g.move(GREEN,c0)
        l.info( g.print_ascii() )
        print('green suggested : offense {} defense {} , actual {}'.format(oc,dc,c0))
        print('won : {}, valid {}'.format(won, valid_move))
        if won:
            break

        count += 1

        if count > 43 :
            break

    print('red lv  : {}'.format(p_r.name) )
    print('green lv: {}'.format(p_g.name) )

    seqs = g.step_trace

    dp.generate_1game_data(seqs)





if __name__ == "__main__":
    main()
