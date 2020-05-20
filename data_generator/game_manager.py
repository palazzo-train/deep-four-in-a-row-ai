import numpy as np
import logging 
import logging as l
import sys, os
from game_env.game_env import GameEnv, RED, GREEN
from . import random_robot_players as rp
from . import data_preparer as dp


def play_game(g, red_player , green_player):
    won = False
    active_idx = np.random.choice( 2 )
    players = [ red_player , green_player ]
    player_color = [ RED, GREEN ]

    move_count = 0
    while not won:
        player = players[ active_idx ]
        color = player_color[active_idx]

        valid_move = False
        while not valid_move:
            c0 = player.move(g)
            valid_move, won , board = g.move( color ,c0)

        move_count += 1

        if won:
            break

        if move_count >= 42 :
            break

        active_idx = (( active_idx + 1 )% 2 )
    return won , move_count, player 

def loop_games(n_game=200):
    l.info('start')
    g = GameEnv()

    red_robots = rp.getRobots(RED,GREEN)
    green_robots = rp.getRobots(GREEN,RED)

    total_move = 0

    move_count_stat = np.zeros( n_game )
    win_stat = np.zeros( n_game )
    winner_level_stat = np.zeros( n_game )

    all_game_seq = []
    won_count = 0
    for gi in range(n_game):
        g.reset()
        r_idx = np.random.choice( len(red_robots))
        g_idx = np.random.choice( len(green_robots))
        red_player = red_robots[r_idx]
        green_player = green_robots[g_idx]

        won, move_count , winner = play_game(g, red_player , green_player)

        move_count_stat[gi] = move_count
        win_stat[gi] = won
        winner_level_stat[gi] = winner.smart_level

        won_count += ( 1 if won else 0 )
        total_move += move_count

        if gi % 100 == 1:
            # l.info('game [{}] move count [{}] player levels [{}] vs [{}]'.format( gi , move_count , red_player.name , green_player.name ))
            l.info('Total {} games played. {} won. average step per game {}'.format(gi, won_count, total_move / gi ) )

        # if move_count <= 6:
        #     print('****')
        #     print(g.print_ascii())
        #     dp.game_seq_study(g.step_trace)
        #     break

        all_game_seq.append( g.step_trace )

    
    l.info('Total {} games played. {} won. total step {}=={}. average step per game {}'.format(
        n_game, win_stat.sum(), move_count_stat.sum(), total_move, move_count_stat.sum() / n_game) )
    l.info('generating data')
    data = dp.generate_games_data(all_game_seq)

    l.info('saving data shape {}'.format(data.shape))

    # np.savetxt("data.csv", data, delimiter=",")
    with open('data/data.npy', 'wb') as f:
        np.save(f, data)

    l.info('saving data stats')
    with open('data/data_stats.npz', 'wb') as f:
        np.savez(f, win_stat=win_stat, move_count_stat=move_count_stat, winner_level_stat=winner_level_stat)

    # with open('data_stats.npz', 'rb') as f:
    #     dd = np.load(f)
    #     ddd = dd['win_stat']

    l.info('saving completed')

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

def test22():
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