import numpy as np
import logging 
import logging as l
import sys, os
from game_env.game_env import GameEnv, RED, GREEN
from . import random_robot_players as rp


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

def test_load_data():
    with open('data/data.npz', 'rb') as f:
        dd = np.load(f)
        data_train = dd['data_train']
        data_dev = dd['data_dev']
        data_test = dd['data_test']
        win_stat = dd['win_stat']
        move_count_stat = dd['move_count_stat']
        winner_level_stat = dd['winner_level_stat']

    print(data_train.shape)
    print(data_dev.shape)
    print(data_test.shape)
    print(win_stat.shape)
    print(move_count_stat.shape)
    print(winner_level_stat.shape)


def _save_data(data_train, data_dev, data_test, win_stat, move_count_stat, winner_level_stat):
    l.info('saving data ')
    with open('data/working/data.npz', 'wb') as f:
        np.savez(f, data_train=data_train, 
        data_dev = data_dev,
        data_test = data_test ,
        win_stat=win_stat, move_count_stat=move_count_stat, winner_level_stat=winner_level_stat)

    # with open('data_stats.npz', 'rb') as f:
    #     dd = np.load(f)
    #     ddd = dd['win_stat']
def _create_data(data, win_stat, move_count_stat, winner_level_stat):
    l.info('saving data shape {}'.format(data.shape))

    np.random.shuffle(data)

    n = data.shape[0]

    train_end = int(n * 0.86)
    dev_end = int( n * ( 0.86 + 0.07 ) )

    data_train = data[0:train_end]
    data_dev = data[train_end:dev_end]
    data_test = data[dev_end:]

    _save_data(data_train, data_dev, data_test, win_stat, move_count_stat, winner_level_stat)

    l.info('saving completed')





def loop_games(n_game=200, save_game_to_file=True):
    l.info('start')
    red_robots = rp.getRobots(RED,GREEN)
    green_robots = rp.getRobots(GREEN,RED)

    all_game_history, win_stat, move_count_stat, winner_level_stat = loop_games_between_robots(red_robots, green_robots, n_game, save_game_to_file)

    l.info(' vstacking data')
    data = np.vstack( all_game_history )
    del all_game_history

    if save_game_to_file:
        _create_data(data, win_stat, move_count_stat, winner_level_stat)

    return data

def loop_games_between_robots(red_robots, green_robots, n_game, save_game_to_file=True):
    g = GameEnv()

    total_move = 0

    move_count_stat = np.zeros( n_game )
    win_stat = np.zeros( n_game )
    winner_level_stat = np.zeros( n_game )

    all_game_history = []
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
        #     break
        all_game_history.append( g.get_history() )

    
    l.info('Total {} games played. {} won. total step {}=={}. average step per game {}'.format(
        n_game, win_stat.sum(), move_count_stat.sum(), total_move, move_count_stat.sum() / n_game) )

    return all_game_history, win_stat, move_count_stat, winner_level_stat , 




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
