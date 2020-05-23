import numpy as np
import logging 
import logging as l
import sys, os
from game_env.game_env import GameEnv, RED, GREEN
from . import random_robot_players as rp
import model_ai.robot as robot 
import global_config as gc



def play_game(g, red_player , green_player):

    red_player.reset()
    green_player.reset()
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
            valid_move, game_end, won , board = g.move( color ,c0)
            player.move_valid_feedback(c0, valid_move)

        move_count += 1

        if won:
            break

        if move_count >= 42 or game_end :
            break

        active_idx = (( active_idx + 1 )% 2 )
    return won , move_count, player 

def load_data():
    path = os.path.join( gc.C_save_data_folder , 'data.npz' )
    l.info('loading data {}'.format(path))

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

    return data_train, data_dev, data_test, win_stat, move_count_stat, winner_level_stat


def _save_data(data_train, data_dev, data_test, win_stat, move_count_stat, winner_level_stat):
    path = os.path.join( gc.C_save_data_folder , 'data.npz' )
    l.info('saving data {}'.format(path))

    with open(path, 'wb') as f:
        np.savez(f, data_train=data_train, 
        data_dev = data_dev,
        data_test = data_test ,
        win_stat=win_stat, move_count_stat=move_count_stat, winner_level_stat=winner_level_stat)

    # with open('data_stats.npz', 'rb') as f:
    #     dd = np.load(f)
    #     ddd = dd['win_stat']


def _create_data(data, win_stat, move_count_stat, winner_level_stat):
    l.info('saving data shape {}'.format(data.shape))

    # np.random.shuffle(data)

    n = data.shape[0]

    train_end = int(n * gc.DATA_TRAINING_SET_RATIO ) 
    dev_end = int( n * ( gc.DATA_TRAINING_SET_RATIO + gc.DATA_DEV_SET_RATIO ) )

    data_train = data[0:train_end]
    data_dev = data[train_end:dev_end]
    data_test = data[dev_end:]

    _save_data(data_train, data_dev, data_test, win_stat, move_count_stat, winner_level_stat)

    l.info('saving completed')


def loop_games(n_game=200, save_game_to_file=True, with_last_step=False):
    l.info('start')
    red_robots = rp.getRobots(RED,GREEN)
    green_robots = rp.getRobots(GREEN,RED)

    all_game_history, win_stat, move_count_stat, winner_level_stat , _ = loop_games_between_robots(
                red_robots, green_robots, n_game, save_game_to_file, with_last_step)

    l.info(' vstacking data')
    data = np.vstack( all_game_history )
    del all_game_history

    if save_game_to_file:
        _create_data(data, win_stat, move_count_stat, winner_level_stat)

    return data

def loop_games_between_robots(red_robots, green_robots, n_game, save_game_to_file=True, with_last_step=False, display_step=100):
    g = GameEnv()

    total_move = 0

    move_count_stat = np.zeros( n_game )
    win_stat = np.zeros( n_game )
    winner_level_stat = np.zeros( n_game )
    winner_names = np.chararray( n_game , 16)

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
        winner_names[gi] = winner.name

        won_count += ( 1 if won else 0 )
        total_move += move_count

        if gi % display_step == 0:
            # l.info('game [{}] move count [{}] player levels [{}] vs [{}]'.format( gi , move_count , red_player.name , green_player.name ))
            l.info('Total {} games played. {} won. average step per game {}'.format(gi, won_count, total_move / (gi+1) ) )

        # if move_count <= 6:
        #     print('****')
        #     print(g.print_ascii())
        #     break
        all_game_history.append( g.get_history(with_last_step) )

    
    l.info('Total {} games played. {} won. total step {}=={}. average step per game {}'.format(
        n_game, win_stat.sum(), move_count_stat.sum(), total_move, move_count_stat.sum() / n_game) )

    return all_game_history, win_stat, move_count_stat, winner_level_stat , winner_names




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



def robot_evaluate_by_path(save_model_path):
    r = robot.Robot(RED, GREEN, save_model_path)
    g = robot.Robot(GREEN, RED, save_model_path)

    win_rate, draw_rate, loss_rate = robot_evaluate( r , g )

    return win_rate, draw_rate, loss_rate 

def robot_evaluate_by_model(model):
    r = robot.ModelRobot(RED, GREEN, model)
    g = robot.ModelRobot(GREEN, RED, model)

    win_rate, draw_rate, loss_rate = robot_evaluate( r , g )
    return win_rate, draw_rate, loss_rate 

def robot_evaluate(red_ai , green_ai):
    l.info(' evaluate ai robot')

    green_opponent= rp.getRobots(GREEN, RED, at_level = 2)
    red_opponent= rp.getRobots(RED, GREEN, at_level = 2)

    ## first as red
    red_robots = [ red_ai ]
    green_robots = green_opponent

    n_game = int(gc.EVAL_ROBOT_EVAL_BY_MODE_GAME_NUM /2 )
    _, win_stat , _, _, winner_names = loop_games_between_robots(
                red_robots, green_robots, n_game, save_game_to_file=False, with_last_step=False, display_step=100)

    n_as_red_win = ( winner_names == bytes(red_ai.name, 'utf-8') ).sum()
    n_as_red_draw = n_game - win_stat.sum()

    ## first as green 
    red_robots = red_opponent
    green_robots = [ green_ai ]
    _, _, _, _, winner_names = loop_games_between_robots(
                red_robots, green_robots, n_game, save_game_to_file=False, with_last_step=False, display_step=100)

    n_as_green_win = ( winner_names == bytes(green_ai.name, 'utf-8') ).sum()
    n_as_green_draw = n_game - win_stat.sum()

    win_rate = (( n_as_red_win + n_as_green_win ) / (2 * n_game) )
    draw_rate = (( n_as_red_draw + n_as_green_draw ) / (2 * n_game) )
    loss_rate = 1 - win_rate - draw_rate

    return win_rate, draw_rate, loss_rate 