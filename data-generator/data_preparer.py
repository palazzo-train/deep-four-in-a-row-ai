import logging as l
import numpy as np
import game_env as g


def generate_1game_data(game_seq):
    l.info('test')

    count = len(game_seq)

    board_size = 6 * 7 * 3
    color_size = 3
    col_move_size = 7
    score_size = 1
    col_moves = np.zeros( [count, 7 ] )

    n_features = board_size + color_size + col_move_size + score_size
    data = np.zeros( [ count, n_features])

    ### even number
    if count % 2 == 0 : 
        start_index = 0
        start_score = 0
    else:
        start_index = 1
        start_score = 2

    boards = []
    colors = []


    index = 0
    for row in game_seq:
        board , col_pos, color, n_player_step , n_step , game_won  = row

        boards.append(board.reshape(-1))
        colors.append(color)
        col_moves[index,int(col_pos)] = 1

        index += 1

    boards = np.vstack(boards)
    colors = np.vstack(colors)

    data[:,:-1] = np.hstack([boards, col_moves, colors])

    ### assign score
    max_score = 100
    data[:,-1] = np.arange( start_score, max_score, (max_score)/count)
    data[start_index::2,-1] = data[start_index::2,-1] * -1
    data[-1,-1] = max_score 

    return data


    
