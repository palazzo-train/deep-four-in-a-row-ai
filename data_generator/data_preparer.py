import logging as l
import numpy as np
import game_env as g


def game_seq_study(game_seq):
    l.info('study')

    step = 1
    for row in game_seq:
        board , col_pos, color, n_player_step , n_step , game_won  = row

        print('step {} color {}'.format(step,color))
        print(g.board_to_ascii(board))

        step += 1


def generate_games_data(list_game_seq):
    data = []
    for seq in list_game_seq: 
        d = generate_1game_data(seq)

        data.append(d)

    data = np.vstack(data)

    return data

def generate_1game_data(game_seq):
    count = len(game_seq)

    board_size = 6 * 7 * 3
    color_size = 3
    col_move_size = 7
    score_size = 1
    col_moves = np.zeros( [count, 7 ] )

    n_features = board_size + color_size + col_move_size + score_size
    data = np.zeros( [ count, n_features])

    winner_start_score = 0.02
    max_score = 1.0
    ### even number
    if count % 2 == 0 : 
        start_index = 0
        start_score = 0
    else:
        start_index = 1
        start_score = winner_start_score 

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
    data[:,-1] = np.arange( start_score, max_score, (max_score)/count)

    ## if someone won, then someone loss, negative the score for the losser
    ## if no one won, both get positive score
    if game_won :
        data[start_index::2,-1] = data[start_index::2,-1] * -1

    data[-1,-1] = max_score 

    return data


    
