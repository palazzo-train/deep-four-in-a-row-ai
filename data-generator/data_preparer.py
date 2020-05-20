import logging as l
import numpy as np
import game_env as g


def generate(game_seq):
    l.info('test')

    for row in game_seq:
        board , col_pos, color, n_player_step , n_step , game_won  = row
        print('step {}  game won {} player step {} color {}'.format(n_step, game_won , n_player_step, color) )
        # print(g.board_to_ascii(board))