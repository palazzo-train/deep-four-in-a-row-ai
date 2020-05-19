import numpy as np 

_n_row = 6
_n_col = 7

BLANK = np.array([1,0,0])
GREEN = np.array([0,1,0])
RED =   np.array([0,0,1])

_blank_index = 0
_red_index = 1
_green_index = 2

class GameEnv():


    def __init__(self):
        self.board = np.zeros( [_n_row, _n_col, 3])
        self.board[:,:] = BLANK
        self.step = 0

    def move(self, color , col_pos, col_row):
        self.board[col_row, col_pos] = color

    def print_ascii(self, console=True):
        print_board = np.zeros( [_n_row, _n_col] , 'U1')

        print_board[ self.board[:,:,_blank_index] == 1] = '_'
        print_board[ self.board[:,:,_red_index] == 1] = 'O'
        print_board[ self.board[:,:,_green_index] == 1] = 'X'

        lines = []
        for r in print_board:
            line = ( ''.join( r ) )
            lines.append( line )


        print_line = '\n'.join(lines)

        if console:
            print_line = '\n' + print_line

        return print_line


