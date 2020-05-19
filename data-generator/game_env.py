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
        self.next_row_pos = np.zeros( _n_col , dtype='int')

    def __move_exact(self, color , col_pos, col_row):
        self.board[col_row, col_pos] = color

    def move(self, color , col_pos):
        col_row = self.next_row_pos[col_pos]
        print(color)
        print(color.shape)
        self.board[col_row, col_pos] = color
        self.next_row_pos[col_pos] += 1

    def print_ascii(self, console=True):
        print_board = np.zeros( [_n_row, _n_col] , 'U1')

        print_board[ self.board[:,:,_blank_index] == 1] = '_'
        print_board[ self.board[:,:,_red_index] == 1] = 'O'
        print_board[ self.board[:,:,_green_index] == 1] = 'X'

        print_board = np.flip(print_board, 0) 

        lines = []
        for r in print_board:
            line = ( ''.join( r ) )
            lines.append( line )


        print_line = '\n'.join(lines)

        if console:
            print_line = '\n' + print_line

        return print_line


