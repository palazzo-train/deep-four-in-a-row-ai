import numpy as np 

_n_row = 6
_n_col = 7

BLANK = np.array([1,0,0])
GREEN = np.array([0,1,0])
RED =   np.array([0,0,1])

_blank_index = 0
_green_index = 1
_red_index = 2

def __create_winning_mask():
    masks = []
    w = np.ones( 4)

    ## horizontal
    for row in range(_n_row):
        for col in range(_n_col - 3):
            mask = np.zeros( [_n_row ,_n_col] )
            mask[row,col:col+4] = w
            masks.append(mask)

    ## vertical
    for row in range(_n_row - 3):
        for col in range(7):
            mask = np.zeros( [_n_row,_n_col] )
            mask[row:row+4,col] = w
            masks.append(mask)

    ## diagonal
    w1 = np.array( [ [ 1, 0, 0, 0] ,
                    [ 0, 1, 0, 0]  , 
                    [ 0, 0, 1, 0]  , 
                    [ 0, 0, 0, 1] ] )
    w2 = np.flip(w1, 0)

    ws = [w1, w2]

    for w in ws:
        for row in range(_n_row - 3):
            for col in range(_n_col - 3):
                mask = np.zeros( [_n_row,_n_col] )
                mask[row:row+4,col:col+4] = w2
                masks.append(mask)

    return np.stack(masks)

_global_winning_mask = __create_winning_mask()

class GameEnv():
    def __init__(self):
        self.board = np.zeros( [_n_row, _n_col, 3])
        self.board[:,:] = BLANK
        self.step = 0
        self.next_row_pos = np.zeros( _n_col , dtype='int')
        self.win_masks = self.__get_winning_masks()
    
    def __get_winning_masks(self):
        return _global_winning_mask.copy()

    def __move_exact(self, color , col_pos, col_row):
        self.board[col_row, col_pos] = color

    def is_win(self,color):
        if color[_green_index] == 1 :
            index = _green_index
        elif color[_red_index] == 1:
            index = _red_index

        masked = (self.board[:,:,index] * self.win_masks)
        count = masked.sum(axis=1).sum(axis=1).max()

        if count == 4:
            return True
        else:
            return False


    def move(self, color , col_pos):
        col_row = self.next_row_pos[col_pos]
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


