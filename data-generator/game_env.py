import numpy as np 

### constants
_n_row = 6
_n_col = 7

GREEN = np.array([1,0,0])
RED =   np.array([0,1,0])
BLANK = np.array([0,0,1])

_green_index = 0
_red_index = 1
_blank_index = 2

def __m_create_winning_mask():
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

    for w in [w1, w2]:
        for row in range(_n_row - 3):
            for col in range(_n_col - 3):
                mask = np.zeros( [_n_row,_n_col] )
                mask[row:row+4,col:col+4] = w
                masks.append(mask)

    ### size [ n , _n_row , _n_col ]
    return np.stack(masks)

### init global winning mask
_global_winning_masks = __m_create_winning_mask()



def _m_is_win(color, board):
    index = _m_color_to_index(color)

    masked = (board[:,:,index] * _global_winning_masks)
    count = masked.sum(axis=1).sum(axis=1).max()

    if count == 4:
        return True
    else:
        return False


def _m_color_to_index(color):
    if color[_green_index] == 1 :
        index = _green_index
    elif color[_red_index] == 1:
        index = _red_index

    return index

def _m_move_test(board, next_row_pos , color , col_pos):
    game_won = False
    valid_move = False 

    col_row = next_row_pos[col_pos]

    if col_row >= _n_row :
        return valid_move, game_won 

    valid_move = True
    board[col_row, col_pos] = color

    game_won = _m_is_win(color,board)

    return valid_move, game_won


class GameEnv():
    def __init__(self):
        self.board = np.zeros( [_n_row, _n_col, 3])
        self.board[:,:] = BLANK
        self.n_player_step = np.array( [0,0])
        self.n_step = 0
        self.next_row_pos = np.zeros( _n_col , dtype='int')
        self.win_masks = self.__get_winning_masks()

        self.winner = BLANK
        self.step_trace = []
    
    def __get_winning_masks(self):
        return _global_winning_masks.copy()

    def __move_exact(self, color , col_pos, col_row):
        self.board[col_row, col_pos] = color


    def test_all_moves(self, color):
        for col_pos in range(_n_col):
            _ , game_won = _m_move_test(self.board.copy(), self.next_row_pos , color , col_pos)
            print('kkkkkkk test    col {}  won {} color {}'.format( col_pos , game_won, color))

            if game_won:
                return col_pos

        return -1

    def test_move(self, color, col_pos):
        valid_move, game_won = _m_move_test(self.board.copy(), self.next_row_pos , color , col_pos)

        return valid_move, game_won, self.board 


    def move(self, color , col_pos):
        valid_move, game_won = _m_move_test(self.board.copy(), self.next_row_pos , color , col_pos)

        if not valid_move:
            return valid_move, game_won, self.board 

        ## commit move
        col_row = self.next_row_pos[col_pos]
        self.board[col_row, col_pos] = color
        self.next_row_pos[col_pos] += 1

        index = _m_color_to_index(color)
        game_step = ( self.board.copy() , col_pos, color, self.n_player_step[index] , self.n_step , game_won )

        self.step_trace.append( game_step )
        self.n_player_step[index] += 1
        self.n_step += 1

        if game_won :
            self.winner = color

        return valid_move, game_won, self.board 

    def print_ascii(self, console=True):
        print_board = np.zeros( [_n_row, _n_col] , 'U1')

        print_board[ self.board[:,:,_blank_index] == 1] = '_'
        print_board[ self.board[:,:,_red_index] == 1] = 'R'
        print_board[ self.board[:,:,_green_index] == 1] = 'O'

        print_board = np.flip(print_board, 0) 

        lines = []
        for r in print_board:
            line = ( ''.join( r ) )
            lines.append( line )

        print_line = '\n'.join(lines)

        if console:
            print_line = '\n' + print_line

        return print_line


