import numpy as np 

### constants
_n_row = 6
_n_col = 7

_board_size = _n_row * _n_col * 3
_history_step_size = _board_size + 7 + 3 + 1

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

    ## revert
    board[col_row, col_pos] = BLANK 

    return valid_move, game_won

def board_to_ascii(board, console=True):
    print_board = np.zeros( [_n_row, _n_col] , 'U1')

    print_board[ board[:,:,_blank_index] == 1] = '_'
    print_board[ board[:,:,_red_index] == 1] = 'R'
    print_board[ board[:,:,_green_index] == 1] = 'O'

    print_board = np.flip(print_board, 0) 

    lines = []
    for r in print_board:
        line = ( ''.join( r ) )
        lines.append( line )

    print_line = '\n'.join(lines)

    if console:
        print_line = '\n' + print_line

    return print_line

class GameEnv():

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros( [_n_row, _n_col, 3])
        self.board[:,:] = BLANK
        self.n_step = 0
        self.next_row_pos = np.zeros( _n_col , dtype='int')
        self.win_masks = self.__get_winning_masks()
        self.game_won = False

        self.winner = BLANK
        self.score_assigned = False

        ### board + col_pos + color + score
        h_size = _history_step_size 
        # max_step = _n_row * _n_row + 1
        max_step = 43 
        self.step_history = np.zeros( [ max_step, h_size] )

    def __get_winning_masks(self):
        return _global_winning_masks.copy()

    def __move_exact(self, color , col_pos, col_row):
        self.board[col_row, col_pos] = color

    def test_all_moves(self, color):
        for col_pos in range(_n_col):
            _ , game_won = _m_move_test(self.board, self.next_row_pos , color , col_pos)

            if game_won:
                return col_pos

        return -1

    def test_move(self, color, col_pos):
        valid_move, game_won = _m_move_test(self.board, self.next_row_pos , color , col_pos)

        return valid_move, game_won, self.board 


    def get_history(self,with_last_step=False):

        if not self.score_assigned :
            step_count = self.n_step
            scores = history_to_assign_score_label(self.game_won, step_count)

            self.step_history[0:self.n_step,-1] = scores

            self.score_assigned = True

        if with_last_step:
            return self.step_history[0:self.n_step+1]
        else:
            return self.step_history[0:self.n_step]

    def __add_history(self, col_pos_index, color):
        col_pos_onehot = np.zeros( 7 )
        col_pos_onehot[col_pos_index] = 1

        x = np.concatenate( [ col_pos_onehot , color ])
        self.step_history[self.n_step, _board_size:-1] = x

    def move(self, color , col_pos):
        valid_move, game_won = _m_move_test(self.board, self.next_row_pos , color , col_pos)

        if not valid_move:
            return valid_move, game_won, self.board 

        ### save the before move board in history
        self.step_history[self.n_step,0:_board_size] = self.board.reshape(-1)

        ## commit move
        col_row = self.next_row_pos[col_pos]
        self.board[col_row, col_pos] = color
        self.next_row_pos[col_pos] += 1

        self.__add_history(col_pos, color)

        ## committed
        self.n_step += 1

        if game_won :
            ## add list history
            self.step_history[self.n_step,0:_board_size] = self.board.reshape(-1)
            self.winner = color

        self.game_won = game_won

        return valid_move, game_won, self.board 

    def print_ascii(self, console=True):
        print_line = board_to_ascii(self.board, console)
        
        return print_line


def history_to_assign_score_label(game_won, count):
    winner_start_score = 0.02
    max_score = 1.0
    ### even number
    if count % 2 == 0 : 
        start_index = 0
        start_score = 0
    else:
        start_index = 1
        start_score = winner_start_score 


    ### assign score
    scores = np.arange( start_score, max_score, (max_score)/count)
    # data[:,-1] = scores

    ## if someone won, then someone loss, negative the score for the losser
    ## if no one won, both get positive score
    if game_won :
        # data[start_index::2,-1] = data[start_index::2,-1] * -1
        scores[start_index::2] = scores[start_index::2] * -1

    scores[-1] = max_score 

    return scores 


def show_data_step_ascii(board, col_pos, color , score):
    cc = 'end game'
    if (color == RED).all() :
        cc = 'red'
    elif (color == GREEN).all() :
        cc = 'green'

    col_ascii = [ '1      ',
                    ' 1     ',
                    '  1    ',
                    '   1   ',
                    '    1  ',
                    '     1 ',
                    '      1' ]

    print( board_to_ascii(board) )

    cc_mask = col_pos * np.arange(7)
    col_i = int(np.sum(cc_mask))

    # print( col_ascii[ col_pos * np.arange(7 )).sum() ] )
    print(col_ascii[col_i])
    print( 'col_pos: {}'.format( col_pos ))
    print( 'color : {}  -> {}'.format( cc, color ))
    print( 'score: {}'.format(score))

def print_game_history_ascii( game_history ):
    if game_history.ndim == 1:
        working_step = [ game_history ]
    else:
        working_step = game_history
        
    for step in working_step :
        board = step[0:_board_size].reshape(6,7,3)
        col_pos = step[_board_size:_board_size+7]
        color = step[_board_size+7:_board_size+7+3]
        score = step[-1]
        show_data_step_ascii(board, col_pos, color , score )
            