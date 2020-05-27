import numpy as np

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

class HistoryLogger():
    def __init__(self, n_row, n_col, n_color_state):
        ### board + col_pos + color + score

        self.n_row = n_row
        self.n_col = n_col
        self.n_color_state = n_color_state 
        self.n_max_step = n_row * n_col 

        board_size = n_row * n_col * n_color_state 
        self.board_size = board_size

        self.history_step_size = board_size + n_col + n_color_state + 1

        h_size = self.history_step_size 

        max_histo_step = self.n_max_step + 1
        self.step_history = np.zeros( [ max_histo_step , h_size] )

    def reset(self):
        ### board + col_pos + color + score
        h_size = self.history_step_size 
        max_histo_step = self.n_max_step + 1
        self.step_history = np.zeros( [ max_histo_step , h_size] )
        self.score_assigned = False

    def save_board(self, n_step, board):
        self.step_history[n_step,0:self.board_size] = board.reshape(-1)

    def add_history(self, n_step, col_pos_index, color):
        col_pos_onehot = np.zeros( self.n_col )
        col_pos_onehot[col_pos_index] = 1

        x = np.concatenate( [ col_pos_onehot , color ])
        self.step_history[n_step, self.board_size:-1] = x

    def add_game_won(self, n_step, board):
        self.step_history[n_step,0:self.board_size] = board.reshape(-1)


    def get_history(self, n_step, game_won, with_last_step):
        if not self.score_assigned :
            step_count = n_step
            scores = history_to_assign_score_label(game_won, step_count)

            self.step_history[0:n_step,-1] = scores

            self.score_assigned = True

        if with_last_step:
            return self.step_history[0:n_step+1]
        else:
            return self.step_history[0:n_step]
