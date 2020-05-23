import tensorflow as tf
import numpy as np
import logging as l


class ModelRobot():
    def __init__(self, mycolor, opponent_color, model):
        self.my_color = mycolor
        self.opponent_color = opponent_color
        self.name = 'AI Robot'
        self.move_order_list = None
        self.smart_level = 1000
        self.reset()

        self.model = model
    
    def reset(self):
        self.__reset_move()

    def __reset_move(self):
        self.is_new_move = True
        self.next_move_index = 0


    def __get_all_moves(self):
        col_size = 7
        a = np.arange(col_size)
        one_hots = np.zeros((col_size, a.max()+1))
        one_hots[np.arange(col_size),a] = 1

        return one_hots

    def move_valid_feedback(self, prev_move_col, valid_move):
        if valid_move:
            # reset
            self.__reset_move()
        else:
            self.is_new_move = False
            self.next_move_index = self.next_move_index + 1


    def move(self, game):
        
        ### if new move, ask model
        if self.is_new_move: 
            board = game.board.copy()
            board = board.reshape(-1)

            # make 7 predict at 1 time
            col_pos = self.__get_all_moves()
            boards = np.broadcast_to(board, (7, board.shape[0])) 
            colors = np.broadcast_to(self.my_color, (7, self.my_color.shape[0])) 
            all_moves_x = np.concatenate( [ boards , col_pos , colors ] , axis=1)
            score = self.model.predict(all_moves_x)
            
            score = score.reshape(-1)
            # -score for des order
            self.move_order_list = np.argsort(-score)


        ## model was consulted, but the last move is not valid
        ## next move_index will be moved if it is invalid 
        col = self.move_order_list[self.next_move_index]
            
        return col


class Robot(ModelRobot):
    def __init__(self, mycolor, opponent_color, saved_model_path):
        self.my_color = mycolor
        self.opponent_color = opponent_color
        self.saved_model_path = saved_model_path
        self.__load_model__()

        super(Robot, self).__init__(mycolor, opponent_color, self.model)

    def __load_model__(self):
        self.model = tf.keras.models.load_model( self.saved_model_path ) 
        self.model.summary(print_fn=l.info)