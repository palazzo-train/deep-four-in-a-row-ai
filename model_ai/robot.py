import tensorflow as tf
import numpy as np
import logging as l
from scipy import special

class Robot():
    def __init__(self, mycolor, opponent_color, saved_model_path):
        self.my_color = mycolor
        self.opponent_color = opponent_color
        self.saved_model_path = saved_model_path
        self.name = 'AI Robot'
        self.index = 0

        self.__load_model__()
    
    def __load_model__(self):
        self.model = tf.keras.models.load_model( self.saved_model_path ) 
        self.model.summary(print_fn=l.info)

    def __get_all_moves(self):
        col_size = 7
        a = np.arange(col_size)
        one_hots = np.zeros((col_size, a.max()+1))
        one_hots[np.arange(col_size),a] = 1

        return one_hots

    def move(self, game):
        board = game.board.copy()
        board = board.reshape(-1)

        self.index = ( self.index + 1 ) % 7

        col = self.index

        # print(x)
        # print(x.shape)

        col_pos = self.__get_all_moves()
        boards = np.broadcast_to(board, (7, board.shape[0])) 
        colors = np.broadcast_to(self.my_color, (7, self.my_color.shape[0])) 
        all_moves_x = np.concatenate( [ boards , col_pos , colors ] , axis=1)
        score = self.model.predict(all_moves_x)
        print('robot move {}'.format(score))

        ss = special.softmax(score , axis = 1)
        print('')
        print(ss)

        return col 


        # if self.smart_level >= 1:
        #     ## defensive 
        #     sc = game.test_all_moves(self.opponent_color)

        #     if sc != -1 :
        #         return sc

        # if self.smart_level >= 2:
        #     ## offensive
        #     sc = game.test_all_moves(self.my_color)

        #     if sc != -1 :
        #         return sc

        # col = np.random.choice( 7 , p=self.proba )
        # return col 
