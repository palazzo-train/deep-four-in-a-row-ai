import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import os
import datetime
import logging as l
import game_env.game_env as ge


## 
## Typicallly Q function Q( s, a) => qa
##
## This model is Q that simultaneously tests all possible a, 
## hence MyModel input state and output all Q :  f(s) -> qa1, qa2, qa3..... qan
## 
##  state = board state + my color
##
##
##  my color = Green  0 or Red = 1
##


class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()

        board_size = ge.NUM_ROW * ge.NUM_COL * ge.NUM_COLOR_STATE
        color_size = 1
        possible_move = ge.NUM_COL 

        n_features = board_size + color_size 

        l.info('total: {}  board : {} color : {} possible move : {}'.format( n_features, board_size, color_size, possible_move))

        ## input
        self.input_layer = layers.InputLayer(input_shape=(n_features,))

        ## input board
        self.input_board = layers.Lambda( lambda x : x[:,0:board_size] , name='board_input')
        self.board_reshape = layers.Reshape([ge.NUM_ROW ,ge.NUM_COL,ge.NUM_COLOR_STATE] , name='board_shape')

        ## input color
        self.input_my_color = layers.Lambda( lambda x : x[:,board_size:]  , name='other_input')

        ## board encoder
        self.board_encoder = []

        self.board_encoder.append( layers.Conv2D(filters=32,kernel_size=[3,3], activation='relu' , kernel_initializer='random_normal', name='conv2d_1') )
        self.board_encoder.append( layers.BatchNormalization(name='bn_1') )
        self.board_encoder.append( layers.Conv2D(filters=64,kernel_size=[3,3], activation='relu', kernel_initializer='random_normal', name='conv2d_2') )
        self.board_encoder.append( layers.BatchNormalization(name='bn_2') )
        self.board_encoder.append( layers.Flatten(name='flat_board') )
        self.board_encoder.append( layers.Dense( 256,  activation='relu', kernel_initializer= tf.keras.initializers.GlorotNormal() , name='board_encoder' ) )

        ### combine color and board encoder
        self.combine_layer = layers.Concatenate( name='combin_input' )

        ### thinking
        self.actor = []
        self.actor.append( layers.Dense( 512, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal(), name='dense_1') )
        self.actor.append( layers.Dropout(0.5 , name='dropout_1') )
        self.actor.append( layers.Dense( 128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal(), name='dense_2') )
        self.actor.append( layers.Dropout(0.5 , name='dropout_2') )

        ### output logits
        self.out_logits = layers.Dense( possible_move , activation='linear', kernel_initializer='RandomNormal', name='logits_out')


    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)

        z1 = self.input_board(z)
        z1 = self.board_reshape(z1)

        for e_layer in self.board_encoder :
            z1 = e_layer(z1)

        z2 = self.input_my_color(z)

        zz = self.combine_layer( [z1, z2])
        
        for act_layer in self.actor:
            zz = act_layer(zz)

        output = self.out_logits(zz)
        return output