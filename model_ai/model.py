import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import logging as l

# class MyModel(tf.keras.Model):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
#         self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
#         self.dropout = tf.keras.layers.Dropout(0.5)

#     def call(self, inputs, training=False):
#         x = self.dense1(inputs)
#         if training: 
#             x = self.dropout(x, training=training)
#         return self.dense2(x)

# model = MyModel()

def create_model():
    board_size = 6 * 7 * 3
    color_size = 3
    col_move_size = 7
    # col_moves = np.zeros( [count, 7 ] )

    n_features = board_size + color_size + col_move_size  

    l.info('total: {}  board : {} color : {} color_move : {}'.format( n_features, board_size, color_size, col_move_size))

    inputs = tf.keras.Input(shape=(n_features,), name='input')
    x0 = layers.Lambda( lambda x : x[:,0:board_size] , name='board_input')(inputs)
    x0 = layers.Reshape([6,7,3] , name='board_shape')(x0)

    x1 = layers.Lambda( lambda x : x[:,board_size:]  , name='other_input')(inputs)

    x0 = layers.Conv2D(filters=8,kernel_size=[3,3], activation='relu', name='conv2d_1')(x0)
    x0 = layers.BatchNormalization(name='bn_1')(x0)

    x0 = layers.Conv2D(filters=16,kernel_size=[3,3], activation='relu', name='conv2d_2')(x0)
    x0 = layers.BatchNormalization(name='bn_2')(x0)

    x0 = layers.Flatten(name='flat_board')(x0)
    x0 = layers.Dense( 128,  activation='relu', name='board_encoder' )(x0)

    x = layers.concatenate( [ x0, x1 ] , name='combin_input' )

    x = layers.Dense( 256 , activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.4 , name='dropout_1')(x)

    x = layers.Dense( 16 , activation='relu', name='dense_2')(x)
    x = layers.Dropout(0.4 , name='dropout_2')(x)

    out = layers.Dense( 1 , activation='linear', name='dense_out')(x)

    model = tf.keras.Model(inputs=inputs, outputs=out, name="four-in-a-row")

    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[ 'mse'] )


    return model
