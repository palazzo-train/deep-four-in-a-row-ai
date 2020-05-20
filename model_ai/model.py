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

    x0 = layers.Conv2D(filters=8,kernel_size=[3,3], name='conv2d_1')(x0)
    x0 = layers.Conv2D(filters=16,kernel_size=[3,3], name='conv2d_2')(x0)
    x0 = layers.Flatten(name='flat_board')(x0)
    x0 = layers.Dense( 400, name='board_encoder' )(x0)

    x = layers.concatenate( [ x0, x1 ] , name='combin_input' )

    x = layers.Dense( 32 ,name='dense_1')(x)

    x = layers.Dense( 8 ,name='dense_2')(x)

    out = layers.Dense( 1 ,name='dense_out')(x)

    model = tf.keras.Model(inputs=inputs, outputs=out, name="four-in-a-row")

    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[ 'mse'] )


    return model

def train_model():

    ToCreate = False 

    if ToCreate:
        model = create_model()
    else:
        model = tf.keras.models.load_model( "./save_model/model1.h5")

    model.summary(print_fn=l.info)

    l.info('loading data file')
    with open('data/data2/data.npy', 'rb') as f:
        data = np.load(f)



    n_example = 1200000

    l.info('total size of data {}'.format(data.shape))
    l.info('shuffling')
    x = data[:,0:-1]
    y = data[:,-1]
    np.random.shuffle(data)
    l.info('converting to tf dataset')
    dataset = tf.data.Dataset.from_tensor_slices((x[0:n_example], y[0:n_example]))
    dataset = dataset.batch(32)
    l.info('dataset shuffle')
    dataset.shuffle(4096)

    l.info('ready to fit')
    history = model.fit(dataset, epochs=1)

    l.info('saving model')

    tf.keras.models.save_model( model, "./save_model/model1.h5", overwrite=True, include_optimizer=True, signatures=None, options=None)

    # history = model.fit(x, y, batch_size=64, epochs=1, validation_data=(x_val, y_val))

    l.info('\nhistory dict: {}'.format(history.history))

    return model
