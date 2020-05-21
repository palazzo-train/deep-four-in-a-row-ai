import numpy as np
import tensorflow as tf
import logging as l
from . import model as m



def get_dataset(n_example=120000):
    l.info('loading data file')
    with open('data/data3/data.npz', 'rb') as f:
        dd = np.load(f)
        data_train = dd['data_train']
        data_dev = dd['data_dev']
        data_test = dd['data_test']
        win_stat = dd['win_stat']
        move_count_stat = dd['move_count_stat']
        winner_level_stat = dd['winner_level_stat']


    l.info('total size of training data {}'.format(data_train.shape))
    l.info('total size of dev data {}'.format(data_dev.shape))
    l.info('total size of test data {}'.format(data_test.shape))
    l.info('total size of win stat data {}'.format(win_stat.shape))
    l.info('total size of move data {}'.format(move_count_stat.shape))
    l.info('total size of player level data {}'.format(winner_level_stat.shape))
    l.info('shuffling...')

    np.random.shuffle(data_train)
    np.random.shuffle(data_dev)
    np.random.shuffle(data_test)

    all_data = []
    for d in [ data_train, data_dev , data_test] :
        x = d[:,0:-1]
        y = d[:,-1]

        l.info('converting to tf dataset')
        dataset = tf.data.Dataset.from_tensor_slices((x[0:n_example], y[0:n_example]))
        dataset = dataset.batch(32)
        l.info('dataset shuffle')
        dataset.shuffle(4096)

        all_data.append(dataset)

    dataset_train = all_data[0]
    dataset_dev = all_data[1]
    dataset_test = all_data[2]

    return dataset_train , dataset_dev , dataset_test 



def _get_model(create_new=False):
    if create_new:
        model = m.create_model()
    else:
        model = tf.keras.models.load_model( "./save_model/model1.h5")

    model.summary(print_fn=l.info)

    return model

def train_model():

    model = _get_model(create_new=True)

    n_example = 120000

    dataset_train , dataset_dev , dataset_test  = get_dataset(n_example)

    l.info('ready to fit')
    csv_logger = tf.keras.callbacks.CSVLogger('./save_model/training.log')

    history = model.fit(dataset_train, epochs=10, validation_data=dataset_dev, callbacks=[csv_logger])

    l.info('saving model')

    tf.keras.models.save_model( model, "./save_model/model1.h5", overwrite=True, include_optimizer=True, signatures=None, options=None)

    # history = model.fit(x, y, batch_size=64, epochs=1, validation_data=(x_val, y_val))
    l.info('\nhistory dict: {}'.format(history.history))

    return model