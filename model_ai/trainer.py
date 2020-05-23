import numpy as np
import tensorflow as tf
import datetime
import pandas as pd
import logging as l
from . import model as m

HP_Batch = 256


def get_numpy_data():
    l.info('loading numpy data file')
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

    return data_train , data_dev , data_test , win_stat , move_count_stat , winner_level_stat 

def get_dataset(n_example=120000):
    l.info('loading dataset file')
    data_train , data_dev , data_test , win_stat , move_count_stat , winner_level_stat = get_numpy_data()

    l.info('shuffling...')

    np.random.shuffle(data_train)
    np.random.shuffle(data_dev)
    np.random.shuffle(data_test)

    all_data = []
    for name, d in [ ( 'training set' , data_train) , ( 'dev set' , data_dev ) , ( 'test set' , data_test )] :
        x = d[:,0:-1]
        y = d[:,-1]

        l.info('converting to tf dataset : {}'.format(name))
        dataset = tf.data.Dataset.from_tensor_slices((x[0:n_example], y[0:n_example]))
        dataset = dataset.batch(HP_Batch)
        l.info('dataset shuffle')
        dataset.shuffle(4096)

        all_data.append(dataset)

    dataset_train = all_data[0]
    dataset_dev = all_data[1]
    dataset_test = all_data[2]

    return dataset_train , dataset_dev , dataset_test 



def _get_model(save_model_path, create_new=False):
    if create_new:
        l.info('***** create new model *******')
        model = m.create_model()
    else:
        l.info('************************************')
        l.info('***** resume model *******')
        l.info('***** loading {} *******'.format( save_model_path ) )
        l.info('************************************')
        model = tf.keras.models.load_model( save_model_path ) 

    model.summary(print_fn=l.info)

    return model


def _get_callback(csv_logger, checkpoint_path):
    csv_logger = tf.keras.callbacks.CSVLogger( csv_logger )
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    
    cb = [ csv_logger , cp_callback ]

    return cb


def _save_history(history_folder, history, run_time):

    x = run_time

    history_path = '{}/history_{}.csv'.format(history_folder, x.strftime( r'%Y%m%d_%H_%M_%S'))
    l.info('saving history to {}'.format(history_path))

    df = pd.DataFrame( history)
    df['date'] = x.strftime( r'%Y-%b-%d' )
    df['time'] = x.strftime( r'%H:%M:%S' )
    df.to_csv(history_path)

def train_model():
    working_folder = 'working'
    base_folder = 'save_model'

    run_time  = datetime.datetime.now()

    csv_logger = './{}/{}/training_{}.log'.format(base_folder, working_folder, run_time.strftime( r'%Y%m%d_%H_%M_%S'))
    history_folder= './{}/{}/'.format(base_folder, working_folder)
    checkpoint_path = './{}/{}/checkpoint/checkpoint'.format(base_folder, working_folder)
    save_model_path = './{}/{}/savemodel/my_model'.format(base_folder, working_folder)

    model = _get_model(save_model_path, create_new=False)

    # n_example = 1200000
    n_example = 5111000
    # n_example = 100 
    epochs = 60
    # epochs = 2

    l.info('loading dataset . n_example {}'.format(n_example))
    dataset_train , dataset_dev , dataset_test  = get_dataset(n_example)

    l.info('ready to fit. n_example {}'.format(n_example))
    cb = _get_callback(csv_logger, checkpoint_path)

    history = model.fit(dataset_train, epochs=epochs, validation_data=dataset_dev, callbacks=cb)

    l.info('saving model')
    tf.keras.models.save_model( model, save_model_path )

    # history = model.fit(x, y, batch_size=64, epochs=1, validation_data=(x_val, y_val))
    l.info('saving history')
    _save_history(history_folder, history.history, run_time)

    # Evaluate the model on the test data using `evaluate`
    l.info('Evaluate on test data')
    results = model.evaluate(dataset_test)
    l.info('result {}'.format(results))

    return model