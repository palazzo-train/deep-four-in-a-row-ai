import os
import numpy as np
import tensorflow as tf
import datetime
import pandas as pd
import logging as l
from . import model as m
import global_config as gc



def get_numpy_data():
    path = os.path.join( gc.C_save_data_folder , 'data.npz' )
    l.info('loading numpy data file {}'.format(path))

    with open(path, 'rb') as f:
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
        dataset = dataset.batch(gc.HP_Batch)
        l.info('dataset shuffle')
        dataset.shuffle( gc.HP_DATA_SHUFFLE_SIZE )

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
    working_folder = gc.C_save_model_current_folder 
    base_folder = gc.C_save_model_base_folder

    run_time  = datetime.datetime.now()

    csv_logger = './{}/{}/training_{}.log'.format(base_folder, working_folder, run_time.strftime( r'%Y%m%d_%H_%M_%S'))
    history_folder= './{}/{}/'.format(base_folder, working_folder)
    checkpoint_path = './{}/{}/checkpoint/checkpoint'.format(base_folder, working_folder)
    save_model_path = './{}/{}/savemodel/my_model'.format(base_folder, working_folder)

    to_create_new = (not gc.MODE_RESUME_TRAINING)
    model = _get_model(save_model_path, create_new=to_create_new)

    if gc.MODE_RESUME_TRAINING:
        model.load_weights(checkpoint_path)

    n_example = gc.HP_NUM_TRAINING_DATA 
    epochs = gc.HP_EPOCH

    l.info('loading dataset . n_example {}'.format(n_example))
    dataset_train , dataset_dev , dataset_test  = get_dataset(n_example)

    l.info('ready to fit. n_example {}'.format(n_example))
    cb = _get_callback(csv_logger, checkpoint_path)

    history = model.fit(dataset_train, epochs=epochs, validation_data=dataset_dev, callbacks=cb)

    l.info('saving model')
    tf.keras.models.save_model( model, save_model_path )
    if gc.MODE_RESUME_TRAINING:
        model.save_weights(checkpoint_path)

    l.info('saving history')
    _save_history(history_folder, history.history, run_time)

    # Evaluate the model on the test data using `evaluate`
    l.info('Evaluate on test data')
    results = model.evaluate(dataset_test)
    l.info('result {}'.format(results))

    return model