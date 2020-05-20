import numpy as np
import logging 
import logging as l
import sys, os
import data_generator.game_manager as gm
import model_ai.model as m
import tensorflow as tf

def setupLogging():
    fileName = 'app.log'
    logPath = '.'
    path = os.path.join( logPath , fileName )

    format='%(asctime)s %(levelname)-8s - %(message)s'
    logFormatter = logging.Formatter(format)
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(logging.DEBUG)




def test_gen1():
    gm.loop_games(200)


def test1():
    model = m.create_model()
    model.summary(print_fn=l.info)
    # tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))

    l.info('loading data file')
    with open('data/data2/data.npy', 'rb') as f:
        data = np.load(f)




    x = data[:,0:-1]
    y = data[:,-1]

    l.info('converting to tf dataset')
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(32)
    l.info('dataset shuffle')
    dataset.shuffle(10000)

    l.info('ready to fit')
    history = model.fit(dataset, epochs=1)

    l.info('saving model')
    tf.keras.models.save_model( model, "./model1.h5", overwrite=True, include_optimizer=True, signatures=None, options=None)

    # history = model.fit(x, y, batch_size=64, epochs=1, validation_data=(x_val, y_val))

    l.info('\nhistory dict: {}'.format(history.history))


def main():
    setupLogging()

    l.info('start')
    test1()


if __name__ == "__main__":
    main()
