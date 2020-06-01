import numpy as np
import logging 
import logging as l
import sys, os

from main_supervised import supervised_main as supervised_main

import reinforcement_learning.model_ai.model as AiModel 
import reinforcement_learning.model_ai.DQN as DQN 

import global_config_supervised_learning as gc

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



def test1():
    a = DQN.DQN(5,5,5,5,5,5,5,5)

    # inputs = tf.keras.Input(shape=(2,), name='input')
    # model = tf.keras.Model(inputs=inputs, outputs=m, name="four-in-a-row")

    # variables = m.trainable_variables

    # print(variables)
    # m.compile(optimizer=tf.keras.optimizers.Adam(),
    #             loss=tf.keras.losses.MeanSquaredError(),
    #             metrics=[ 'mse'] )
    # m.summary(print_fn=l.info)


def print_board(state):
    import game_env.game_env as ge

    board = state[0:126].reshape( 6,7,3)

    xx = ge.board_to_ascii(board)
    print(xx)


def test_ai():
    import tensorflow as tf
    import game_env.game_env_robot as ger
    import game_env.game_env as ge
    import reinforcement_learning.model_ai.model as rm 


    m = rm.MyModel()
    env = ger.Env()

    state = env.reset()
    color_index = state[-1]

    ai_color_name = color_index_to_name( color_index )

    print('player color {}'.format( ai_color_name ) )

    print_board(state)

    game_end = False

    while not game_end:
        print(' ------- new step ---------------')
        logits = m( np.atleast_2d(state) )
        probs = tf.nn.softmax(logits)

        action = np.random.choice(env.action_size, p=probs.numpy()[0])

        state, game_end, valid_move, player_won, robot_won = env.step( action  )
        print_board(state)
        print('player {} action {}, game_end {}'.format(ai_color_name, action, game_end ))
        print('player won: {} , robot won: {}, valid move {}'.format(player_won, robot_won, valid_move))
        print('')
        print('')
        print('')


def test_eval_model_ai():
    import reinforcement_learning.model_ai.model_eval as me

    stats = me.eval_model_ai(n=100)
    print('*********************')
    print(stats)
    print(stats.sum(axis=0))

def testtest111():
    import game_env.game_env_robot as ger
    import game_env.game_env as ge

    env = ger.Env()


    state = env.reset()
    print(state.shape)

    def print_board(state):
        board = state[0:126].reshape( 6,7,3)

        xx = ge.board_to_ascii(board)
        print(xx)

    color_index = state[-1]
    if color_index == ge.GREEN_INDEX:
        print('player color GREEN {}'.format(color_index))
    elif color_index == ge.RED_INDEX:
        print('player color RED {}'.format(color_index))

    print_board(state)

    def player_action( index, action ):
        print('step {}  ------------------------------------------------'.format(index))
        state , game_env = env.step( action  )
        print_board(state)
        print('player action {}, game_env {}'.format(action, game_env))

    player_action( 1, 1 )
    player_action( 2, 2 )
    player_action( 3, 5 )
    player_action( 4, 2 )
    player_action( 5, 6 )


def test_train():
    import reinforcement_learning.model_ai.ddqn_trainer as trainer

    trainer.train(5000000)

def reforcement_main():
    # testtest111()
    # test_ai()
    # test_eval_model_ai()
    test_train()



if __name__ == "__main__":
    setupLogging()

    # supervised_main()
    reforcement_main()
