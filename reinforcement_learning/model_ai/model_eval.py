import numpy as np
import tensorflow as tf
import game_env.game_env_robot as ger
import game_env.game_env as ge
import reinforcement_learning.model_ai.model as rm 
import logging as l

def color_index_to_name(color_index):
    import game_env.game_env as ge

    if color_index == ge.GREEN_INDEX:
        return 'GREEN'
    elif color_index == ge.RED_INDEX:
        return 'RED'

def play_game(env, ai):
    state = env.reset()

    game_end = False
    while not game_end:
        logits = ai( np.atleast_2d(state) )

        action = np.argmax(logits[0])

        # probs = tf.nn.softmax(logits)
        # action = np.random.choice(env.action_size, p=probs.numpy()[0])

        state, game_end, valid_move, player_won, robot_won = env.step( action  )

    return player_won, robot_won, valid_move

def eval_model_ai(n=100):

    l.info('')
    l.info(' evalulate model ai. games to play : {}'.format(n))
    l.info('')
    env = ger.Env()
    ai = rm.MyModel()

    stats = np.zeros( [ n, 3 ])

    for i in range(n):
        player_won, robot_won, valid_move = play_game(env, ai)

        stats[i,0] = player_won
        stats[i,1] = robot_won
        stats[i,2] = valid_move 

    l.info(' evalulate model ai ends')
    return stats