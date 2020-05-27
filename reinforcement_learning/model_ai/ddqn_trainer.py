import datetime
import tensorflow as tf
import numpy as np
import logging as l
import os,sys

import game_env.game_env_robot as ger
from .DQN import DQN 
from . import model_eval as mevl
import global_config_reinforcement_learning as gc

# N = 50000
def train(N=1000):
    env = ger.Env()

    gamma = gc.HP_GAMMA
    copy_step = gc.HP_DDQN_TARGET_NETWORK_UPDATE_STEP 
    num_actions = env.action_size

    max_experiences = gc.HP_EXPERIMENT_REPLAY_MAX 
    min_experiences = gc.HP_EXPERIMENT_REPLAY_MIN 
    batch_size = gc.HP_Batch
    lr = gc.HP_LEARNING_RATE

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    log_dir = os.path.join( gc.C_save_model_base_folder, gc.C_save_model_current_folder, 'logs' , current_time)
    summary_writer = tf.summary.create_file_writer(log_dir)

    TrainNet = DQN(num_actions , gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(num_actions , gamma, max_experiences, min_experiences, batch_size, lr)

    total_rewards = np.empty(N)
    epsilon = gc.HP_EPSILON
    decay = gc.HP_EPSILON_DECAY 
    min_epsilon = gc.HP_MIN_EPSILON 

    l.info('preparing to run for N = {}'.format(N))
    tmp_state = env.reset()
    tmp_state = np.atleast_2d(tmp_state)
    TrainNet.model(tmp_state)
    TargetNet.model(tmp_state)
    TrainNet.model.save('saved_model/rein/trainnet')
    TargetNet.model.save('saved_model/rein/targetnet')
    # num_variables = TrainNet.model.trainable_variables
    # num_variables = count_trainable_parameters(num_variables)

    TrainNet.model.summary(print_fn=l.info)

    # l.info('number of model parameters: {}'.format(num_variables))

    for n in range(N):
        # l.info('************************** start : {}'.format(n))

        epsilon = max(min_epsilon, epsilon * decay)
        total_reward, losses = play_game(env, TrainNet, TargetNet, epsilon, copy_step)
        total_rewards[n] = total_reward
        avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
        with summary_writer.as_default():
            tf.summary.scalar('episode reward', total_reward, step=n)
            tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)
            tf.summary.scalar('average loss', losses, step=n)

        if n % 100 == 0:
            # l.info("episode: {}, n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):", avg_rewards,
            #       "episode loss: ", losses)
            l.info("")
            l.info("")
            l.info("")
            l.info("episode: {}".format(n))
            l.info("")
            l.info("")
            l.info("episode: {}, episode reward: {}, epsilon: {}, avg reward (last 100): {} episode loss: {}".format( n,
                       total_reward, epsilon, avg_rewards, losses))
            tf.summary.scalar('epsilon', epsilon, step=n)


        if n % 2000 == 0:  
            l.info('saving...............')
            TrainNet.model.save('saved_model/rein/trainnet')
            TargetNet.model.save('saved_model/rein/targetnet')
            l.info('saving done')

            player_won, robot_won, draw, eval_reward, invalid_move = get_eval_stat()
            with summary_writer.as_default():
                tf.summary.scalar('player win', player_won, step=n)
                tf.summary.scalar('robot win', robot_won, step=n)
                tf.summary.scalar('draw game', draw, step=n)
                tf.summary.scalar('eval reward', eval_reward, step=n)
                tf.summary.scalar('invalid move', invalid_move, step=n)

            l.info('player_won, robot_won, reward, invalid_move : {} , {} , {}, {}'.format( player_won, robot_won, eval_reward, invalid_move ))

        
    l.info("avg reward for last 100 episodes:", avg_rewards)

def get_eval_stat():
    eval_n = 100
    stats = mevl.eval_model_ai(eval_n)
    vv = (stats.sum(axis=0))
    player_won = vv[0]
    robot_won =  vv[1]
    draw = eval_n - player_won - robot_won
    reward = vv[2]
    invalid_move = eval_n - vv[3]

    return player_won, robot_won, draw, reward, invalid_move

def play_game(env, TrainNet, TargetNet, epsilon, copy_step):
    rewards = 0
    iter = 0
    done = False
    observations = env.reset()
    losses = list()
    while not done:
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations
        observations , done , reward, valid_move, player_won, robot_won = env.step( action  )

        rewards += reward

        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp)
        loss = TrainNet.train(TargetNet)
        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())
        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)

    return rewards, np.mean(losses)


def count_trainable_parameters(variables):
    total_parameters = 0
    for variable in variables:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters

    return total_parameters