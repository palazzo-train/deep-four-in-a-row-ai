import logging
import os
import tensorflow as tf
import datetime
import numpy as np
from .A2CAgent import A2CAgent
from .A2CModel import A2CModel
import game_env.game_env_robot as ger
import global_config_reinforcement_learning as gc


def train():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join( gc.C_a2c_save_model_base_folder, 'logs' , current_time)
    summary_writer = tf.summary.create_file_writer(log_dir)

    env = ger.GymEnv(robot_level=2)
    num_actions = env.action_size
    # num_state = env.state_size
    checkpoint_path = os.path.join( gc.C_a2c_save_model_base_folder, 'checkpoint' , 'checkpoint')

    model = A2CModel(num_actions=num_actions)


    tmp_input = np.zeros( model.input_size )
    model.action_value( tmp_input[None,:])
    model.summary(print_fn=logging.info)


    if gc.C_a2c_resume_training:
        logging.info('')
        logging.info('')
        logging.info('')
        logging.info('Resume training. loading weights')
        logging.info('')
        model.load_weights(checkpoint_path)
    else:
        logging.info('')
        logging.info('')
        logging.info('')
        logging.info('New training')
        logging.info('')

    agent = A2CAgent(model , summary_writer)
    agent.train(env, checkpoint_path)

    logging.info("Finished training. Testing...")
    logging.info("Total Episode Reward: %d out of 200" % agent.test(env))

# def reference_reward():
#     from game_env.game_env import GameEnv, RED_INDEX, GREEN_INDEX
#     from game_env.game_env_robot import GymEnv
#     from game_env.random_robot_players import getRobots
#     import numpy as np



#     ## who use which color 
#     referrence_player_color_index = np.choice( RED_INDEX, GREEN_INDEX, p=[0.5, 0.5])
#     opponent_robots_color_index = GREEN_INDEX if referrence_player_color_index == RED_INDEX else GREEN_INDEX

#     ## pick robot
#     #
#     # player
#     reference_players = getRobots(referrence_player_color_index, opponent_robots_color_index, at_level = 2)
#     reference_robot_idx = np.random.choice( len(reference_players))
#     reference_player = reference_players[reference_robot_idx]
#     #
#     # opponent
#     robot_env = GymEnv()
#     opponent_robots_inventry = robot_env.robots_inventry
#     robots = opponent_robots_inventry[opponent_robots_color_index]
#     robot_idx = np.random.choice( len(robots ))
#     opponent_robot = robots[robot_idx]

#     def color_index_to_color(index):
#         if index == GREEN_INDEX:
#             return GREEN, PLAYER_GREEN_STATE
#         elif index == RED_INDEX:
#             return RED, PLAYER_RED_STATE

#     reference_player_color = color_index_to_color(referrence_player_color_index)
#     opponent_robot_color = color_index_to_color(opponent_robots_color_index)

#     env = GameEnv()
#     env.reset()

#     reference_player.
#             c0 = robot_player.move(self.game)
#     valid_move, game_end, player_won, board = env.move( reference_player_color ,action)




    obs, done, ep_reward = env.reset(), False, 0
    while not done:
      action, _ = self.model.action_value(obs[None, :])
      next_obs, reward, done , _ = env.step( action  )

      ep_reward += reward

    return ep_reward