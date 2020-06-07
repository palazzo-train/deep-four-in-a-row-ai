import logging
import os
from .A2CAgent import A2CAgent
from .A2CModel import A2CModel
import game_env.game_env_robot as ger
import global_config_reinforcement_learning as gc


def train():
    env = ger.GymEnv()
    num_actions = env.action_size
    # num_state = env.state_size
    checkpoint_path = os.path.join( gc.C_a2c_save_model_base_folder, 'checkpoint' , 'checkpoint')

    model = A2CModel(num_actions=num_actions)

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

    agent = A2CAgent(model )
    agent.train(env)

    logging.info("Finished training. Testing...")
    logging.info("Total Episode Reward: %d out of 200" % agent.test(env))