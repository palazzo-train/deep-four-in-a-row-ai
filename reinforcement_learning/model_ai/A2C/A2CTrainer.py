from .A2CAgent import A2CAgent
from .A2CModel import A2CModel
import game_env.game_env_robot as ger


def train():
    env = ger.GymEnv()
    num_actions = env.action_size
    # num_state = env.state_size

    model = A2CModel(num_actions=num_actions)
    agent = A2CAgent(model )

    rewards_history = agent.train(env)

    print("Finished training. Testing...")
    print("Total Episode Reward: %d out of 200" % agent.test(env, args.render_test))