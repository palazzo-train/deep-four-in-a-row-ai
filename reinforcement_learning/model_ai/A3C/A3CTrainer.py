import tensorflow as tf
import numpy as np
from queue import Queue
import logging as l
import multiprocessing
import os, sys

from .A3C import ActorCriticModel
from .A3CWorker import Worker

import global_config_reinforcement_learning as gconfig
import gym

class MasterAgent():
  def __init__(self):
    self.game_name = 'CartPole-v0'
    save_dir = gconfig.C_save_a3c_worker 
    self.save_dir = save_dir
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    env = gym.make(self.game_name)

    self.state_size = env.observation_space.shape[0]
    self.action_size = env.action_space.n

    # self.opt = tf.train.AdamOptimizer(args.lr, use_locking=True)
    self.opt = tf.keras.optimizers.Adam(learning_rate=)
    l.info(self.state_size, self.action_size)

    self.global_model = ActorCriticModel(self.state_size, self.action_size)  # global network
    self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))

  def train(self):
    res_queue = Queue()

    workers = [Worker(self.state_size,
                      self.action_size,
                      self.global_model,
                      self.opt, res_queue,
                      i, game_name=self.game_name,
                      save_dir=self.save_dir) for i in range(multiprocessing.cpu_count())]

    for i, worker in enumerate(workers):
      l.info("Starting worker {}".format(i))
      worker.start()

    moving_average_rewards = []  # record episode reward to plot
    while True:
      reward = res_queue.get()
      if reward is not None:
        moving_average_rewards.append(reward)
      else:
        break
    [w.join() for w in workers]

    # plt.plot(moving_average_rewards)
    # plt.ylabel('Moving average ep reward')
    # plt.xlabel('Step')
    # plt.savefig(os.path.join(self.save_dir,
    #                          '{} Moving Average.png'.format(self.game_name)))
    # plt.show()

  def play(self):
    env = gym.make(self.game_name).unwrapped
    state, player_color_index = env.reset()
    model = self.global_model
    model_path = os.path.join(self.save_dir, 'model_{}.h5'.format(self.game_name))
    l.info('Loading model from: {}'.format(model_path))
    model.load_weights(model_path)
    done = False
    step_counter = 0
    reward_sum = 0

    try:
      while not done:
        env.render(mode='rgb_array')
        policy, value = model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
        policy = tf.nn.softmax(policy)
        action = np.argmax(policy)
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        l.info("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
        step_counter += 1
    except KeyboardInterrupt:
      l.info("Received Keyboard Interrupt. Shutting down.")
    finally:
      env.close()

def train():
  master = MasterAgent()
  master.train()
