import datetime
import os
import sys
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

import game_env.feature_plans as fp
from game_env.game_env import NUM_COL , NUM_ROW , NUM_COLOR_STATE , NUM_IN_A_ROW 
import global_config_reinforcement_learning as gc
from .A2CTensorboard import A2CTensorboard


g_i = 0
def debug_print(action, obs, reward, done):
  global g_i
  print('********************* step {}'.format(g_i))
  g_i += 1
  
  for i in range( 12 ):
    print('plan {}'.format(i))
    bb = (obs.reshape(6,7,-1)[:,:,i])
    bb = np.flip(bb, axis=0)
    print(bb)

  print('  action, reward, done : {} , {}, {}'.format(action, reward, done))
  print('')
  print('')
  print('')

class A2CAgent:

  def __init__(self, model, summary_writer, log_dir):
    # `gamma` is the discount factor; coefficients are used for the loss terms.
    self.gamma = gc.C_a2c_gamma
    self.value_c = gc.C_a2c_value_coeff
    self.entropy_c = gc.C_a2c_entropy_coeff

    self.model = model
    self.model.compile(
      optimizer=ko.RMSprop(clipnorm=gc.C_a2c_clip_norm, clipvalue=gc.C_a2c_clip_value  ,lr=gc.C_a2c_learning_rate),
      # optimizer=ko.RMSprop(lr=gc.C_a2c_learning_rate),
      # Define separate losses for policy logits and value estimate.
      loss=[self._logits_loss, self._value_loss])

    self.summary_logger = A2CTensorboard(model, summary_writer, log_dir)

  def train(self, env , checkpoint_path):

    total_update = 0
    update_period = gc.C_a2c_update_period 
    total_episode = 0
    batch_sz = gc.C_a2c_batch_size

    # Storage helpers for a single batch of data.
    actions = np.empty(batch_sz, dtype=np.int32)
    rewards, dones, values = np.empty((3, batch_sz))
    observations = np.empty((batch_sz, env.state_size))

    for n in range(gc.C_a2c_training_size):
      ep_rewards, losses, action_history, game_infos = self.train_in_group( env, 
                                actions, rewards, dones, values, observations,
                                batch_sz, updates=update_period)

      n_episode = ep_rewards.shape[0]

      total_episode += n_episode
      total_update += update_period

      num_win, num_draw, num_loss, mean_step, avg_reward, avg_losses = self.summary_logger.log_latest_trainning_group(total_episode, 
          total_update, ep_rewards, losses, game_infos, action_history)

      logging.info('n: {}, episode: {}, upd: {}, reward: {:.2f}, move {:.2f} game (w,d,l): {},{},{} losses: {:.2f},{:.2f},{:.2f}'.format(
                  n, total_episode, total_update, avg_reward, mean_step,  
                  num_win, num_draw, num_loss,
                  avg_losses[0], avg_losses[1], avg_losses[2]))

      if ( n %  gc.C_a2c_save_weight_period ) == 0:
        self.model.save_weights(checkpoint_path)
        logging.info('    saved weight to {}'.format(checkpoint_path))

  def train_in_group(self, env, 
                      actions, rewards, dones, values, observations,
                      batch_sz, updates):


    # Training loop: collect samples, send to optimizer, repeat updates times.
    ep_rewards = [0.0]
    next_obs = env.reset()
    list_losses = []


    game_infos = []

    action_history = np.zeros( NUM_COL )

    for update in range(updates):
      for step in range(batch_sz):
        observations[step] = next_obs.copy()
        actions[step], values[step] = self.model.action_value(next_obs[None, :])

        action_history[ actions[step]] += 1

        next_obs, reward, done, info = env.step( actions[step]  )
        # debug_print(actions[step], observations[step], reward, done)

        rewards[step], dones[step] = reward, done

        ep_rewards[-1] += rewards[step]
        if dones[step]:
          step_used = info['step']
          player_won = info['player_won']
          robot_won = info['robot_won']
          player_draw = True if not player_won and not robot_won else False
          invalid_move = not info['valid_move']

          game_infos.append( [ player_won, player_draw, robot_won, step_used, invalid_move ] )
          ep_rewards.append(0.0)

          next_obs = env.reset()
          # logging.info("Episode: %03d, Reward: %03d" % (len(ep_rewards) - 1, ep_rewards[-2]))

      _, next_value = self.model.action_value(next_obs[None, :])
      returns, advs = self._returns_advantages(rewards, dones, values, next_value)
      # A trick to input actions and advantages through same API.
      acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
      # Performs a full training step on the collected batch.
      # Note: no need to mess around with gradients, Keras API handles it.

      batch_losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
      list_losses.append( batch_losses )
      # logging.info("[%d/%d] Losses: %s" % (update + 1, updates, losses))

    losses = np.array(list_losses)
    game_infos = np.array(game_infos)
    ep_rewards = np.array(ep_rewards)

    return ep_rewards, losses,  action_history , game_infos

  def test(self, env):
    obs, done, ep_reward = env.reset(), False, 0
    while not done:
      action, _ = self.model.action_value(obs[None, :])
      next_obs, reward, done , _ = env.step( action  )

      ep_reward += reward
    return ep_reward

  def _returns_advantages(self, rewards, dones, values, next_value):
    # `next_value` is the bootstrap value estimate of the future state (critic).
    returns = np.append(np.zeros_like(rewards), next_value, axis=-1)

    # Returns are calculated as discounted sum of future rewards.
    for t in reversed(range(rewards.shape[0])):
      returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
    returns = returns[:-1]
    # Advantages are equal to returns - baseline (value estimates in our case).
    advantages = returns - values
    return returns, advantages

  def _value_loss(self, returns, value):
    # Value loss is typically MSE between value estimates and returns.
    return self.value_c * kls.mean_squared_error(returns, value)

  @tf.function
  def _logits_loss(self, actions_and_advantages, logits):
    # A trick to input actions and advantages through the same API.
    actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
    # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
    # `from_logits` argument ensures transformation into normalized probabilities.
    weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
    # Policy loss is defined by policy gradients, weighted by advantages.
    # Note: we only calculate the loss on the actions we've actually taken.
    actions = tf.cast(actions, tf.int32)

    ### NOT SURE, but to ignore the case where advantages equal negative, which causing neg loss
    # adv2 = tf.where( advantages > 0 , advantages, 0)

    policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
    # policy_loss = weighted_sparse_ce(actions, logits, sample_weight=adv2)
    # Entropy loss can be calculated as cross-entropy over itself.
    probs = tf.nn.softmax(logits)
    entropy_loss = kls.categorical_crossentropy(probs, probs)
    # We want to minimize policy and maximize entropy losses.
    # Here signs are flipped because the optimizer minimizes.

    policy_loss2 = tf.where( policy_loss > 0 , policy_loss , tf.math.maximum( policy_loss , -10 ))

    # return policy_loss - self.entropy_c * entropy_loss
    return policy_loss2 - self.entropy_c * entropy_loss
