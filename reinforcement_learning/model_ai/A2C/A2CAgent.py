import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

import game_env.feature_plans as fp
from game_env.game_env import NUM_COL , NUM_ROW , NUM_COLOR_STATE , NUM_IN_A_ROW 



def observation_to_state(obs, player_color_index):
  print('***************')
  print(obs.shape)

  robot_index = int((player_color_index + 1) % 2)

  player_board = obs[:,:,player_color_index]
  robot_board = obs[:,:,robot_index]
  blank_board = obs[:,:,2]

  all_one = np.ones( [ NUM_ROW, NUM_COL])
  all_zero = np.zeros( [ NUM_ROW, NUM_COL])

  ### play board first, then opponent (robot), then blank
  features = [ player_board, robot_board, blank_board , all_one]

  for index in [player_color_index, robot_index]:
    for n_in_row in [ 2, 3 , 4] :
      board = obs[:,:,index]
      feature = fp.get_feature(board, n_in_row, NUM_ROW, NUM_COL)

      features.append(feature)

  features.append(all_zero)
  features = np.stack(features, axis=-1)

  return features

state_shape = 6 * 7



class A2CAgent:
  def __init__(self, model, lr=7e-3, gamma=0.99, value_c=0.5, entropy_c=1e-4):
    # `gamma` is the discount factor; coefficients are used for the loss terms.
    self.gamma = gamma
    self.value_c = value_c
    self.entropy_c = entropy_c

    self.model = model
    self.model.compile(
      optimizer=ko.RMSprop(lr=lr),
      # Define separate losses for policy logits and value estimate.
      loss=[self._logits_loss, self._value_loss])

  def train(self, env, batch_sz=64, updates=250):
    # Storage helpers for a single batch of data.
    actions = np.empty(batch_sz, dtype=np.int32)
    rewards, dones, values = np.empty((3, batch_sz))

    observations = np.empty((batch_sz, state_shape))

    print('****skdsdf')
    print(observations.shape)

    # Training loop: collect samples, send to optimizer, repeat updates times.
    ep_rewards = [0.0]
    next_obs , player_color_index = env.reset()
    for update in range(updates):
      for step in range(batch_sz):
        cur_obs = next_obs.copy()
        observations[step] = observation_to_state(cur_obs, player_color_index)
        actions[step], values[step] = self.model.action_value(next_obs[None, :])

        next_obs, done , reward, valid_move, player_won, robot_won = env.step( actions[step]  )
        rewards[step], dones[step] = reward, done

        ep_rewards[-1] += rewards[step]
        if dones[step]:
          ep_rewards.append(0.0)
          next_obs , player_color_index = env.reset()
          logging.info("Episode: %03d, Reward: %03d" % (len(ep_rewards) - 1, ep_rewards[-2]))

      _, next_value = self.model.action_value(next_obs[None, :])
      returns, advs = self._returns_advantages(rewards, dones, values, next_value)
      # A trick to input actions and advantages through same API.
      acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
      # Performs a full training step on the collected batch.
      # Note: no need to mess around with gradients, Keras API handles it.
      losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
      logging.info("[%d/%d] Losses: %s" % (update + 1, updates, losses))

    return ep_rewards

  def test(self, env, render=False):
    obs, player_color_index, done, ep_reward = env.reset(), False, 0
    while not done:
      action, _ = self.model.action_value(obs[None, :])
      next_obs, done , reward, valid_move, player_won, robot_won = env.step( action  )

      ep_reward += reward
      if render:
        env.render()
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

  def _logits_loss(self, actions_and_advantages, logits):
    # A trick to input actions and advantages through the same API.
    actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
    # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
    # `from_logits` argument ensures transformation into normalized probabilities.
    weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
    # Policy loss is defined by policy gradients, weighted by advantages.
    # Note: we only calculate the loss on the actions we've actually taken.
    actions = tf.cast(actions, tf.int32)
    policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
    # Entropy loss can be calculated as cross-entropy over itself.
    probs = tf.nn.softmax(logits)
    entropy_loss = kls.categorical_crossentropy(probs, probs)
    # We want to minimize policy and maximize entropy losses.
    # Here signs are flipped because the optimizer minimizes.
    return policy_loss - self.entropy_c * entropy_loss