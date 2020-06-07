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
  def __init__(self, model):
    # `gamma` is the discount factor; coefficients are used for the loss terms.
    self.gamma = gc.C_a2c_gamma
    self.value_c = gc.C_a2c_value_coeff
    self.entropy_c = gc.C_a2c_entropy_coeff

    self.model = model
    self.model.compile(
      optimizer=ko.RMSprop(lr=gc.C_a2c_learning_rate),
      # Define separate losses for policy logits and value estimate.
      loss=[self._logits_loss, self._value_loss])

    tmp_input = np.zeros( self.model.input_size )
    self.model.action_value( tmp_input[None,:])
    self.model.summary(print_fn=logging.info)


  def log_training(self, summary_writer, ep_rewards, losses, env, total_episode, total_update):
    num_win = (ep_rewards == env.reward_player_win ).sum()
    num_loss = (ep_rewards == env.reward_player_loss ).sum()
    num_draw = (ep_rewards == env.reward_draw_game).sum()
    avg_reward = np.mean(ep_rewards)
    with summary_writer.as_default():
      tf.summary.scalar('episode reward', avg_reward, step=total_episode)
      tf.summary.scalar('num update', total_update, step=total_episode)
      tf.summary.scalar('game/win', num_win, step=total_episode)
      tf.summary.scalar('game/loss', num_loss, step=total_episode)
      tf.summary.scalar('game/draw', num_draw, step=total_episode)
      tf.summary.scalar('loss/losses 0', losses[0], step=total_episode)
      tf.summary.scalar('loss/losses 1', losses[1], step=total_episode)
      tf.summary.scalar('loss/losses 2', losses[2], step=total_episode)

    return avg_reward

  def train(self, env ):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    log_dir = os.path.join( gc.C_a2c_save_model_base_folder, 'logs' , current_time)
    checkpoint_path = os.path.join( gc.C_a2c_save_model_base_folder, 'checkpoint' , 'checkpoint')
    summary_writer = tf.summary.create_file_writer(log_dir)

    total_update = 0
    update_period = gc.C_a2c_update_period 
    total_episode = 0
    batch_sz = gc.C_a2c_batch_size

    # Storage helpers for a single batch of data.
    actions = np.empty(batch_sz, dtype=np.int32)
    rewards, dones, values = np.empty((3, batch_sz))
    observations = np.empty((batch_sz, env.state_size))

    for n in range(gc.C_a2c_training_size):

      ep_rewards_list , losses = self.train_in_group(summary_writer, env, 
                                actions, rewards, dones, values, observations,
                                 batch_sz, updates=update_period)

      ep_rewards = np.array(ep_rewards_list)
      n_episode = ep_rewards.shape[0]

      total_episode += n_episode
      total_update += update_period
      avg_reward = self.log_training(summary_writer, ep_rewards, losses, env, total_episode, total_update)

      logging.info('n: {}, Total episode: {}, update: {}, group avg reward: {:.2f}, losses : {:.2f},{:.2f},{:.2f}'.format(
                  n, total_episode, total_update, avg_reward, losses[0], losses[1], losses[2]))

      if ( n %  gc.C_a2c_save_weight_period ) == 0:
        self.model.save_weights(checkpoint_path)
        logging.info('    saved weight to {}'.format(checkpoint_path))

  def train_in_group(self, summary_writer, env, 
                      actions, rewards, dones, values, observations,
                      batch_sz, updates):


    # Training loop: collect samples, send to optimizer, repeat updates times.
    ep_rewards = [0.0]
    next_obs = env.reset()

    for update in range(updates):
      for step in range(batch_sz):
        observations[step] = next_obs.copy()
        actions[step], values[step] = self.model.action_value(next_obs[None, :])

        next_obs, reward, done, _ = env.step( actions[step]  )
        # debug_print(actions[step], observations[step], reward, done)

        rewards[step], dones[step] = reward, done

        ep_rewards[-1] += rewards[step]
        if dones[step]:
          ep_rewards.append(0.0)

          next_obs = env.reset()
          # logging.info("Episode: %03d, Reward: %03d" % (len(ep_rewards) - 1, ep_rewards[-2]))

      _, next_value = self.model.action_value(next_obs[None, :])
      returns, advs = self._returns_advantages(rewards, dones, values, next_value)
      # A trick to input actions and advantages through same API.
      acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
      # Performs a full training step on the collected batch.
      # Note: no need to mess around with gradients, Keras API handles it.

      losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
      # logging.info("[%d/%d] Losses: %s" % (update + 1, updates, losses))

    return ep_rewards, losses

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
