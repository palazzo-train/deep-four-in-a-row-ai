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

  def __init__(self, model, summary_writer, log_dir):
    # `gamma` is the discount factor; coefficients are used for the loss terms.
    self.gamma = gc.C_a2c_gamma
    self.value_c = gc.C_a2c_value_coeff
    self.entropy_c = gc.C_a2c_entropy_coeff

    self.model = model
    self.model.compile(
      optimizer=ko.RMSprop(clipvalue=gc.C_a2c_clip_value  ,lr=gc.C_a2c_learning_rate),
      # optimizer=ko.RMSprop(lr=gc.C_a2c_learning_rate),
      # Define separate losses for policy logits and value estimate.
      loss=[self._logits_loss, self._value_loss])

    self.summary_writer = summary_writer
    self.summary_action_writers = []
    for i in range(NUM_COL):
      self.summary_action_writers.append( tf.summary.create_file_writer(os.path.join(log_dir, 'action/action_{}'.format(i)) ))

    self.summary_game_results = []
    self.summary_game_results.append( tf.summary.create_file_writer(os.path.join(log_dir, 'game/win' ) ) )
    self.summary_game_results.append( tf.summary.create_file_writer(os.path.join(log_dir, 'game/draw' ) ) )
    self.summary_game_results.append( tf.summary.create_file_writer(os.path.join(log_dir, 'game/loss' ) ) )

    self.summary_game_moves = []
    self.summary_game_moves.append( tf.summary.create_file_writer(os.path.join(log_dir, 'game/min_move' ) ) )
    self.summary_game_moves.append( tf.summary.create_file_writer(os.path.join(log_dir, 'game/avg_move' ) ) )
    self.summary_game_moves.append( tf.summary.create_file_writer(os.path.join(log_dir, 'game/max_move' ) ) )



  def log_game_results(self, total_episode, game_infos):
    game_sum = game_infos.sum(axis=0)
    mean_step = game_infos[:,3].mean()

    ### win analysis
    num_win = game_sum[0]
    win_game_infos = game_infos[ game_infos[:,0] == 1, :]
    if win_game_infos.shape[0] > 0:
      win_mean_step = win_game_infos[:,3].mean()
      win_min_step = win_game_infos[:,3].min()
      win_max_step = win_game_infos[:,3].max()
    else:
      win_mean_step = 0
      win_min_step = 0
      win_max_step = 0

    ## draw
    num_draw = game_sum[1]
    draw_game_infos = game_infos[ game_infos[:,1] == 1, :]
    if draw_game_infos.shape[0] > 0:
      draw_mean_step = draw_game_infos[:,3].mean()
      draw_min_step = draw_game_infos[:,3].min()
      draw_max_step = draw_game_infos[:,3].max()
    else:
      draw_mean_step = 0
      draw_min_step = 0
      draw_max_step = 0

    ### loss
    num_loss = game_sum[2]
    loss_game_infos = game_infos[ game_infos[:,2] == 1, :]
    if loss_game_infos.shape[0] > 0:
      loss_mean_step = loss_game_infos[:,3].mean()
      loss_min_step = loss_game_infos[:,3].min()
      loss_max_step = loss_game_infos[:,3].max()
    else:
      loss_mean_step = 0
      loss_min_step = 0
      loss_max_step = 0


    with self.summary_writer.as_default():
      tf.summary.scalar('game/win_loss_diff', num_win - num_loss, step=total_episode)

    with self.summary_game_results[0].as_default():
      tf.summary.scalar('game/results', num_win, step=total_episode)
      tf.summary.scalar('game/win_count', num_win, step=total_episode)
    with self.summary_game_results[1].as_default():
      tf.summary.scalar('game/results', num_draw, step=total_episode)
      tf.summary.scalar('game/draw_count', num_draw, step=total_episode)
    with self.summary_game_results[2].as_default():
      tf.summary.scalar('game/results', num_loss, step=total_episode)
      tf.summary.scalar('game/loss_count', num_loss, step=total_episode)

    # min
    with self.summary_game_moves[0].as_default():
      tf.summary.scalar('game/win_game_step', win_min_step, step=total_episode)
      tf.summary.scalar('game/draw_game_step', draw_min_step, step=total_episode)
      tf.summary.scalar('game/loss_game_step', loss_min_step, step=total_episode)

    # mean
    with self.summary_game_moves[1].as_default():
      tf.summary.scalar('game/win_game_step', win_mean_step, step=total_episode)
      tf.summary.scalar('game/draw_game_step', draw_mean_step, step=total_episode)
      tf.summary.scalar('game/loss_game_step', loss_mean_step, step=total_episode)

    # max
    with self.summary_game_moves[2].as_default():
      tf.summary.scalar('game/win_game_step', win_max_step, step=total_episode)
      tf.summary.scalar('game/draw_game_step', draw_max_step, step=total_episode)
      tf.summary.scalar('game/loss_game_step', loss_max_step, step=total_episode)

    return num_win, num_draw, num_loss, mean_step

  def log_training(self, summary_writer, ep_rewards, losses, total_episode, total_update):

    avg_reward = np.mean(ep_rewards)
    avg_losses = losses.mean(axis=0)
    with summary_writer.as_default():
      tf.summary.scalar('game/episode reward', avg_reward, step=total_episode)
      tf.summary.scalar('num update', total_update, step=total_episode)
      tf.summary.scalar('loss/losses 0', avg_losses[0], step=total_episode)
      tf.summary.scalar('loss/losses 1', avg_losses[1], step=total_episode)
      tf.summary.scalar('loss/losses 2', avg_losses[2], step=total_episode)


    return avg_reward, avg_losses

  def log_weight_histo(self, summary_writer, total_episode ):

    with summary_writer.as_default():
      for layer in self.model.layers:
        if layer.trainable:
          for weight in layer.trainable_weights:
            tf.summary.histogram('weights/{}'.format(layer.name), weight, step=total_episode)

  def log_selected_action_histo(self, summary_writer, total_episode, 
                                action_history, game_infos):
    total = action_history.sum()

    mean_step = game_infos[:,3].mean()
    max_step = game_infos[:,3].max()
    min_step = game_infos[:,3].min()

    with self.summary_game_moves[0].as_default():
      tf.summary.scalar('game/move_per_episode', min_step, step=total_episode)
    with self.summary_game_moves[1].as_default():
      tf.summary.scalar('game/move_per_episode', mean_step, step=total_episode)
    with self.summary_game_moves[2].as_default():
      tf.summary.scalar('game/move_per_episode', max_step, step=total_episode)

    for i in range(action_history.shape[0]):
      count = action_history[i]
      rate = count / total 
      with self.summary_action_writers[i].as_default():
        tf.summary.scalar('action/acount_rate', rate , step=total_episode)
        tf.summary.scalar('action/acount_count', count , step=total_episode)
        tf.summary.scalar('action/acount_{}'.format(i), count , step=total_episode)

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
      ep_rewards, losses, action_history, game_infos = self.train_in_group(self.summary_writer, 
                                env, 
                                actions, rewards, dones, values, observations,
                                batch_sz, updates=update_period)

      n_episode = ep_rewards.shape[0]

      total_episode += n_episode
      total_update += update_period

      avg_reward , avg_losses = self.log_training(self.summary_writer, ep_rewards, losses, total_episode, total_update)
      num_win, num_draw, num_loss , mean_step = self.log_game_results(total_episode, game_infos)

      self.log_weight_histo(self.summary_writer, total_episode)
      self.log_selected_action_histo(self.summary_writer, total_episode, action_history, game_infos )

      logging.info('n: {}, episode: {}, upd: {}, reward: {:.2f}, move {:.2f} game (w,d,l): {},{},{} losses: {:.2f},{:.2f},{:.2f}'.format(
                  n, total_episode, total_update, avg_reward, mean_step,  
                  num_win, num_draw, num_loss,
                  avg_losses[0], avg_losses[1], avg_losses[2]))

      if ( n %  gc.C_a2c_save_weight_period ) == 0:
        self.model.save_weights(checkpoint_path)
        logging.info('    saved weight to {}'.format(checkpoint_path))

  def train_in_group(self, summary_writer, env, 
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
