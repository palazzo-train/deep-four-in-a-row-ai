import datetime
import os
import sys
import logging
import numpy as np
import tensorflow as tf
from game_env.game_env import NUM_COL , NUM_ROW , NUM_COLOR_STATE , NUM_IN_A_ROW 


class A2CTensorboard():

  def __init__(self, model, summary_writer, log_dir):
    self.model = model
    self.summary_writer = summary_writer
    self.summary_action_writers = []

    self.summary_low_quality_loss_writers = []
    for i in range(3,9):
      self.summary_low_quality_loss_writers.append( tf.summary.create_file_writer(os.path.join(log_dir, 'game/low_step_loss_step_{}'.format(i)) ))

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

  def log_latest_trainning_group(self, total_episode, total_update, ep_rewards, losses, game_infos, action_history):
    avg_reward , avg_losses = self.log_training(ep_rewards, losses, total_episode, total_update)
    num_win, num_draw, num_loss , mean_step = self.log_game_results(total_episode, game_infos)

    self.log_weight_histo(total_episode)
    self.log_selected_action_histo(total_episode, action_history, game_infos )
    self.log_low_quality_game_loss(total_episode, game_infos)

    return num_win, num_draw, num_loss, mean_step, avg_reward, avg_losses 


  def log_low_quality_game_loss(self,total_episode, game_infos):
    loss_game_steps = game_infos[ game_infos[:,2] == 1, 3]
    if loss_game_steps.shape[0] > 0:
        for i in range(3,9):
            num_step = (loss_game_steps == i).sum()
            with self.summary_low_quality_loss_writers[i-3].as_default():
                tf.summary.scalar('game/low_step_loss', num_step, step=total_episode)

  def log_weight_histo(self, total_episode ):

    with self.summary_writer.as_default():
      for layer in self.model.layers:
        if layer.trainable:
          for weight in layer.trainable_weights:
            tf.summary.histogram('weights/{}'.format(layer.name), weight, step=total_episode)

  def log_selected_action_histo(self, total_episode, 
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

  def log_training(self, ep_rewards, losses, total_episode, total_update):

    avg_reward = np.mean(ep_rewards)
    avg_losses = losses.mean(axis=0)
    with self.summary_writer.as_default():
      tf.summary.scalar('game/episode reward', avg_reward, step=total_episode)
      tf.summary.scalar('num update', total_update, step=total_episode)
      tf.summary.scalar('loss/losses 0', avg_losses[0], step=total_episode)
      tf.summary.scalar('loss/losses 1', avg_losses[1], step=total_episode)
      tf.summary.scalar('loss/losses 2', avg_losses[2], step=total_episode)


    return avg_reward, avg_losses

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
      tf.summary.histogram('game/win_step_dist', win_game_infos[:,3], step=total_episode)

    with self.summary_game_results[1].as_default():
      tf.summary.scalar('game/results', num_draw, step=total_episode)
      tf.summary.scalar('game/draw_count', num_draw, step=total_episode)
      tf.summary.histogram('game/draw_step_dist', draw_game_infos[:,3], step=total_episode)

    with self.summary_game_results[2].as_default():
      tf.summary.scalar('game/results', num_loss, step=total_episode)
      tf.summary.scalar('game/loss_count', num_loss, step=total_episode)
      tf.summary.histogram('game/loss_step_dist', loss_game_infos[:,3], step=total_episode)

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