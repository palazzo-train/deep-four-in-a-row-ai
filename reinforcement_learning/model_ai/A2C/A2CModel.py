import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

from game_env.game_env_robot import FEATURE_PLAN_INDEX_VALID_MOVE, NUM_FEATURE_PLAN 
from game_env.game_env import NUM_ROW, NUM_COL


class ProbabilityDistribution(tf.keras.Model):
  def call(self, logits, **kwargs):
    # Sample a random categorical action from the given logits.
    return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class A2CModel(tf.keras.Model):
  def __init__(self, num_actions):
    super().__init__('four_in_a_row_policy')
    # Note: no tf.get_variable(), just simple Keras API!


    self.input_board = kl.Reshape([NUM_ROW ,NUM_COL,NUM_FEATURE_PLAN] , name='board_input')

    self.common_encoder = kl.Conv2D(filters=128,kernel_size=[4,4], 
                            activation='relu' , padding='same',
                            kernel_initializer='random_normal', name='common_conv2d_1') 

    ##### logits network ###############
    self.logits_encoder = [
      kl.Conv2D(filters=128,kernel_size=[3,3], 
                            activation='relu' , padding='same',
                            kernel_initializer='random_normal', name='logits_conv2d_1') ,
      kl.BatchNormalization(name='logits_bn_1') ,
      kl.Conv2D(filters=128,kernel_size=[3,3], 
                            activation='relu' , padding='same',
                            kernel_initializer='random_normal', name='logits_conv2d_2') ,
      kl.BatchNormalization(name='logits_bn_2') ,
      kl.Flatten(name='logits_flat_board') ,
      kl.Dense( 512,  activation='relu', kernel_initializer= tf.keras.initializers.GlorotNormal() , name='logits_encoder' ) 
    ]

    ##### value network ###############
    self.value_encoder = [
      kl.Conv2D(filters=128,kernel_size=[3,3], 
                            activation='relu' , padding='same',
                            kernel_initializer='random_normal', name='value_conv2d_1') ,
      kl.BatchNormalization(name='value_bn_1') ,
      kl.Conv2D(filters=128,kernel_size=[3,3], 
                            activation='relu' , padding='same',
                            kernel_initializer='random_normal', name='value_conv2d_2') ,
      kl.BatchNormalization(name='value_bn_2') ,
      kl.Flatten(name='value_flat_board') ,
      kl.Dense( 512,  activation='relu', kernel_initializer= tf.keras.initializers.GlorotNormal() , name='value_encoder' ) 
    ]

    ### value output
    self.value = kl.Dense(1, name='value')
    # Logits are unnormalized log probabilities.
    self.logits = kl.Dense(num_actions, name='policy_logits')
    self.dist = ProbabilityDistribution()

  def call(self, inputs, **kwargs):
    # Inputs is a numpy array, convert to a tensor.
    x = tf.convert_to_tensor(inputs)
    # Separate hidden layers from the same input tensor.
    x = self.input_board(x)
    x = self.common_encoder(x)

    hidden_logs = x
    for llayer in self.logits_encoder:
      hidden_logs = llayer(hidden_logs)

    hidden_vals = x
    for llayer in self.value_encoder:
      hidden_vals= llayer(hidden_vals)

    return self.logits(hidden_logs), self.value(hidden_vals)

  def action_value(self, obs):
    # Executes `call()` under the hood.
    feature_plans = obs.reshape( NUM_ROW , NUM_COL, NUM_FEATURE_PLAN )

    valid_move_plan = feature_plans[:,:,FEATURE_PLAN_INDEX_VALID_MOVE]
    valid_move = valid_move_plan.sum(axis=0).reshape(-1,NUM_COL)

    logits, value = self.predict_on_batch(obs)

    ## remove invalid move
    logits2 = logits.copy()
    logits2[ valid_move == 0 ] = -np.Inf 
    action = self.dist.predict_on_batch(logits2)

    # Another way to sample actions:
    #   action = tf.random.categorical(logits, 1)
    # Will become clearer later why we don't use it.
    return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)















class ReferenceModel(tf.keras.Model):
  def __init__(self, num_actions):
    super().__init__('mlp_policy')
    # Note: no tf.get_variable(), just simple Keras API!
    self.hidden1 = kl.Dense(128, activation='relu')
    self.hidden2 = kl.Dense(128, activation='relu')
    self.value = kl.Dense(1, name='value')
    # Logits are unnormalized log probabilities.
    self.logits = kl.Dense(num_actions, name='policy_logits')
    self.dist = ProbabilityDistribution()

  def call(self, inputs, **kwargs):
    # Inputs is a numpy array, convert to a tensor.
    x = tf.convert_to_tensor(inputs)
    # Separate hidden layers from the same input tensor.
    hidden_logs = self.hidden1(x)
    hidden_vals = self.hidden2(x)
    return self.logits(hidden_logs), self.value(hidden_vals)

  def action_value(self, obs):
    # Executes `call()` under the hood.
    logits, value = self.predict_on_batch(obs)
    action = self.dist.predict_on_batch(logits)
    # Another way to sample actions:
    #   action = tf.random.categorical(logits, 1)
    # Will become clearer later why we don't use it.


    return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)