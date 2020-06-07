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
  input_size = ( NUM_ROW * NUM_COL * NUM_FEATURE_PLAN )

  @tf.function
  def call_arch4(self, inputs, **kwargs):
    x = tf.convert_to_tensor(inputs)
    # Separate hidden layers from the same input tensor.
    # x = self.input_board(x)
    # x = self.common_encoder(x)
    # x = self.common_encoder2(x)
    # x = self.common_encoder3(x)

    hidden_logs = self.logits_encoder(x)
    hidden_vals = self.value_encoder(x)

    return self.logits(hidden_logs), self.value(hidden_vals)

  def init_arch4(self, num_actions):
    # self.input_board = kl.Reshape([NUM_ROW ,NUM_COL,NUM_FEATURE_PLAN] , name='board_input')

    # self.common_encoder = kl.Conv2D(filters=32,kernel_size=[4,4], 
    #                         activation='relu' , padding='same',
    #                         kernel_initializer='glorot_normal', name='common_conv2d_1') 
    # self.common_encoder2 = kl.Conv2D(filters=32,kernel_size=[4,4], 
    #                         activation='relu' , padding='same',
    #                         kernel_initializer='glorot_normal', name='common_conv2d_2') 
    # self.common_encoder3 = kl.Flatten(name='common_flat')

    ##### logits network ###############
    self.logits_encoder = kl.Dense( 32,  activation='relu', kernel_initializer= tf.keras.initializers.GlorotNormal() , name='logits_encoder' ) 

    ##### value network ###############
    self.value_encoder = kl.Dense( 16,  activation='relu', kernel_initializer= tf.keras.initializers.GlorotNormal() , name='value_encoder' ) 

    ### value output
    self.value = kl.Dense(1, kernel_initializer='glorot_normal', activation='linear', name='value')
    # Logits are unnormalized log probabilities.
    self.logits = kl.Dense(num_actions, kernel_initializer='glorot_normal' , activation='linear', name='policy_logits')
    self.dist = ProbabilityDistribution()

  @tf.function
  def call_arch3(self, inputs, **kwargs):
    x = tf.convert_to_tensor(inputs)
    # Separate hidden layers from the same input tensor.
    x = self.input_board(x)

    for cl in self.common_encoder :
      x = cl(x)

    hidden_logs = x
    for cl in self.logits_encoders:
      hidden_logs = cl(hidden_logs)

    hidden_vals = x
    for cl in self.value_encoders:
      hidden_vals = cl(hidden_vals)

    return self.logits(hidden_logs), self.value(hidden_vals)

  def init_arch3(self, num_actions):
    self.input_board = kl.Reshape([NUM_ROW ,NUM_COL,NUM_FEATURE_PLAN] , name='board_input')

    self.common_encoder = [
                        kl.Conv2D(filters=48,kernel_size=[4,4], 
                            activation='relu' , padding='same',
                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01) ,
                            kernel_initializer='glorot_uniform', name='common_conv2d_1') ,
                        kl.BatchNormalization(name='common_bn_1') ,
                        kl.Conv2D(filters=56,kernel_size=[3,3], 
                            activation='relu' , padding='valid',
                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01) ,
                            kernel_initializer='glorot_uniform', name='common_conv2d_2') ,
                        kl.BatchNormalization(name='common_bn_2') ,
                        kl.Conv2D(filters=64,kernel_size=[3,3], 
                            activation='relu' , padding='valid',
                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01) ,
                            kernel_initializer='glorot_uniform', name='common_conv2d_3') ,
                        kl.BatchNormalization(name='common_bn_3') ,
                        kl.Flatten(name='common_flat'),
                        kl.Dropout( rate = 0.25, name='logits_dropout1'),
    ]

    ##### logits network ###############
    self.logits_encoders = [ kl.Dense( 48,  activation='relu', kernel_initializer='glorot_normal' , name='logits_encoder1' ,
                                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01) ),
                             kl.Dropout( rate = 0.4, name='logits_dropout1'),
                             kl.Dense( 32,  activation='relu', kernel_initializer='glorot_normal' , name='logits_encoder2' ,
                                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01) ),
                             kl.Dropout( rate = 0.4, name='logits_dropout2'),
    ]

    ##### value network ###############
    self.value_encoders = [ kl.Dense( 24,  activation='relu', kernel_initializer='glorot_normal' , name='value_encoder1' ,
                                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01) ),
                            kl.Dropout( rate = 0.4, name='value_dropout1'),
                            kl.Dense( 16,  activation='relu', kernel_initializer='glorot_normal' , name='value_encoder2' ,
                                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01) ),
                            kl.Dropout( rate = 0.4, name='value_dropout2'),
    ]

    ### value output
    self.value = kl.Dense(1, kernel_initializer='glorot_normal', activation='linear', name='value_out',
                          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01) )
    # Logits are unnormalized log probabilities.
    self.logits = kl.Dense(num_actions, kernel_initializer='glorot_normal' , activation='linear', name='policy_logits',
                          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01) )
    self.dist = ProbabilityDistribution()

  @tf.function
  def call_arch2(self, inputs, **kwargs):
    x = tf.convert_to_tensor(inputs)
    # Separate hidden layers from the same input tensor.
    x = self.input_board(x)
    x = self.common_encoder(x)
    x = self.common_encoder2(x)

    hidden_logs = self.logits_encoder(x)
    hidden_vals = self.value_encoder(x)

    return self.logits(hidden_logs), self.value(hidden_vals)

  def init_arch2(self, num_actions):
    self.input_board = kl.Reshape([NUM_ROW ,NUM_COL,NUM_FEATURE_PLAN] , name='board_input')

    self.common_encoder = kl.Conv2D(filters=64,kernel_size=[4,4], 
                            activation='relu' , padding='same',
                            kernel_initializer='random_normal', name='common_conv2d_1') 
    self.common_encoder2 = kl.Flatten(name='common_flat')

    ##### logits network ###############
    self.logits_encoder = kl.Dense( 96,  activation='relu', kernel_initializer= tf.keras.initializers.GlorotNormal() , name='logits_encoder' ) 

    ##### value network ###############
    self.value_encoder = kl.Dense( 64,  activation='relu', kernel_initializer= tf.keras.initializers.GlorotNormal() , name='value_encoder' ) 

    ### value output
    self.value = kl.Dense(1, name='value')
    # Logits are unnormalized log probabilities.
    self.logits = kl.Dense(num_actions, name='policy_logits')
    self.dist = ProbabilityDistribution()

  @tf.function
  def call_arch1(self, inputs, **kwargs):
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

  def init_arch1(self, num_actions):
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

  def __init__(self, num_actions):
    super().__init__('four_in_a_row_policy')
    # Note: no tf.get_variable(), just simple Keras API!
    # self.init_arch2(num_actions)
    self.init_arch3(num_actions)

  def call(self, inputs, **kwargs):
    # return self.call_arch2(inputs, **kwargs)
    return self.call_arch3(inputs, **kwargs)

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