import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

from game_env.game_env_robot import FEATURE_PLAN_INDEX_VALID_MOVE, NUM_FEATURE_PLAN 
from game_env.game_env import NUM_ROW, NUM_COL
import global_config_reinforcement_learning as gc


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
  def call_arch5(self, inputs, **kwargs):
    x = tf.convert_to_tensor(inputs)
    # Separate hidden layers from the same input tensor.
    x = self.input_board(x)
    x = self.padding(x)
    x = self.padding_bn(x)

    ## common res
    shortcut_x = x 
    for cl in self.common_resnet:
      x = cl(x)

    for cl in self.shortcut_conv:
      shortcut_x = cl(shortcut_x)
    
    x = self.res_add_layer([x,shortcut_x])
    x = self.res_add_bn(x)
    x = self.res_activate(x)
    x = self.common_flatten(x)
    x = self.common_dropout(x)

    ### logits
    hidden_logs = x
    shortcut_hidden_logs = x
    for cl in self.logits_encoders:
      hidden_logs = cl(hidden_logs)

    for cl in self.logits_encoder2:
      shortcut_hidden_logs = cl(shortcut_hidden_logs)

    hidden_logs = self.logits_add_layer([hidden_logs,shortcut_hidden_logs])
    hidden_logs = self.logits_add_bn(hidden_logs)
    hidden_logs = self.logits_activate(hidden_logs)
    hidden_logs = self.logits_bn(hidden_logs)

    ### value
    hidden_vals = x
    shortcut_hidden_vals = x
    for cl in self.value_encoders:
      hidden_vals = cl(hidden_vals)

    for cl in self.value_encoder2:
      shortcut_hidden_vals = cl(shortcut_hidden_vals)

    hidden_vals = self.value_add_layer([hidden_vals,shortcut_hidden_vals])
    hidden_vals = self.value_add_bn(hidden_vals)
    hidden_vals = self.value_activate(hidden_vals)
    hidden_vals = self.value_bn(hidden_vals)

    return self.logits(hidden_logs), self.value(hidden_vals)

  def init_arch5(self, num_actions):
    self.input_board = kl.Reshape([NUM_ROW ,NUM_COL,NUM_FEATURE_PLAN] , name='board_input')
    self.padding = kl.ZeroPadding2D((3,3))
    self.padding_bn = kl.BatchNormalization(name='padding_bn_1')

    self.common_resnet = [
                        kl.Conv2D(filters=64,kernel_size=[1,1],strides=(1,1) ,
                            activation='linear' , padding='valid',
                            kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                            bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                            kernel_initializer=tf.keras.initializers.he_uniform(),
                            name='common_conv2d_1') ,
                        kl.BatchNormalization(name='common_bn_1') ,
                        kl.PReLU(name='common_act_1'),
                        kl.Conv2D(filters=64,kernel_size=[3,3],strides=(1,1),
                            activation='linear' , padding='same',
                            kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                            bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                            kernel_initializer=tf.keras.initializers.he_uniform(),
                            name='common_conv2d_2') ,
                        kl.BatchNormalization(name='common_bn_2') ,
                        kl.PReLU(name='common_act_2'),
                        kl.Conv2D(filters=96,kernel_size=[1,1],strides=(1,1),
                            activation='linear' , padding='valid',
                            kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                            bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                            kernel_initializer=tf.keras.initializers.he_uniform(),
                            name='common_conv2d_3') ,
    ]
    self.shortcut_conv = [ kl.Conv2D(filters=96,kernel_size=[1,1],strides=(1,1) ,
                              activation='linear' , padding='valid',
                              kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                              bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                              kernel_initializer=tf.keras.initializers.he_uniform(),
                              name='shortcut_conv2d_1') ,
    ]
    self.res_add_layer = kl.Add(name='resnet_add')
    self.res_add_bn= kl.BatchNormalization(name='resnet_add_bn_1') 
    self.res_activate= kl.PReLU(name='res_act')
    self.common_flatten= kl.Flatten(name='common_flat')
    self.common_dropout = kl.Dropout( rate = 0.5, name='common_dropout1')

    ##### logits network ###############
    self.logits_encoders = [ kl.Dense( 48,  name='logits_encoder1' ,
                                      bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                                      kernel_initializer=tf.keras.initializers.he_uniform(),
                                      kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ),
                             kl.PReLU(name='logits_act1'),
                             kl.BatchNormalization(name='logits_bn_1') ,
                             kl.Dropout( rate = 0.5, name='logits_dropout1'),
                             kl.Dense( 32,  activation='linear', name='logits_encoder2' ,
                                      kernel_initializer=tf.keras.initializers.he_uniform(),
                                      bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                                      kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ),
                             kl.Dropout( rate = 0.5, name='logits_dropout2'),
    ]
    self.logits_encoder2 = [ kl.Dense( 32,  activation='linear', name='logits_encoder2a' ,
                                      kernel_initializer=tf.keras.initializers.he_uniform(),
                                      bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                                      kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ),
                             kl.Dropout( rate = 0.5, name='logits_dropout2a'),
    ]
    self.logits_add_layer = kl.Add(name='logits_add')
    self.logits_add_bn = kl.BatchNormalization(name='logits_add_bn_1') 
    self.logits_activate= kl.PReLU( name='logits_act')
    self.logits_bn = kl.BatchNormalization(name='logits_bn_4')

    ##### value network ###############
    self.value_encoders = [ kl.Dense( 48, name='value_encoder1' ,
                                      kernel_initializer=tf.keras.initializers.he_uniform(),
                                      bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                                      kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ),
                            kl.PReLU(name='value_act1'),
                            kl.BatchNormalization(name='value_bn_1') ,
                            kl.Dropout( rate = 0.5, name='value_dropout1'),
                            kl.Dense( 32,  activation='linear', name='value_encoder2' ,
                                      kernel_initializer=tf.keras.initializers.he_uniform(),
                                      bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                                      kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ),
                            kl.Dropout( rate = 0.5, name='value_dropout2'),
    ]
    self.value_encoder2 = [ kl.Dense( 32,  activation='linear', name='value_encoder2a' ,
                                      kernel_initializer=tf.keras.initializers.he_uniform(),
                                      bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                                      kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ),
                            kl.Dropout( rate = 0.5, name='logits_dropout2a'),
    ]
    self.value_add_layer = kl.Add(name='value_add')
    self.value_add_bn = kl.BatchNormalization(name='value_add_bn_1') 
    self.value_activate= kl.PReLU(name='value_act')
    self.value_bn = kl.BatchNormalization(name='value_bn_4')

    ### value output
    self.value = kl.Dense(1, activation='linear', name='value_out',
                          kernel_initializer=tf.keras.initializers.he_uniform(),
                          bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                          kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) )
    # Logits are unnormalized log probabilities.
    self.logits = kl.Dense(num_actions, activation='linear', name='policy_logits',
                          kernel_initializer=tf.keras.initializers.he_uniform(),
                          bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                          kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) )
    self.dist = ProbabilityDistribution()

  @tf.function
  def call_arch3(self, inputs, **kwargs):
    x = tf.convert_to_tensor(inputs)
    # Separate hidden layers from the same input tensor.
    x = self.input_board(x)
    x = self.padding(x)
    x = self.padding_bn(x)

    ## common res
    shortcut_x = x 
    for cl in self.common_resnet:
      x = cl(x)

    for cl in self.shortcut_conv:
      shortcut_x = cl(shortcut_x)
    
    x = self.res_add_layer([x,shortcut_x])
    x = self.res_add_bn(x)
    x = self.res_activate(x)
    x = self.common_flatten(x)
    x = self.common_dropout(x)

    ### logits
    hidden_logs = x
    shortcut_hidden_logs = x
    for cl in self.logits_encoders:
      hidden_logs = cl(hidden_logs)

    for cl in self.logits_encoder2:
      shortcut_hidden_logs = cl(shortcut_hidden_logs)

    hidden_logs = self.logits_add_layer([hidden_logs,shortcut_hidden_logs])
    hidden_logs = self.logits_add_bn(hidden_logs)
    hidden_logs = self.logits_activate(hidden_logs)
    hidden_logs = self.logits_bn(hidden_logs)

    ### value
    hidden_vals = x
    shortcut_hidden_vals = x
    for cl in self.value_encoders:
      hidden_vals = cl(hidden_vals)

    for cl in self.value_encoder2:
      shortcut_hidden_vals = cl(shortcut_hidden_vals)

    hidden_vals = self.value_add_layer([hidden_vals,shortcut_hidden_vals])
    hidden_vals = self.value_add_bn(hidden_vals)
    hidden_vals = self.value_activate(hidden_vals)
    hidden_vals = self.value_bn(hidden_vals)

    return self.logits(hidden_logs), self.value(hidden_vals)


  def init_arch3(self, num_actions):
    self.input_board = kl.Reshape([NUM_ROW ,NUM_COL,NUM_FEATURE_PLAN] , name='board_input')
    self.padding = kl.ZeroPadding2D((3,3))
    self.padding_bn = kl.BatchNormalization(name='padding_bn_1')

    self.common_resnet = [
                        kl.Conv2D(filters=48,kernel_size=[1,1],strides=(1,1) ,
                            activation='linear' , padding='valid',
                            kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                            bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                            kernel_initializer='glorot_uniform', name='common_conv2d_1') ,
                        kl.BatchNormalization(name='common_bn_1') ,
                        kl.PReLU(name='common_act_1'),
                        kl.Conv2D(filters=48,kernel_size=[3,3],strides=(1,1),
                            activation='linear' , padding='same',
                            kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                            bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                            kernel_initializer='glorot_uniform', name='common_conv2d_2') ,
                        kl.BatchNormalization(name='common_bn_2') ,
                        kl.PReLU(name='common_act_2'),
                        kl.Conv2D(filters=68,kernel_size=[1,1],strides=(1,1),
                            activation='linear' , padding='valid',
                            kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                            bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                            kernel_initializer='glorot_uniform', name='common_conv2d_3') ,
    ]
    self.shortcut_conv = [ kl.Conv2D(filters=68,kernel_size=[1,1],strides=(1,1) ,
                              activation='linear' , padding='valid',
                              kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                              bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                              kernel_initializer='glorot_uniform', name='shortcut_conv2d_1') ,
    ]
    self.res_add_layer = kl.Add(name='resnet_add')
    self.res_add_bn= kl.BatchNormalization(name='resnet_add_bn_1') 
    self.res_activate= kl.PReLU(name='res_act')
    self.common_flatten= kl.Flatten(name='common_flat')
    self.common_dropout = kl.Dropout( rate = 0.25, name='common_dropout1')

    ##### logits network ###############
    self.logits_encoders = [ kl.Dense( 32,  kernel_initializer='glorot_normal' , name='logits_encoder1' ,
                                      bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                                      kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ),
                             kl.PReLU(name='logits_act1'),
                             kl.BatchNormalization(name='logits_bn_1') ,
                             kl.Dropout( rate = 0.4, name='logits_dropout1'),
                             kl.Dense( 16,  activation='linear', kernel_initializer='glorot_normal' , name='logits_encoder2' ,
                                      bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                                      kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ),
                             kl.Dropout( rate = 0.4, name='logits_dropout2'),
    ]
    self.logits_encoder2 = [ kl.Dense( 16,  activation='linear', kernel_initializer='glorot_normal' , name='logits_encoder2a' ,
                                      bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                                      kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ),
                             kl.Dropout( rate = 0.4, name='logits_dropout2a'),
    ]
    self.logits_add_layer = kl.Add(name='logits_add')
    self.logits_add_bn = kl.BatchNormalization(name='logits_add_bn_1') 
    self.logits_activate= kl.PReLU( name='logits_act')
    self.logits_bn = kl.BatchNormalization(name='logits_bn_4')

    ##### value network ###############
    self.value_encoders = [ kl.Dense( 32,  kernel_initializer='glorot_normal' , name='value_encoder1' ,
                                      bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                                      kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ),
                            kl.PReLU(name='value_act1'),
                            kl.BatchNormalization(name='value_bn_1') ,
                            kl.Dropout( rate = 0.4, name='value_dropout1'),
                            kl.Dense( 16,  activation='linear', kernel_initializer='glorot_normal' , name='value_encoder2' ,
                                      bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                                      kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ),
                            kl.Dropout( rate = 0.4, name='value_dropout2'),
    ]
    self.value_encoder2 = [ kl.Dense( 16,  activation='linear', kernel_initializer='glorot_normal' , name='value_encoder2a' ,
                                      bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                                      kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ),
                            kl.Dropout( rate = 0.4, name='logits_dropout2a'),
    ]
    self.value_add_layer = kl.Add(name='value_add')
    self.value_add_bn = kl.BatchNormalization(name='value_add_bn_1') 
    self.value_activate= kl.PReLU(name='value_act')
    self.value_bn = kl.BatchNormalization(name='value_bn_4')

    ### value output
    self.value = kl.Dense(1, kernel_initializer='glorot_normal', activation='linear', name='value_out',
                          bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                          kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) )
    # Logits are unnormalized log probabilities.
    self.logits = kl.Dense(num_actions, kernel_initializer='glorot_normal' , activation='linear', name='policy_logits',
                          bias_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) ,
                          kernel_regularizer=tf.keras.regularizers.l2(l=gc.C_a2c_regularizer_l2) )
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
    # self.init_arch3(num_actions)
    self.init_arch5(num_actions)

  def call(self, inputs, **kwargs):
    # return self.call_arch2(inputs, **kwargs)
    # return self.call_arch3(inputs, **kwargs)
    return self.call_arch5(inputs, **kwargs)

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