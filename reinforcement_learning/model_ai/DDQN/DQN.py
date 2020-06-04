import numpy as np
import tensorflow as tf
from .model import MyModel as MyModel

class DQN:
    def __init__(self, num_actions, num_state, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = MyModel()

        self.max_experiences = max_experiences
        self.experiences_count  = 0
        self.experiences_cursor = 0
        self.experience_a_r_done = np.zeros( [ max_experiences , 3 ])

        self.experience_s = np.zeros( [ max_experiences , num_state ])
        self.experience_s2 = np.zeros( [ max_experiences , num_state ])

        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def train(self, TargetNet):
        if self.experiences_count < self.min_experiences:
            return 0

        ids = np.random.randint(low=0, high=self.max_experiences, size=self.batch_size)
        states = np.asarray([self.experience_s[i] for i in ids])
        actions = np.asarray([self.experience_a_r_done[i,0] for i in ids])
        rewards = np.asarray([self.experience_a_r_done[i,1] for i in ids])
        dones = np.asarray([self.experience_a_r_done[i,2] for i in ids])
        states_next= np.asarray([self.experience_s2[i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))

        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states.astype('float32')))[0])

    def add_experience(self, prev_observations, action, reward, observations, done):
        if self.experiences_count < self.min_experiences:
            self.experiences_count += 1

        cursor = self.experiences_cursor
        self.experience_s[cursor] = prev_observations
        self.experience_a_r_done[cursor, 0] = action
        self.experience_a_r_done[cursor, 1] = reward
        self.experience_a_r_done[cursor, 2] = done
        self.experience_s2[cursor] = observations
        self.experiences_cursor = (self.experiences_cursor + 1) % self.max_experiences

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())
