from config import config
from typing import List, Deque, Tuple
from collections import deque
import numpy as np
from keras.layers import Dense, Dropout, Input
from keras.regularizers import l2
from keras import Sequential
from keras.optimizers import Adam
from keras.losses import Huber
import logging 
import tensorflow as tf
from random import sample

class DRLAgent:
    def __init__(self, state_dim: int = config['obs_len'], num_actions: int = config['num_actions'],
                 learning_rate: float = config['learning_rate'], gamma: float = config['gamma'],
                 epsilon_start: float = config['epsilon_start'], epsilon_end: float = config['epsilon_end'],
                 epsilon_decay_steps: int = config['epsilon_decay_steps'], architecture: Tuple[int] = config['architecture'],
                 epsilon_exponential_decay: float = config['epsilon_exponential_decay'],
                 replay_capacity: int = config['replay_capacity'], l2_reg: float = config['l2_reg'],
                 tau: int = config['tau'], batch_size: int = config['batch_size'], train: bool = True):
        self.state_dim: int = state_dim
        self.num_actions: int = num_actions
        self.experience: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque([], maxlen=replay_capacity)
        self.learning_rate: float = learning_rate
        self.gamma: float = gamma
        self.l2_reg: float = l2_reg
        self.architecture = architecture
        self.train = train
        self.online_network = self.build_model(trainable=train)
        self.target_network = self.build_model(trainable=False)
        self.update_target()

        self.epsilon: float = epsilon_start
        self.epsilon_decay_steps: int = epsilon_decay_steps
        self.epsilon_end = epsilon_end
        self.epsilon_decay: float = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.epsilon_exponential_decay: float = epsilon_exponential_decay
        self.epsilon_history: List[float] = []

        self.total_steps: int = 0
        self.train_steps: int = 0
        self.episodes: int = 0
        self.episode_length: int = 0
        self.train_episodes: int = 0
        self.steps_per_episode: List[int] = []
        self.episode_reward: float = 0
        self.rewards_history: List[float] = []

        self.batch_size: int = batch_size
        self.tau: int = tau
        self.losses: List[float] = []
        self.idx: np.ndarray = np.arange(batch_size)

    def build_model(self, trainable: bool = True) -> Sequential:
        # https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc
        # https://wandb.ai/ayush-thakur/dl-question-bank/reports/Input-Keras-Layer-Explanation-With-Code-Samples--VmlldzoyMDIzMDU
        layers = [Input((self.state_dim,))]
        for i, units in enumerate(self.architecture):
            layers.append(Dense(units=units,
                    activation='relu',
                    kernel_regularizer=l2(self.l2_reg),
                    name=f'Dense_{i}',
                    trainable=trainable))
        layers.append(Dropout(.1, seed=config['seed']))
        layers.append(Dense(units=self.num_actions, trainable=trainable, name='Output'))
        model = Sequential(layers)
        model.compile(loss=Huber(), optimizer=Adam(learning_rate=self.learning_rate))

        model.summary()
        return model

    def update_target(self):
        self.target_network.set_weights(self.online_network.get_weights())
        logging.info("Updated target network")
    
    def predict(self, state: list[float]) -> int:
        state_len = len(state)
        q_values = self.online_network(tf.reshape(tf.constant(state), [1,state_len]))
        action = np.argmax(q_values[0])
        assert action.is_integer(), "Action must be integer"
        return action

    def epsilon_greedy_policy(self, state: list[float]) -> int:
        state_len = len(state)
        assert state_len == self.state_dim, "State shape mismatch"
        self.total_steps += 1
        if self.train and np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        return self.predict(state)

    def memorize_transition(self, state: np.ndarray, action: int, reward: float, 
                            state_prime: np.ndarray, not_done: bool = True):
        self.episode_reward += reward
        self.episode_length += 1

        self.experience.append((state, action, reward, state_prime, 1.0 if not_done else .0))
        
    def epsilon_decay_step(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_end) 
        else:
            self.epsilon *= self.epsilon_exponential_decay
        self.episodes += 1
        self.losses = []


    def experience_replay(self):
        # if len(self.experience) < 50 * self.batch_size:
        if len(self.experience) < self.batch_size:
            logging.info("Not enough experiences to replay")
            return
        minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))
        states, actions, rewards, next_states, not_dones = minibatch

        next_q_values = self.online_network.predict_on_batch(next_states)
        best_actions = tf.argmax(next_q_values, axis=1)

        next_q_values_target = self.target_network.predict_on_batch(next_states)
        target_q_values = tf.gather_nd(next_q_values_target,
                                       tf.stack((self.idx, tf.cast(best_actions, tf.int32)), axis=1))
        targets = rewards + not_dones * self.gamma * target_q_values
        q_values = self.online_network.predict_on_batch(states)
        q_values[self.idx, actions] = targets
        loss = self.online_network.fit(x=states, y=q_values, batch_size=config['batch_size'], epochs=1, verbose=0).history['loss'][0]
        self.losses.append(loss)
        if self.total_steps % self.tau == 0:
            self.update_target()

