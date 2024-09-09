from datetime import date
import numpy as np

config = {
    'num_actions': None, # Number of actions and output length of the neural network. Calculated below.
    'stop_max': 1.9, # Maximum stop-loss percent
    'stop_min': 0.1, # Minimum stop-loss percent
    'stop_step': 0.2, # Step for stop-loss percent
    'stop_pct': None, # List of stop-loss percents. Generated below based on max, min, step.
    'learning_rate': 0.0001, # Learning rate for Adam optimazer
    'd_start': date(2014, 1, 1), # Dataset start date
    'd_end': date(2024, 1, 1), # Dataset end date
    'slippage_pct': 0.01,
    'gamma': .0, # How much future rewards matter
    'epsilon_start': 1.0, # Starting probability of selecting a random action
    'epsilon_decay_steps': 250, # Number of episodes during which probability of selecting a random action is \
    # lowered by substraction. Decay is started when replay buffer has at least batch size number of experiences.
    'eval_years': 2, # Evaluation time in years
    'epsilon_end': .01, # Goal epsilon value which is reached after 'epsilon_decays_steps' + 'batch_size' number of steps
    'epsilon_exponential_decay': .99, # Epsilon is multiplied by this value every step after 'epsilon_end' step 
    'replay_capacity': 1000*252, # Replay buffer capacity. Used to minimize q-network forgetting problems.
    'trading_days': 252, # Number of days per episode. 252 is number of trading days on the NASDAQ stock exchange during a year.
    'l2_reg': 1e-6, # l2 regulazer parameter for kernel regulazer
    'tau': 100, # Frequency of updating the target neural network. Two networks are used. Target and online networks. \
    # Target network is used for stabilizing training. Target network' weights are updated every 'tau' step.
    'batch_size': 4096, #  Size of a sample batch from replay buffer that is used for training in each iteration 
    # after number of experiences is at least the size of the batch. Used to minimize q-network forgetting problems.
    'max_episodes': 1000, # Numer of max. episodes
    'seed': 42, # Seed for random numbers reporoduction in each run
    'architecture': (256,256), # Each value in the sequnece is a new Dense layer with provided number of units.
    'obs_len': 15, # Input length for the neural network. Do not change this value unless you change observation function in the Backtrader strategy.
}

config['stop_pct'] = np.arange(config['stop_min'], config['stop_max'] + config['stop_step'], config['stop_step'])
config['num_actions'] = len(config['stop_pct'])+1
config['stop_pct']