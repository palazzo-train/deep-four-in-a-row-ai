###  running mode
MODE_RESUME_TRAINING = False 
MODE_ENABLE_MODEL_ROBOT_EVULATION = True 


### environment
C_save_data_folder = 'project_data/reinforcement/data/working'
C_save_model_base_folder = 'project_data/reinforcement/saved_model'
C_save_model_current_folder = 'working'



### Hyperparameters
HP_GAMMA = 0.9
HP_Batch = 64 
HP_LEARNING_RATE = 1e-2
HP_EXPERIMENT_REPLAY_MIN = 200
HP_EXPERIMENT_REPLAY_MAX = 2000
HP_DDQN_TARGET_NETWORK_UPDATE_STEP = 1000
HP_EPSILON = 0.99
HP_EPSILON_DECAY = 0.99999
HP_MIN_EPSILON = 0.1
# HP_Batch = 8
HP_EPOCH = 1 
# HP_EPOCH = 500
# HP_NUM_TRAINING_DATA = 5111000
HP_NUM_TRAINING_DATA = 200

HP_DATA_SHUFFLE_SIZE = 32768
EVAL_ROBOT_EVAL_BY_MODE_GAME_NUM = 100


#### Data
DATA_TRAINING_SET_RATIO = 0.86
DATA_DEV_SET_RATIO = 0.07