###  running mode
MODE_RESUME_TRAINING = True 
MODE_ENABLE_MODEL_ROBOT_EVULATION = True 


### environment
C_save_data_folder = 'project_data/data/working'
C_save_model_base_folder = 'project_data/saved_model'
C_save_model_current_folder = 'working'



### Hyperparameters
HP_Batch = 4096 
# HP_Batch = 8
HP_EPOCH = 10 
# HP_EPOCH = 500
HP_NUM_TRAINING_DATA = 5111000
# HP_NUM_TRAINING_DATA = 200

HP_DATA_SHUFFLE_SIZE = 32768
EVAL_ROBOT_EVAL_BY_MODE_GAME_NUM = 100


#### Data
DATA_TRAINING_SET_RATIO = 0.86
DATA_DEV_SET_RATIO = 0.07