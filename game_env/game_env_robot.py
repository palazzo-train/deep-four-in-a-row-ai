import numpy as np
from . import game_env as ge
from .game_env import RED, GREEN , BLANK, GREEN_INDEX , RED_INDEX , BLANK_INDEX , NUM_IN_A_ROW 
from . import random_robot_players as rp
import game_env.feature_plans as fp


PLAYER_RED_STATE = np.array([0,1])
PLAYER_GREEN_STATE = np.array([1,0])


FEATURE_PLAN_INDEX_VALID_MOVE = 3
NUM_FEATURE_PLAN = 12

def _color_index_to_color(index):
    if index == GREEN_INDEX:
        return GREEN, PLAYER_GREEN_STATE
    elif index == RED_INDEX:
        return RED, PLAYER_RED_STATE



class Env():
    action_size = ge.NUM_COL
    state_size = ( ge.NUM_ROW * ge.NUM_COL * ge.NUM_COLOR_STATE ) + 2

    reward_invalid = -2
    reward_valid_move = 0.01
    reward_draw_game = 0.05

    def prepare_rewards(self):
        ## prepare the curve shape
        ### heavily penalize low quality game loss (loss by few steps moved)
        n = np.arange(1,27)

        win_curve = np.zeros( 27 )
        win_curve[1:] =  ( 4 / (n ** 0.1)) 

        loss_curve = np.zeros( 27 )
        loss_curve[1:] = -( 4 / ((n) ** 1)) 


        n=np.arange(30)

        y_win = np.concatenate(([0,0,0],win_curve)) + n* 0.01 - 2.6
        y_loss = np.concatenate(([0,0],loss_curve,[0])) + n * 0.01 - 0.3

        y_win = np.concatenate(([0,0,0],win_curve)) + n * self.reward_valid_move - 2.6
        y_loss = np.concatenate(([0,0],loss_curve)) + n * self.reward_valid_move

        ### y_win shape = 30, y_loss shape = 30
        ### but we only use y_win [4,21] and y_loss [3,21]
        self.reward_win_game_by_step = y_win
        self.reward_loss_game_by_step = y_loss

    def reward_player_win(self):
        return self.reward_win_game_by_step[self.n_step]

    def reward_player_loss(self):
        return self.reward_loss_game_by_step[self.n_step]


    def __init__(self, robot_level=-1):
        self.game = ge.GameEnv()
        self.robot_level = robot_level

        self.robots_inventry = { 
            GREEN_INDEX : rp.getRobots(GREEN,RED, self.robot_level),
            RED_INDEX : rp.getRobots(RED,GREEN, self.robot_level) }

        self.prepare_rewards()

        self.reset()

    def _get_state(self):
        b = self.game.board.reshape(-1)
        return np.append(b, [self.player_color_state])

    def _robot_move(self):
        robot_player = self.current_robot

        valid_move = False
        while not valid_move:
            c0 = robot_player.move(self.game)
            valid_move, game_end, won , board = self.game.move( self.robot_color ,c0)
            robot_player.move_valid_feedback(c0, valid_move)

        return game_end, won, valid_move


    def step(self, action):
        player_won = False
        robot_won = False

        ## player's action
        valid_move, game_end, player_won, board = self.game.move( self.player_color ,action)

        if not valid_move:
            game_end = True
            reward = self.reward_invalid 
            return self.step_return_value(game_end, reward, valid_move, player_won, robot_won)

        self.n_step += 1

        if game_end: 
            if player_won:
                reward = self.reward_player_win()
            else:
                ### draw
                reward = self.reward_draw_game

            return self.step_return_value(game_end, reward, valid_move, player_won, robot_won)

        game_end, robot_won, valid_move = self._robot_move()

        reward = self.reward_valid_move
        if game_end and robot_won:
            reward = self.reward_player_loss()

        return self.step_return_value(game_end, reward, valid_move, player_won, robot_won)

    def step_return_value(self, game_end, reward, valid_move , player_won , robot_won ):
        observation = self._get_state()
        done = game_end
        info = { 'valid_move' : valid_move , 
                  'player_won' : player_won ,
                  'robot_won' : robot_won ,
                  'step' : self.n_step
                  }

        return observation, reward, done, info 

    def reset_return_value(self):
        return self._get_state(), self.player_color_index

    def reset(self):
        self.n_step = 0
        self.game.reset()

        self.player_color_index = int(np.random.choice(2))
        self.robot_color_index = int((self.player_color_index + 1 ) % 2)
        
        self.player_color, self.player_color_state  = _color_index_to_color(self.player_color_index)
        self.robot_color , self.robot_color_state = _color_index_to_color(self.robot_color_index)
        self.first_move_color_index = int(np.random.choice(2))

        robots = self.robots_inventry[self.robot_color_index]
        robot_idx = np.random.choice( len(robots))

        self.current_robot = robots[robot_idx]

        if self.first_move_color_index == self.robot_color_index:
            self._robot_move()

        return self.reset_return_value()




def _board_to_feature_plans(game_board, next_row_pos , player_color_index , robot_color_index):
    ### 12 plans
    ### NUM_FEATURE_PLAN = 12

    player_board = game_board[:,:,player_color_index]
    robot_board = game_board[:,:,robot_color_index]
    blank_board = game_board[:,:,BLANK_INDEX]

    all_one = np.ones( [ ge.NUM_ROW, ge.NUM_COL])
    all_zero = np.zeros( [ ge.NUM_ROW, ge.NUM_COL])

    next_move = np.zeros( [ ge.NUM_ROW, ge.NUM_COL])
    next_move[ next_row_pos[ next_row_pos < ge.NUM_ROW ] , 
               np.arange(ge.NUM_COL)[ next_row_pos < ge.NUM_ROW ]]  = 1

    ### play board first, then opponent (robot), then blank
    features = [ player_board, robot_board, blank_board , next_move, all_one]

    for index in [player_color_index, robot_color_index]:
        for n_in_row in [ 2, 3 ,4] :
            board = game_board[:,:,index]
            feature = fp.get_feature(board, n_in_row, ge.NUM_ROW, ge.NUM_COL)

            features.append(feature)

    features.append(all_zero)
    features = np.stack(features, axis=-1)

    return features

class GymEnv(Env):
    ### 12 feature plans
    state_size = ( ge.NUM_ROW * ge.NUM_COL ) * NUM_FEATURE_PLAN 

    def __init__(self, robot_level=-1):
        super().__init__(robot_level)

    def _get_state(self):
        features = _board_to_feature_plans(self.game.board, 
                                            self.game.next_row_pos ,
                                            self.player_color_index ,
                                            self.robot_color_index).reshape(-1)

        return features 

    def step_return_value(self, game_end, reward, valid_move , player_won , robot_won ):
        observation = self._get_state()
        done = game_end
        info = { 'valid_move' : valid_move , 
                  'player_won' : player_won ,
                  'robot_won' : robot_won ,
                  'step' : self.n_step
                  }

        return observation, reward, done, info 

    def reset_return_value(self):
        return self._get_state()

