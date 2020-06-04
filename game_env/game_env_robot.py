import numpy as np
from . import game_env as ge
from .game_env import RED, GREEN , BLANK, GREEN_INDEX , RED_INDEX , BLANK_INDEX 
from . import random_robot_players as rp


PLAYER_RED_STATE = np.array([0,1])
PLAYER_GREEN_STATE = np.array([1,0])

def _color_index_to_color(index):
    if index == GREEN_INDEX:
        return GREEN, PLAYER_GREEN_STATE
    elif index == RED_INDEX:
        return RED, PLAYER_RED_STATE



class Env():
    action_size = ge.NUM_COL
    state_size = ( ge.NUM_ROW * ge.NUM_COL * ge.NUM_COLOR_STATE ) + 2

    def __init__(self, robot_level=-1):
        self.game = ge.GameEnv()
        self.robot_level = robot_level

        self.robots_inventry = { 
            GREEN_INDEX : rp.getRobots(GREEN,RED, self.robot_level),
            RED_INDEX : rp.getRobots(RED,GREEN, self.robot_level) }

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
            reward = -1
            return self._get_state() , game_end, reward, valid_move, player_won, robot_won

        if game_end: 
            if player_won:
                reward = 1.0
            else:
                ### draw
                reward = 0.5

            return self._get_state() , game_end, reward, valid_move, player_won, robot_won

        game_end, robot_won, valid_move = self._robot_move()

        reward = 0.0
        if game_end and robot_won:
            reward = -0.5

        return self._get_state() , game_end, reward, valid_move, player_won, robot_won

    def reset(self):
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

        return self._get_state(), self.player_color_index





class Env_v2(Env):
  def __init__(self, robot_level=-1):
    super().__init__(robot_level)

  def _get_state(self):
    b = self.game.board
    return b