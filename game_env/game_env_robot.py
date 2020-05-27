import numpy as np
from . import game_env as ge
from .game_env import RED, GREEN , BLANK, GREEN_INDEX , RED_INDEX , BLANK_INDEX 
from . import random_robot_players as rp



def _color_index_to_color(index):
    if index == GREEN_INDEX:
        return GREEN
    elif index == RED_INDEX:
        return RED

class Env():
    def __init__(self):
        self.game = ge.GameEnv()
        self.red_robots = rp.getRobots(RED,GREEN)
        self.green_robots = rp.getRobots(GREEN,RED)

        self.robots_inventry = { 
            GREEN_INDEX : rp.getRobots(GREEN,RED),
            RED_INDEX : rp.getRobots(RED,GREEN) }

        self.reset()

    def _get_state(self):
        b = self.game.board.reshape(-1)
        return np.append(b, [self.player_color_index])

    def _robot_move(self):
        robot_player = self.current_robot

        valid_move = False
        while not valid_move:
            c0 = robot_player.move(self.game)
            valid_move, game_end, won , board = self.game.move( self.robot_color ,c0)
            robot_player.move_valid_feedback(c0, valid_move)

        return game_end, won

    def step(self, action):
        ## player's action
        valid_move, game_end, won , board = self.game.move( self.player_color ,action)

        if not valid_move:
            game_end = True
            return self._get_state() , game_end

        self._robot_move()

        return self._get_state() , game_end

    def reset(self):
        self.game.reset()

        self.player_color_index = int(np.random.choice(2))
        self.robot_color_index = int((self.player_color_index + 1 ) % 2)
        
        self.player_color = _color_index_to_color(self.player_color_index)
        self.robot_color = _color_index_to_color(self.robot_color_index)
        self.first_move_color_index = int(np.random.choice(2))

        robots = self.robots_inventry[self.robot_color_index]
        robot_idx = np.random.choice( len(robots))

        self.current_robot = robots[robot_idx]

        if self.first_move_color_index == self.robot_color_index:
            self._robot_move()

        return self._get_state()




