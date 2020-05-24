from . import game_env as ge
from .game_env import RED, GREEN , BLANK


class Env():
    def __init__(self):
        self.player_color = BLANK
        self.player_color_index = 0
        self.game_env = ge.GameEnv()
        self.red_robots = rp.getRobots(RED,GREEN)
        self.green_robots = rp.getRobots(GREEN,RED)

    def reset(self):
        self.game_env.reset()
