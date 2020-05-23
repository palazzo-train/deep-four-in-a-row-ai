import numpy as np
import game_env.game_env as ge


class RobotRandom():
    def __init__(self,p):
        self.proba = p[0:ge.NUM_COL].copy()
        self.org_proba = self.proba  / self.proba.sum()
        self.reset()
    
    def reset(self):
        self.reset_move()

    def reset_move(self):
        self.last_invalid_move = np.zeros( ge.NUM_COL)
        self.proba = self.org_proba.copy()


    def move(self, game):
        col = np.random.choice( ge.NUM_COL , p=self.proba )
        return col 
    
    def move_valid_feedback(self, prev_move_col, valid_move):
        if valid_move:
            self.reset_move()
        else:
            self.proba[prev_move_col] = 0
            self.proba = self.proba  / self.proba.sum()
            self.last_invalid_move[prev_move_col] = 1

class RobotSmartRandom(RobotRandom):
    def __init__(self,p,mycolor, opponent_color, smart_level):
        super(RobotSmartRandom, self).__init__(p)

        self.my_color = mycolor
        self.opponent_color = opponent_color
        self.smart_level = smart_level
        self.name = 'level {}'.format(smart_level)

    def move(self, game):
        if self.smart_level >= 1:
            ## defensive 
            sc = game.test_all_moves(self.opponent_color)

            if sc != -1 :
                return sc

        if self.smart_level >= 2:
            ## offensive
            sc = game.test_all_moves(self.my_color)

            if sc != -1 :
                return sc

        col = np.random.choice( ge.NUM_COL , p=self.proba )
        return col 


def getRobots(mycolor, opponent_color, at_level = -1):
    ps = []
    robots = []
    p = np.array( [ 1 , 1, 1, 1, 1  ,1 , 1 ] )
    ps.append( p )
    p = np.array( [ 1 , 1, 1, 1, 1  ,1 , 1 ] )
    ps.append( p )
    p = np.array( [ 1 , 1, 1, 1, 3  ,3 , 3 ] )
    ps.append( p )
    p = np.array( [ 7.1 , 5.9, 4, 4, 4  ,5.9 , 7.1 ] )
    ps.append( p )
    p = np.array( [ 3 , 3, 3, 1, 1  ,1 , 1 ] )
    ps.append( p )
    p = np.array( [ 3 , 3, 5, 5, 5  ,3 , 3 ] )
    ps.append( p )

    if at_level == -1 :
        levels =[ 0 , 1 , 1 , 2 , 2, 2]
    else:
        levels =[ at_level ] 

    for p in ps:
        for level in  levels:
            r = RobotSmartRandom(p, mycolor, opponent_color, level )
            robots.append(r)

    return robots
