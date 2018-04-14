import numpy as np
from gym.envs.toy_text.tictactoe import TicTacToeEnv
ttt= TicTacToeEnv()
ttt.state = np.array([1,1,-1,0,0,0,-1,-1,1])
ttt.minimax()

def play():
    i,_=ttt.minimax()
    if i > -1:
        ttt.state[i] = 1
    print('')
    ttt.render()

import string
digs = string.digits + string.ascii_letters


def int2base(x, base):
    if x < 0:
        sign = -1
    elif x == 0:
        return digs[0]
    else:
        sign = 1

    x *= sign
    digits = []

    while x:
        digits.append(int(x % base) - 1)
        x = int(x / base)

    return digits

tictactoe_dict = {}
for i in range(0, 3**9):
    k = np.array(int2base(i,3))
    ttt.state = k
    j,_ = ttt.minimax()
    tictactoe_dict[tuple(k)] = j


    
def play():
    moves = tictactoe_dict[tuple(np.multiply(ttt.state, -1))]
    move = random.choice(moves)
    if move > -1:
        ttt.state[move] = -1
    print('')
    ttt.render()

res_filt = {k:v for (k,v) in res.items() if v[0] == -1}
res


res_bad = {k:v for (k,v) in res.items() if not v[1]}
res_bad_pos = [0] * 9
for p in res_bad.values():
    res_bad_pos[p[0]] += 1

res_bad_pos
[0, 57, 309, 60, 546, 118, 39, 77, 59]
[0, 50, 268, 72, 607, 167, 22, 49, 67]

res_good = {k:v for (k,v) in res.items() if v[1]}
res_good_pos = [0] * 9
for p in res_good.values():
    res_good_pos[p[0]] += 1

res_good_pos


[1, 258, 562, 202, 1198, 254, 195, 202, 383]
[1, 240, 554, 184, 1253, 284, 165, 172, 365]
