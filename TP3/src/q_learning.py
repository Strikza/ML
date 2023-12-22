import numpy  as np
import random as rd


# ========================================
def move(action, position, limit):

    if (action == 0):
        return [position[0], max(position[1] - 1, 0)]
    elif (action == 1):
        return [min(position[0] + 1, limit), position[1]]
    elif (action == 2):
        return [position[0], min(position[1] + 1, limit)]
    elif (action == 3):
        return [max(position[0] - 1, 0), position[1]]
    else:
        print("WTF DUDE ?!")



# ========================================
def application_action(action, position):

    pos_move = move(action, position, GRID_SIZE-1)
    r   = -10 if position == pos_move else GRID[pos_move[1], pos_move[0]]
    end = False

    mi = GRID.min()
    ma = GRID.max()

    # La valeur minimal => DRAGON
    # La valeur maximal => ARRIVÉE
    if(r == mi or r == ma):
        pos_move = [0, 0]
        if(r == ma):
            end = True
    
    return pos_move, r, end


# ========================================
def init_q_table(nb_state, nb_action):
    q_table = np.zeros((nb_state, nb_action))
    
    for i in range(nb_state):
        for j in range(nb_action):
            q_table[i, j] = rd.randrange(-10, 10)

    return q_table


# ========================================
def choose_action(state, epsilon, mat_q):
    dart = rd.random()

    if(dart < epsilon):
        return rd.randrange(0, 4)
    else:
        return mat_q[state].argmax()


# ========================================
def onestep(pos, epsilon, mat_q):
    state = pos[0] + pos[1]*GRID_SIZE

    a = choose_action(state, epsilon, mat_q)
    next_pos, r, end = application_action(a, pos)

    q      = mat_q[state, a]
    state2 = next_pos[0] + next_pos[1]*4
    max_q2 = mat_q[state2].max()

    mat_q[state, a] = q + alpha*(r + gamma*max_q2 - q)

    return mat_q, next_pos, end


# ========================================
def start_learn():
    pos   = [0, 0]
    mat_q = init_q_table(GRID_LEN, 4)

    for e in range(epoch):

        epsilon = epoch/(epoch + e)

        for j in range(max_step):
            mat_q, pos, end = onestep(pos, epsilon, mat_q)
            if(end): break

    learned_path = np.zeros(GRID_LEN)

    for i in range(GRID_LEN):
        learned_path[i] = mat_q[i].argmax()

    return learned_path



# ============================ #
#        Hyperparamètre        #
# ============================ #
alpha    = 0.81
gamma    = 0.96
epsilon  = 1
epoch    = 1000
max_step = 100



# ======================================== #
#        Initialisation des données        #
# ======================================== #
reward = np.array([
    [0,  0,  0,  0],
    [-1, 0, -1,  0],
    [0,  0,  0, -1],
    [0, -1,  0,  1]
])

reward2 = np.array([
    [0,   5,  -9,  -9],
    [-10, 5,  -10, -9],
    [-9,  5,   5,  -10],
    [-9, -10,  5,   10]
])



# ======================================== #
#      Apprentissage et Comparaison        #
# ======================================== #
GRID      = reward
GRID_SIZE = 4
GRID_LEN  = 16
l1 = start_learn()

GRID      = reward2
GRID_SIZE = 4
GRID_LEN  = 16
l2 = start_learn()

print(f"Learned path => {l1}")
print(f"Learned path => {l2}")
