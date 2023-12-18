import numpy  as np
import random as rd


# ============================ #
#        Hyperparamètre        #
# ============================ #
alpha    = 0.81
gamma    = 0.96
epsilon  = 1
epoch    = 1000
max_step = 100



# ========================================
def move(action, position):
    if(action == 0):
        position[1] -= 1
    elif(action == 1):
        position[0] += 1
    elif(action == 2):
        position[1] += 1
    elif(action == 3):
        position[0] -= 1
    else:
        print("DUDE ? WTF DUDE ?!")
    
    return position 


# ========================================
def clamped_pos(position, limit):
    position[0] = max(0, position[0])
    position[0] = min(position[0], limit)
    position[1] = max(0, position[1])
    position[1] = min(position[1], limit)

    return position


# ========================================
def application_action(action, position, space):
    pos_move = clamped_pos(move(action, position), 3)
    r = space[pos_move[1], pos_move[0]]
    end = False

    mi = space.min()
    ma = space.max()

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
def onestep(state, epsilon, mat_q):
    a = choose_action(state, epsilon, mat_q)
    pos, r, end = application_action(a, [state%4, int(state/4)], GRID)

    q      = mat_q[state, a]
    state2 = pos[0] + pos[1]*4
    max_q2 = mat_q[state2].max()

    mat_q[state, a] = q + alpha*(r + gamma*max_q2 - q)

    return mat_q, state2, end


# ========================================
def start_learn():
    state = 0
    epsilon = 1
    mat_q = init_q_table(16, 4)

    for i in range(epoch):

        epsilon = epoch/(epoch + i)
        for j in range(max_step):
            mat_q, state, end = onestep(state, epsilon, mat_q)
            if(end): break

    learned_path = np.zeros(len(mat_q))

    for i in range(len(learned_path)):
        learned_path[i] = mat_q[i].argmax()

    return learned_path



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
GRID = reward
l1 = start_learn()

GRID = reward2
l2 = start_learn()

print(l1)
print(l2)
