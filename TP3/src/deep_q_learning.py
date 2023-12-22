import numpy  as np
import random as rd
import tensorflow as tf

import matplotlib.pyplot as plt 


# ========================================
def init_model():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape = [GRID_LEN]),
        tf.keras.layers.Dense(4)
    ])

    return model


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
        print("Oh hello fellow Hacker !")



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
def choose_action(state, epsilon):
    
    dart = rd.random()

    if(dart < epsilon):
        return rd.randrange(0, 4)
    else:
        vec_etat = np.zeros((1, GRID_LEN))
        vec_etat[0, state] = 1
        exit_Q = model.predict(vec_etat, verbose=0)
        return np.argmax(exit_Q)


# ========================================
def onestep(pos, epsilon, epoch):
    state = pos[0] + pos[1]*GRID_SIZE

    a = choose_action(state, epsilon)
    next_pos, r, end = application_action(a, pos)

    next_state = next_pos[0] + next_pos[1]*GRID_SIZE

    vec_etat = np.zeros((1, GRID_LEN))
    vec_etat[0, state] = 1

    vec_etat_next = np.zeros((1, GRID_LEN))
    vec_etat_next[0, next_state] = 1
    next_Q     = target.predict(vec_etat_next, verbose=0)
    next_Q_max = np.max(next_Q)

    with tf.GradientTape() as tape:
        predict = model(vec_etat)
        mask = tf.one_hot(a, 4)
        val_predict = tf.reduce_sum(predict*mask, axis=1)

        Q_target = r + gamma*next_Q_max * (1-end)

        loss = loss_fn(Q_target, val_predict)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    suivi[epoch] += loss.numpy()

    return next_pos, end


# ========================================
def start_learn():
    pos = [0, 0]

    for e in range(epoch):

        print(f"EPOCH - {e+1}")

        epsilon = epoch/(epoch + e)
        pos     = [0, 0]
        
        for j in range(max_step):
            pos, end = onestep(pos, epsilon, e)

            if(j % 10 == 0):
                target.set_weights(model.get_weights())

            if(end): 
                print(f"FINITO [in {j+1} steps]")
                break

    learned_path = np.zeros(GRID_LEN)

    for i in range(GRID_LEN):
        vec_etat = np.zeros((1, GRID_LEN))
        vec_etat[0, i] = 1
        exit_Q = model.predict(vec_etat, verbose=0)
        learned_path[i] = np.argmax(exit_Q)

    return learned_path



# ============================ #
#        Hyperparamètre        #
# ============================ #
gamma     = 0.999
epoch     = 1000
max_step  = 100
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
# optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-3)
suivi     = np.zeros(epoch)
loss_fn   = tf.keras.losses.mean_squared_error



# ======================================== #
#        Initialisation des données        #
# ======================================== #
reward = np.array([
    [  0,   0,  -1,    -1],
    [-20,   0, -20,    -1],
    [ -1,   0,   0,   -20],
    [ -1, -20,   0,   100]
])

reward2 = np.array([
    [ 0,  0,  0,  0],
    [-1,  0, -1,  0],
    [ 0,  0,  0, -1],
    [ 0, -1,  0,  1]
])

GRID      = reward
GRID_SIZE = 4
GRID_LEN  = 16

model   = init_model()
target  = tf.keras.models.clone_model(model)
target.set_weights(model.get_weights())



# ======================================== #
#               Apprentissage              #
# ======================================== #
l = start_learn()
print(f"Learned path => {l}")



# ======================================== #
#               Test de jeu                #
# ======================================== #
end      = False
pos      = [0, 0]
count    = 0
max_step = 20
print("START PLAYING")
while (not(end) and count < max_step):
    vec_etat = np.zeros((1, GRID_LEN))
    vec_etat[0, pos[0] + pos[1]*GRID_SIZE] = 1
    Sortie_Q = model.predict(vec_etat, verbose=0)
    a = np.argmax(Sortie_Q)
    print(f"{pos} => {a}")

    next_pos, r, end = application_action(a, pos)
    pos = next_pos
    count += 1
print("FINISHED")



# ======================================== #
#          Affichage des courbes           #
# ======================================== #
plt.title("Line graph - Optimizer SGD") 
plt.xlabel("X axis") 
plt.ylabel("Y axis") 
plt.plot(suivi, color ="red") 
plt.show()
