import numpy  as np
import random as rd
import tensorflow as tf

import matplotlib.pyplot as plt 

# ============================ #
#        Hyperparamètre        #
# ============================ #
gamma     = 0.999
epoch     = 10
max_step  = 100
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001)
loss_fn   = tf.keras.losses.mean_squared_error # Utilisation de la fonction de coût MSE



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
    # La valeur maximal => ARRIVÉ
    if(r == mi or r == ma):
        pos_move = [0, 0]
        end = True
    
    return pos_move, r, end


# ========================================
def init_model(nb_action):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape = [16]),
        tf.keras.layers.Dense(nb_action)
    ])

    return model


# ========================================
def choose_action(state, epsilon, model):
    dart = rd.random()

    if(dart < epsilon):
        return rd.randrange(0, 4)
    else:
        vec_etat = np.zeros((1, 16))
        vec_etat[0, state] = 1
        exit_Q = model.predict(vec_etat, verbose=0)
        return np.argmax(exit_Q[0])


# ========================================
def onestep(state, epsilon, model, target):
    global PLOPLOT

    a = choose_action(state, epsilon, model)
    pos, r, end = application_action(a, [state%4, int(state/4)], GRID)

    # print("POS: " + str([state%4, int(state/4)]))
    # print("DIR: " + str(a))

    state2 = pos[0] + pos[1]*4
    vec_etat_next = np.zeros((1, 16))
    vec_etat_next[0, state2] = 1
    next_Q = target.predict(vec_etat_next, verbose=0)
    next_Q_max = np.max(next_Q[0])

    Q_target = r + gamma * next_Q_max * (1-end)

    vec_etat = np.zeros((1, 16))
    vec_etat[0, state] = 1

    with tf.GradientTape() as tape:
        predict = model(vec_etat) #Ce que l'on pense obtenir

        mask = tf.one_hot(a, 4)
        val_predict = tf.reduce_sum(predict*mask, axis = 1)

        loss = loss_fn(Q_target, val_predict)
        PLOPLOT = np.append(loss, PLOPLOT)
        

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return model, state2, end


# ========================================
def start_learn():
    state = 0
    epsilon = 1
    model = init_model(4)
    target = tf.keras.models.clone_model(model)
    target.set_weights(model.get_weights())

    for i in range(epoch):

        print("EPOCH - " + str(i))

        epsilon = epoch/(epoch + i)
        for j in range(max_step):
            model, state, end = onestep(state, epsilon, model, target)

            if(j % 10 == 0):
                target.set_weights(model.get_weights())

            if(end): break


    learned_path = np.zeros(16)

    for i in range(len(learned_path)):
        vec_etat = np.zeros((1, 16))
        vec_etat[0, i] = 1
        learned_path[i] = model.predict(np.array(vec_etat), verbose=0).argmax()

    return learned_path



# ======================================== #
#        Initialisation des données        #
# ======================================== #
reward = np.array([
    [0,   9,  -9,  -9  ],
    [-10, 9,  -10, -9  ],
    [-9,  9,   9,  -10 ],
    [-9, -10,  9,   100]
])

# reward = np.array([
#     [0,  0,  0,  0],
#     [-1, 0, -1,  0],
#     [0,  0,  0, -1],
#     [0, -1,  0,  1]
# ])

PLOPLOT = np.array([])


# ======================================== #
#              Apprentissage               #
# ======================================== #
GRID = reward
l = start_learn()

print(l)


# ======================================== #
#          Affichage des courbes           #
# ======================================== #
x = range(0, len(PLOPLOT))
y = PLOPLOT

# plotting
plt.title("Line graph") 
plt.xlabel("X axis") 
plt.ylabel("Y axis") 
plt.plot(x, y, color ="red") 
plt.show()
