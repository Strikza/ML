import numpy as np
import math
import random as rd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

# === Hyperparamètre === #
alpha = 0.5


# ========================================
def display_plot(w, t, c=None):
    a = -(w[1]/w[2])
    b = -(w[0]/w[2])
    z = [a*t[0] + b, a*t[1] + b]

    if(c):
        plt.plot(t, z, color=c)
    else:
        plt.plot(t, z)


# ========================================
def perceptron_simple(x, w):

    u = w[0] + w[1]*x[0] + w[2]*x[1]

    return 1/(1+math.exp(-u))


# ========================================
def multiperceptron(x, w1, w2):

    x1 = perceptron_simple(x, w1[0])
    x2 = perceptron_simple(x, w1[1])

    xf = perceptron_simple([x1, x2], w2)

    return xf, x1, x2


# ========================================
def multiperceptron_widrow(x, yd, epoch, batch_size):
    plt.subplot(211) # On utilise le même graphique que les points

    w1 = [
        [rd.random(), rd.random(), rd.random()],
        [rd.random(), rd.random(), rd.random()]
    ]
    w2 = [rd.random(), rd.random(), rd.random()]
    
    w1_batch = [
        [0, 0, 0],
        [0, 0, 0]
    ]
    w2_batch = [0, 0, 0]

    error = []

    for e in range(epoch):
        err = 0

        for i in range(len(x[0])):
            x1 = x[0][i]
            x2 = x[1][i]
            y, y1, y2  = multiperceptron([x1, x2], w1, w2)

            L   = -(yd[i] - y)*(y - y*y)
            L1  = w2[1]*L*(y1 - y1*y1)
            L2  = w2[2]*L*(y2 - y2*y2)

            w2_batch[0] += alpha * L
            w2_batch[1] += alpha * L * y1
            w2_batch[2] += alpha * L * y2

            w1_batch[0][0] += alpha * L1
            w1_batch[0][1] += alpha * L1 * x1
            w1_batch[0][2] += alpha * L1 * x2

            w1_batch[1][0] += alpha * L2
            w1_batch[1][1] += alpha * L2 * x1
            w1_batch[1][2] += alpha * L2 * x2
            

            if(i%batch_size == 0 or 
               i == len(x[0])-1
               ):
                w1[0][0] = w1[0][0] - w1_batch[0][0]
                w1[0][1] = w1[0][1] - w1_batch[0][1]
                w1[0][2] = w1[0][2] - w1_batch[0][2]

                w1[1][0] = w1[1][0] - w1_batch[1][0]
                w1[1][1] = w1[1][1] - w1_batch[1][1]
                w1[1][2] = w1[1][2] - w1_batch[1][2]

                w2[0] = w2[0] - w2_batch[0]
                w2[1] = w2[1] - w2_batch[1]
                w2[2] = w2[2] - w2_batch[2]


                w1_batch = [
                    [0, 0, 0],
                    [0, 0, 0]
                ]
                w2_batch = [0, 0, 0]

            err += (yd[i] - y)*(yd[i] - y)

        error.append(err)

        t = [-5, 5]
        c = [0.8, 0.2, 0, e/epoch]
        display_plot(w1[0], t, c)
        c = [0, 0.8, 0.2, e/epoch]
        display_plot(w1[1], t, c)

        if(round(err, 3) == 0.000):
            return w1, w2, error

    return w1, w2, error


# ======================================== #
#        Initialisation des données        #
# ======================================== #
X = [
    [0, 1, 0, 1],
    [0, 0, 1, 1]
]
yd = [0, 1, 1, 0]
y_pred = []


# ======================================== #
#      Apprentissage et Comparaison        #
# ======================================== #
w1, w2, error = multiperceptron_widrow(X, yd, 3500, 1)

for i in range(len(X[0])):
    x = [X[0][i], X[1][i]]
    y = round(multiperceptron(x, w1, w2)[0])
    y_pred.append(y)

print(y_pred)

rate = accuracy_score(yd, y_pred)
print("Accuracy: " + str(rate * 100) + "%")


# ======================================== #
#          Affichage des points            #
# ======================================== #
plt.subplot(211)
plt.axis([-0.5, 1.5, -0.5, 1.5])

for i in range(len(X[0])):
    plt.plot(X[0][i], X[1][i], 'o', c=[0, 0, 0])
    plt.text(X[0][i]+.01, X[1][i]+.01, str(yd[i]))


# ======================================== #
#     Affichage de la courbe d'erreur      #
# ======================================== #
t = [-5, 5]
c = [0, 0, 0]
display_plot(w1[0], t, c)
display_plot(w1[1], t, c)

plt.subplot(212)
plt.plot(range(len(error)), error)
plt.axis([0, len(error)-1, 0, np.array(error).max()])

plt.show()
