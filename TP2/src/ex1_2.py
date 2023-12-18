import numpy as np
import random as rd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score


# === Hyperparamètre === #
alpha = 0.1


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
def myRound(n):
    if(n > 0):
        return 1
    else:
        return -1


# ========================================
def perceptron_simple(x, w, active):

    u = w[0] + w[1]*x[0] + w[2]*x[1]
    
    if(active):
        return np.sign(u)
    else:
        return np.tanh(u)


# ========================================
def apprentissage_widrow(x, yd, epoch, batch_size):
    plt.subplot(211) # On utilise le même graphique que les points

    w     = [rd.random(), rd.random(), rd.random()]
    w1    = 0
    w2    = 0
    w3    = 0
    error = []

    for e in range(epoch):
        err = 0

        for i in range(len(x[0])):
            x1 = x[0][i]
            x2 = x[1][i]
            y  = perceptron_simple([x1, x2], w, False)
            L  = -(yd[i] - y)*(1 - y*y)

            w1 += alpha * L
            w2 += alpha * L * x1
            w3 += alpha * L * x2

            if(i%batch_size == 0 or 
               i == len(x[0])-1
            ):
                w  = [w[0] - w1, w[1] - w2, w[2] - w3]
                w1 = 0
                w2 = 0
                w3 = 0

            err += (yd[i] - y)*(yd[i] - y)

        error.append(err)

        t = [-5, 5]
        c = [0, 0, 0, e/epoch]
        display_plot(w, t, c)

        if(round(err, 3) == 0.000):
            return w, error

    return w, error


# ======================================== #
#        Initialisation des données        #
# ======================================== #
X      = np.loadtxt('./Data/p2_d1.txt')
yd     = 25*[-1] + 25*[1]
y_pred = []


# ======================================== #
#      Apprentissage et Comparaison        #
# ======================================== #
w, error = apprentissage_widrow(X, yd, 10, 12)

for i in range(len(X[0])):
    x = [X[0][i], X[1][i]]
    y = myRound(perceptron_simple(x, w, False))
    y_pred.append(y)

rate = accuracy_score(yd, y_pred)
print("Accuracy: " + str(rate * 100) + "%")


# ======================================== #
#          Affichage des points            #
# ======================================== #
plt.subplot(211)
plt.axis([-5, 5, -5, 5])

for i in range(len(X[0])):
    if(yd[i] == -1):
        plt.plot(X[0][i], X[1][i], 'ro')
    else:
        plt.plot(X[0][i], X[1][i], 'bo')


# ======================================== #
#     Affichage de la courbe d'erreur      #
# ======================================== #
plt.subplot(212)
plt.plot(range(len(error)), error)
plt.axis([0, len(error)-1, 0, np.array(error).max()])

plt.show()
