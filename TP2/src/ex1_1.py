import numpy as np
import matplotlib.pyplot as plt


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
def perceptron_simple(x, w, active):

    u = w[0] + w[1]*x[0] + w[2]*x[1]
    
    if(active):
        return np.sign(u)
    else:
        return np.tanh(u)


# ======================================== #
#        Initialisation des données        #
# ======================================== #
w = [-0.5, 1, 1]
or_value = [
    [0, 0, 1, 1],
    [0, 1, 0, 1]
]

t = [-0.5, 1]
display_plot(w, t)


# ======================================== #
#            Test et affichage             #
# ======================================== #
for i in range(len(or_value[0])):
    y = perceptron_simple([or_value[0][i], or_value[1][i]], w, True)
    plt.plot(or_value[0][i], or_value[1][i],'ro')
    plt.text(or_value[0][i]+.01, or_value[1][i]+.01, str(y))

plt.axis([-0.5, 1.5, -0.5, 1.5])
plt.show()

