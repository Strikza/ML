import math
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
def perceptron_simple(x, w):

    u = w[0] + w[1]*x[0] + w[2]*x[1]

    return 1/(1+math.exp(-u))


# ========================================
def multiperceptron(x, w1, w2):

    x1 = perceptron_simple(x, [w1[0][0], w1[0][1], w1[0][2]])
    x2 = perceptron_simple(x, [w1[1][0], w1[1][1], w1[1][2]])

    return perceptron_simple([x1, x2], w2)


# ======================================== #
#        Initialisation des données        #
# ======================================== #
w1 = [
    [-0.5, 2, -1],
    [ 0.5, 1,  0.5]
]
w2 = [2, -1, 1]
x  = [1, 1]

y = multiperceptron(x, w1, w2)
print("Class: " + str(y))

display_plot(w1[0], [-2, 4], [0.9, 0.5, 0.1])
display_plot(w1[1], [-2, 4], [0.9, 0.1, 0.5])
plt.plot(x[0], x[1], 'ro')

plt.axis([-2, 4, -2, 4])
plt.show()