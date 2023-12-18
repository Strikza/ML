import numpy    as np
import scipy.io as scio

from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


#=======================================================================
def kppv(apprent, classe_origine, x, k):
    res = []
    for N2 in range(0, len(x[0])):
        xn   = [x[0][N2], x[1][N2]]
        dist = [500] * k
        elem = [0]   * k
        class_nbh = []

        for N1 in range(0, len(apprent[0])):
            app = [apprent[0][N1], apprent[1][N1]]

            d = np.sqrt((xn[0] - app[0])*(xn[0] - app[0]) + (xn[1] - app[1])*(xn[1] - app[1]))

            j = -1
            dtmp = -1
            for i in range(0, k):
                if(dist[i] > dtmp):
                    dtmp = dist[i]
                    j = i

            if(j != -1):
                if(d < dist[j]):
                    dist[j] = d
                    elem[j] = N1

        for i in range(0, k):
            class_nbh.append(classe_origine[elem[i]])


        res.append(predominanteInArr(class_nbh))

    return res


#=======================================================================
def predominanteInArr(arr):
    # m stocke l'élément majoritaire (si présent)
    m = -1
 
    # initialise le compteur i avec 0
    i = 0
 
    # faire pour chaque élément arr[j] dans la liste
    for j in range(len(arr)):
 
        # si le compteur i devient 0
        if i == 0:
 
            # définit le candidat actuel sur arr[j]
            m = arr[j]
 
            # remet le compteur à 1
            i = 1
 
        # sinon, incrémenter le compteur si arr[j] est un candidat courant
        elif m == arr[j]:
            i = i + 1
 
        # sinon, décrémenter le compteur si arr[j] est un candidat courant
        else:
            i = i - 1

    return m


#=======================================================================
def win_rate(result, expected):

    if(len(result) != len(expected)):
        print("Les tableaux n'ont pas la même taille.")
        return
    
    count = 0
    n = len(result)

    for i in range(0, n):
        if(result[i] == expected[i]):
            count += 1
    
    return count/n


####################
# Test des données #
####################
Data = scio.loadmat("./Data/p1_test.mat") # Chargé depuis la racine du projet
class_test = [1]*50 + [2]*50 + [3]*50


for i in [1, 3, 5, 7, 13, 15]:
    clas = kppv(Data['test'], class_test, Data['x'], k=i)
    rate = win_rate(clas, Data['clasapp'][0])
    print(str(i) +" voisin : " + str(rate * 100) + "%")
