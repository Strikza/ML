import pandas as pd

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors             import KNeighborsClassifier
from sklearn.metrics               import confusion_matrix, accuracy_score
from warnings                      import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

CONFUSION_BAYES = True
CONFUSION_KNN   = True


#=======================================================================
def result_analysis(y_test, y_pred, label="", confusion=False):
    rate = accuracy_score(y_test, y_pred)
    print(label + str(rate * 100) + "%")

    if(confusion):
        matrix = confusion_matrix(y_test, y_pred)
        print(matrix)
        print()


#=======================================================================
def bayes_fct(data):
    # data[0]: X_train
    # data[1]: X_test
    # data[2]: y_train
    # data[3]: y_test

    bayes = QuadraticDiscriminantAnalysis()

    # Train the model using the training sets
    bayes.fit(data[0], data[2])

    # Predict Output: 
    y_pred = bayes.predict(data[1])

    # Result analysis:
    result_analysis(
        data[3], 
        y_pred, 
        label="Bayes : ", 
        confusion=CONFUSION_BAYES
    )


#=======================================================================
def kppv_fct(data, n_neighbors):
    # data[0]: X_train
    # data[1]: X_test
    # data[2]: y_train
    # data[3]: y_test

    knn = KNeighborsClassifier(n_neighbors = n_neighbors)

    # Train the model using the training sets
    knn.fit(data[0], data[2])

    # Predict Output: 
    y_pred = knn.predict(data[1])
    
    # Result analysis:
    result_analysis(
        data[3], 
        y_pred, 
        label="Kppv (" + str(n_neighbors) + " voisin(s)) : ", 
        confusion=CONFUSION_KNN
    )


# Lecture des données
print("Start to read files...")
p1_learning = pd.read_excel("./Data/p1_NonGaussien.xlsx", sheet_name="Ensemble Apprentissage")
print("Readed: 1/2")
p1_unknown  = pd.read_excel("./Data/p1_NonGaussien.xlsx", sheet_name="Inconnu"               )
print("Readed: 2/2")
print("Read done.")


X_train = []
y_train = []
X_test  = []
y_test  = []

for i in range(1, len(p1_learning.columns)):
    X_train.append([p1_learning.iloc[0, i], p1_learning.iloc[1, i]])
    y_train.append(p1_learning.iloc[2, i])

for i in range(1, len(p1_unknown.columns)):
    X_test.append([p1_unknown.iloc[0, i], p1_unknown.iloc[1, i]])
    y_test.append(p1_unknown.iloc[2, i])

data_set = [X_train, X_test, y_train, y_test]


#######################################
# Discrimination paramétrique (Bayes) #
#######################################

bayes_fct(data_set)


########
# Kppv #
########

for i in [1, 3, 5, 7, 13, 15]:
        kppv_fct(data_set, i)
