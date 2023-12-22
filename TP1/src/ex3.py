import pandas as pd
import random as rd
import numpy  as np

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors             import KNeighborsClassifier
from sklearn.model_selection       import train_test_split
from sklearn.metrics               import confusion_matrix, accuracy_score
from warnings                      import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

CONFUSION_BAYES = False
CONFUSION_KNN   = False


#=======================================================================
def parse_name(name):
    return int(name[:-4])


#=======================================================================
def result_analysis(y_test, y_pred, label="", confusion=False):
    rate = accuracy_score(y_test, y_pred)
    print(label + str(rate * 100) + "%")

    if(confusion):
        matrix = confusion_matrix(y_test, y_pred)
        print(matrix)
        print()


#=======================================================================
def bayes_fct(data, label, size, r_state=42):

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=size, random_state=r_state, stratify=label)

    bayes = QuadraticDiscriminantAnalysis()

    # Train the model using the training sets
    bayes.fit(X_train, y_train)

    # Predict Output: 
    y_pred = bayes.predict(X_test)

    # Result analysis:
    result_analysis(
        y_test, 
        y_pred, 
        label="Bayes : ", 
        confusion=CONFUSION_BAYES
    )


#=======================================================================
def kppv_fct(data, label, size, r_state=42):

    data_split = [X_train, X_test, y_train, y_test] = train_test_split(data, label, test_size=size, random_state=r_state, stratify=label)

    for i in [1, 3, 5, 7, 13, 15]:
        aux_kppv(data_split, i)
    
#-----------------------------------------------------
def aux_kppv(datas, n_neighbors):
    # datas[0]: X_train
    # datas[1]: X_test
    # datas[2]: y_train
    # datas[3]: y_test

    knn = KNeighborsClassifier(n_neighbors = n_neighbors)

    # Train the model using the training sets
    knn.fit(datas[0], datas[2])

    # Predict Output: 
    y_pred = knn.predict(datas[1])
    
    # Result analysis:
    result_analysis(
        datas[3], 
        y_pred, 
        label="Kppv (" + str(n_neighbors) + " voisin(s)) : ", 
        confusion=CONFUSION_KNN
    )


#=======================================================================
#=======================================================================

# Lecture des données
print("Start to read files...")
PHOG = np.array(pd.read_excel("./Data/WangSignatures.xlsx", 
                              sheet_name="WangSignaturesPHOG", 
                              header=None,
                              converters={0:parse_name}
                            ).sort_values(by=0).drop(0, axis=1))
print("Read: 1/5")
JCD = np.array(pd.read_excel("./Data/WangSignatures.xlsx", 
                              sheet_name="WangSignaturesJCD", 
                              header=None,
                              converters={0:parse_name}
                            ).sort_values(by=0).drop(0, axis=1))
print("Read: 2/5")
CEDD = np.array(pd.read_excel("./Data/WangSignatures.xlsx", 
                              sheet_name="WangSignaturesCEDD", 
                              header=None,
                              converters={0:parse_name}
                            ).sort_values(by=0).drop(0, axis=1))
print("Read: 3/5")
FCTH = np.array(pd.read_excel("./Data/WangSignatures.xlsx", 
                              sheet_name="WangSignaturesFCTH", 
                              header=None,
                              converters={0:parse_name}
                            ).sort_values(by=0).drop(0, axis=1))
print("Read: 4/5")
FCH = np.array(pd.read_excel("./Data/WangSignatures.xlsx", 
                              sheet_name="WangSignaturesFuzzyColorHistogr", 
                              header=None,
                              converters={0:parse_name}
                            ).sort_values(by=0).drop(0, axis=1))
print("Read: 5/5")
print("Read done.")

label = [0]*100 + [1]*100 + [2]*100 + [3]*100 + [4]*100  + [5]*100 + [6]*100 + [7]*100 + [8]*100 + [9]*100
ratio_extracted = 0.2

print("[ PHOG ]")
bayes_fct(PHOG, label, ratio_extracted)
print("[ JCD ]")
bayes_fct(JCD,  label, ratio_extracted)
print("[ CEDD ]")
bayes_fct(CEDD, label, ratio_extracted)
print("[ FCTH ]")
bayes_fct(FCTH, label, ratio_extracted)
print("[ FCH ]")
bayes_fct(FCH,  label, ratio_extracted)

print("\n========================== [ PHOG ] ==========================")
kppv_fct(PHOG,label, ratio_extracted)
print("\n========================== [ JCD ] ===========================")
kppv_fct(JCD, label, ratio_extracted)
print("\n========================== [ CEDD ] ==========================")
kppv_fct(CEDD,label, ratio_extracted)
print("\n========================== [ FCTH ] ==========================")
kppv_fct(FCTH,label, ratio_extracted)
print("\n========================== [ FCH ] ===========================")
kppv_fct(FCH, label, ratio_extracted)