
import numpy  as np
import pandas as pd

from sklearn.metrics         import confusion_matrix, accuracy_score
from tensorflow              import keras
from sklearn.model_selection import train_test_split

# ========================================
def parse_name(name):
    return int(name[:-4])


# ========================================
def result_analysis(y_test, y_pred, label="", confusion=False):
    rate = accuracy_score(y_test, y_pred)
    print(label + str(rate * 100) + "%")

    if(confusion):
        matrix = confusion_matrix(y_test, y_pred)
        print(matrix)
        print()


# ========================================
def learning_fct(data, label, size, oui, r_state=42):

    X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=size, random_state=r_state, stratify=label)

    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))

    input_space = oui

    # === Definition du Modele == #
    model = keras.models.Sequential([
        keras.layers.Dense(8, activation='tanh', input_shape=[input_space]),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
    score   = model.evaluate(X_test, Y_test, verbose=1)

    print(history.history['loss'])


# ======================================== #
#        Initialisation des données        #
# ======================================== #
print("Start to read files...")
PHOG = np.array(pd.read_excel(".\Data\WangSignatures.xlsx", 
                              sheet_name="WangSignaturesPHOG", 
                              header=None,
                              converters={0:parse_name}
                            ).sort_values(by=0).drop(0, axis=1))
print("Read: 1/5")
JCD = np.array(pd.read_excel(".\Data\WangSignatures.xlsx", 
                              sheet_name="WangSignaturesJCD", 
                              header=None,
                              converters={0:parse_name}
                            ).sort_values(by=0).drop(0, axis=1))
print("Read: 2/5")
CEDD = np.array(pd.read_excel(".\Data\WangSignatures.xlsx", 
                              sheet_name="WangSignaturesCEDD", 
                              header=None,
                              converters={0:parse_name}
                            ).sort_values(by=0).drop(0, axis=1))
print("Read: 3/5")
FCTH = np.array(pd.read_excel(".\Data\WangSignatures.xlsx", 
                              sheet_name="WangSignaturesFCTH", 
                              header=None,
                              converters={0:parse_name}
                            ).sort_values(by=0).drop(0, axis=1))
print("Read: 4/5")
FCH = np.array(pd.read_excel(".\Data\WangSignatures.xlsx", 
                              sheet_name="WangSignaturesFuzzyColorHistogr", 
                              header=None,
                              converters={0:parse_name}
                            ).sort_values(by=0).drop(0, axis=1))
print("Read: 5/5")
print("Read done.")

label = [0]*100 + [1]*100 + [2]*100 + [3]*100 + [4]*100  + [5]*100 + [6]*100 + [7]*100 + [8]*100 + [9]*100
ratio_extracted = 0.2

learning_fct(PHOG, label, ratio_extracted, 255)
# learning_fct(JCD,  label, ratio_extracted, 168)
# learning_fct(CEDD, label, ratio_extracted, 144)
# learning_fct(FCTH, label, ratio_extracted, 192)
# learning_fct(FCH,  label, ratio_extracted, 125)
