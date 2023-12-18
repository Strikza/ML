import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow import keras
from sklearn.model_selection import train_test_split

CONFUSION = False

# ========================================
def parse_name(name):
  return int(name[:-4])


# ========================================
def result_analysis(y_test, y_pred, label=""):
  y_pred = np.argmax(y_pred, axis=1)
  y_test = np.argmax(y_test, axis=1)

  rate = accuracy_score(y_test, y_pred)
  print(label + str(rate * 100) + "%")

  if(CONFUSION):
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)
    print()


# ========================================
def learning_fct(data, label, size, r_state=42, title=''):

  X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=size, random_state=r_state)

  model = keras.models.Sequential()
  model.add(keras.layers.Dense(128, activation='relu'))
  model.add(keras.layers.Dense(64,  activation='relu'))
  model.add(keras.layers.Dense(10,  activation='softmax'))
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  # model.compile(optimizer='rmsprop', loss='MSE', metrics=['accuracy'])

  history = model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_split=size, verbose=0)

  Y_pred = model.predict(X_test)
  score  = model.evaluate(X_test, Y_test, verbose=1)

  result_analysis(Y_test, Y_pred, title + ': ')

  return score, history


# ======================================== #
#        Initialisation des données        #
# ======================================== #
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

ALL = np.concatenate((PHOG, JCD, CEDD, FCTH, FCH), axis=1)

label = np.zeros((1000, 10))

for i in range(1000):
  ii = int(i/100)
  label[i, ii] = 1

ratio_extracted = 0.1


# ======================================== #
#              Apprentissage               #
# ======================================== #
p,  history_p  = learning_fct(PHOG, label, ratio_extracted, title='PHOG')
j,  history_j  = learning_fct(JCD,  label, ratio_extracted, title='JCD')
c,  history_c  = learning_fct(CEDD, label, ratio_extracted, title='CEDD')
ft, history_ft = learning_fct(FCTH, label, ratio_extracted, title='FCTH')
f,  history_f  = learning_fct(FCH,  label, ratio_extracted, title='FCH')
a,  history_a  = learning_fct(ALL,  label, ratio_extracted, title='ALL')

print("PHOG - score: " + str(p))
print("JCD  - score: " + str(j))
print("CEDD - score: " + str(c))
print("FCTH - score: " + str(ft))
print("FCH  - score: " + str(f))
print("ALL  - score: " + str(a))


# ======================================== #
#          Affichage des courbes           #
# ======================================== #
plt.subplot(221)
plt.plot(history_a.history['accuracy'],     "b--", label="Accuracy of training data")
plt.plot(history_a.history['val_accuracy'], "b",   label="Accuracy of validation data")
plt.legend()
plt.title('Model Accuracy - ALL')
plt.ylabel('Accuracy')

plt.subplot(223)
plt.plot(history_a.history['loss'],     "y--", label="Loss of training data")
plt.plot(history_a.history['val_loss'], "y",   label="Loss of validation data")
plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('Training Epoch')

plt.subplot(222)
plt.plot(history_f.history['accuracy'],     "b--", label="Accuracy of training data")
plt.plot(history_f.history['val_accuracy'], "b",   label="Accuracy of validation data")
plt.legend()
plt.title('Model Accuracy - FCH')
plt.ylabel('Accuracy')

plt.subplot(224)
plt.plot(history_f.history['loss'],     "y--", label="Loss of training data")
plt.plot(history_f.history['val_loss'], "y",   label="Loss of validation data")
plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('Training Epoch')

plt.show()
