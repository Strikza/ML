import cv2 as cv
import numpy as np
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
  model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=data[0].shape))
  model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(10,  activation='softmax'))
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  # model.compile(optimizer='rmsprop', loss='MSE', metrics=['accuracy'])

  history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=size, verbose=1)

  Y_pred = model.predict(X_test)
  score  = model.evaluate(X_test, Y_test, verbose=1)

  result_analysis(Y_test, Y_pred, title + ': ')

  return score, history


# ======================================== #
#        Initialisation des données        #
# ======================================== #
print("Start to load images...")

path = "./Data/Wang/"

label = np.zeros((1000, 10))
X = []

for i in range(1000):
  ii = int(i/100)
  label[i, ii] = 1
  img = cv.imread(path + str(i) + '.jpg')
  norm = img[0:256, 0:256, :]
  X.append(norm)

print(label.shape)

X = np.array(X)
print("Load done.")

ratio_extracted = 0.1


# ======================================== #
#              Apprentissage               #
# ======================================== #
d, history_d = learning_fct(X, label, ratio_extracted, title='DEEP')


# ======================================== #
#          Affichage des courbes           #
# ======================================== #
plt.subplot(211)
plt.plot(history_d.history['accuracy'],     "b--", label="Accuracy of training data")
plt.plot(history_d.history['val_accuracy'], "b",   label="Accuracy of validation data")
plt.legend()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')

plt.subplot(212)
plt.plot(history_d.history['loss'],     "y--", label="Loss of training data")
plt.plot(history_d.history['val_loss'], "y",   label="Loss of validation data")
plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('Training Epoch')

plt.show()
