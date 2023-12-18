# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 15:13:20 2022

@author: Romain
"""

from typing import Tuple, Optional
from enum import Enum
import os
import numpy as np
import random 
from tensorflow import GradientTape,one_hot,reduce_sum

from keras.models import Sequential,clone_model
from keras.layers import Dense
from keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Nadam


class Box(Enum):
    EMPTY = 0
    HERO = 1
    BEGIN = 2
    DRAGON = 3
    TREASURE = 4
    SWORD = 5 


ELEM_CHAR_DICT = {
    Box.EMPTY.value: '.',
    Box.BEGIN.value: 'B',
    Box.HERO.value: 'X',
    Box.TREASURE.value: 'T',
    Box.SWORD.value: 'S',
    Box.DRAGON.value: 'D',
}

#affichage
def draw(space: np.ndarray, hero: Optional[Tuple[int, int]] = None, name: Optional[str] = None) -> None:
    os.system('cls' if os.name == 'nt' else 'clear')
    if name is not None:
        print(name)
    line_expr = "| v |"

    dashes = '-'*(space.shape[1]*2+1)
    bounds = line_expr.replace('v',dashes)
    bounds = bounds.replace('| ','+').replace(' |','+')
    print(bounds)
    for idx, row in enumerate(space):
        elems_str = []
        for idy, elem in enumerate(row):
            if hero is not None and hero == [idx, idy]:
                elem = Box.HERO.value
            elems_str.append(ELEM_CHAR_DICT[elem])
        values_str = ' '.join(f'{v}' for v in elems_str)
        line = line_expr.replace('v',values_str)
        print(line)
    print(bounds, flush=True)
    
# fin affichage 

dragon=[[1,0],[1,2],[2,3],[3,1]]

def generate_space(x,y,dragon):
    space = np.zeros((x,y),int)
    for pos_drag in dragon:
        space[pos_drag[0],pos_drag[1]]=3
    space[0][0]=2
    space[x-1][y-1]=4
    return space

space= generate_space(4,4,dragon)


draw(space)

list_action=np.array([[-1,0],[1,0],[0,-1],[0,1]])

def application_action(action,hero_pos,space):
    new_Hpos=action+hero_pos
    if new_Hpos[0]<0 or new_Hpos[0]>=len(space) or new_Hpos[1]<0 or new_Hpos[1]>=space[0].size :
        new_Hpos=hero_pos
    state=new_Hpos[0]*space[0].size+new_Hpos[1]
    type_case=space[new_Hpos[0]][new_Hpos[1]] 
    if type_case==3 :
        reward = -10
    elif type_case==4 : 
        reward = 50
    else: 
        reward =-1

    return new_Hpos,state,reward,(type_case==4 or type_case==3)


model = Sequential()
model.add(Dense(16, activation='relu',input_shape=[space.size]))
model.add(Dense(4))



optimizer =Nadam(learning_rate=0.001)
loss_fn = mean_squared_error

model_stable = clone_model(model)
model_stable.set_weights(model.get_weights())

gamma=.99
nbPartie = 100
for i in range(nbPartie):
    fin=False
    nbCoup=0
    H_Pos=np.array([0,0])
    
    vec_etat = np.zeros(space.size)
    vec_etat[int((space.size/len(space))*H_Pos[0] + H_Pos[1])] = 1
    
    while(nbCoup<100 and not fin):
        epsi=nbPartie/(nbPartie+i)
        if random.uniform(0,1)<epsi:
            #choix random
            action=random.randint(0,3)
        else:
            #max proba de la pos
            Sortie_Q = model.predict(np.array([vec_etat]))
            action = np.argmax(Sortie_Q[0])

        H_Pos,state,reward,fin = application_action(list_action[action],H_Pos,space)
        vec_etat_next = np.zeros(space.size) #ca sera l'entrée du réseau
        vec_etat_next[int((space.size/len(space))*H_Pos[0] + H_Pos[1])] = 1
        next_Q = model_stable.predict(np.array([vec_etat_next]))
        next_Q_max = np.max(next_Q[0])
        if fin :
            target = reward 
        else:
            target = reward + gamma * next_Q_max
        x = np.array([vec_etat])
        with GradientTape() as tape:
            predict = model(x)
            mask=one_hot(action,4)
            val_predict = reduce_sum(predict*mask,axis = 1)
            loss = loss_fn(target,val_predict)
        gradients = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        if nbCoup % 10 == 0:
            model_stable.set_weights(model.get_weights()) 
        nbCoup+=1
        
print ("FIN")

fin =False
H_Pos=np.array([0,0])

nbCoup = 0
while (not fin and nbCoup < 20) :
    draw(space,H_Pos.tolist())
    vec_etat = np.zeros(space.size)
    vec_etat[int((space.size/len(space))*H_Pos[0] + H_Pos[1])] = 1
    Sortie_Q = model.predict(np.array([vec_etat]))
    action = np.argmax(Sortie_Q[0])
    H_Pos,state,reward,fin = application_action(list_action[action],H_Pos,space)
    nbCoup+=1
    #draw(space,H_Pos.tolist())
