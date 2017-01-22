# -*- coding: utf-8 -*-
import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('FrozenLake-v0')

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

model = Sequential([
    Dense(4, input_dim=16, weights=[0.01*np.random.rand(16,4), np.zeros(4)]),
])

sgd = SGD(lr=0.1)
model.compile(optimizer=sgd,
              loss='mse',
              )

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
#create lists to contain total rewards and steps per episode
jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    print "episode %d"%i
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    #The Q-Network
    while j < 99:
        j+=1
        #Choose an action by greedily (with e chance of random action) from the Q-network
        allQ = model.predict(np.identity(16)[s:s+1])
        
        a = np.argmax(allQ)
        if np.random.rand(1) < e:
            a = env.action_space.sample()
        #Get new state and reward from environment
        s1,r,d,_ = env.step(a)
        #Obtain the Q' values by feeding the new state through our network
        Q1 = model.predict(np.identity(16)[s1:s1+1])
        #Obtain maxQ' and set our target value for chosen action.
        maxQ1 = np.max(Q1)

        targetQ = allQ
        
        targetQ[0,a] = r + y*maxQ1
        #Train our network using target and predicted Q values
        model.fit(np.identity(16)[s:s+1], targetQ, verbose=False)
        rAll += r
        print rAll
        s = s1
        if d == True:
#            Reduce chance of random action as we train the model.
            e = 1./((i/50) + 10)
            break
    jList.append(j)
    rList.append(rAll)
    if i%1000==0:
        plt.plot(rList)
        plt.show()
        plt.plot(jList)
        plt.show()
print "Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%"

plt.plot(rList)
plt.show()
plt.plot(jList)
plt.show()
env.close()