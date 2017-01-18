import gym
from gym import wrappers
import numpy as np
env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

model = Sequential([
    Dense(2, input_dim=4, init='uniform'),
])

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='mse',
              )

def learn(data, labels, model):
    
    model.fit(data, labels, nb_epoch=1, batch_size=1)
    return model
    

for i_episode in range(100):
    observation = env.reset()
    observation = np.asarray(observation)
    observation = np.reshape(observation,[1,4])
    for t in range(100):
        env.render()

        allQ = model.predict(observation)
        action = np.argmax(allQ)
        observation, reward, done, info = env.step(action)
        observation = np.asarray(observation)
        observation = np.reshape(observation,[1,4])
        nextQ = model.predict(observation)
        maxNextQ = np.max(nextQ)
        targetQ = allQ
        targetQ[0,action] = reward + 0.8*maxNextQ
        model = learn(observation, targetQ, model)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
# gym.upload('/tmp/cartpole-experiment-1', api_key='sk_FCbQriTVe7CzkJhNLYyA')