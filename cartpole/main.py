import gym
from gym import wrappers
import numpy as np
env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

model = Sequential([
    Dense(8, input_dim=4, init='uniform'),
    Dense(1, init='uniform'),
])

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='mse',
              )

def learn(data, labels, model):
    
    model.fit(data, labels, nb_epoch=1, batch_size=1,verbose=False,)
    return model
    

for i_episode in range(100):
    observation = env.reset()
    observation = np.asarray(observation)
    observation = np.reshape(observation,[1,4])
    data=[]
    for t in range(1000):
        env.render()
        a = model.predict(observation)
        action = np.argmax(a)
        observation, reward, done, info = env.step(action)
        data.append(observation)
        observation = np.asarray(observation)
        observation = np.reshape(observation,[1,4])
        if done:
            if len(data)>=9+i_episode:
                labels=np.ones((len(data),1))
            else:
                labels=np.zeros((len(data),1))
            print np.shape(data)
            print np.shape(labels)
            model = learn(data, labels, model)
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
# gym.upload('/tmp/cartpole-experiment-1', api_key='sk_FCbQriTVe7CzkJhNLYyA')