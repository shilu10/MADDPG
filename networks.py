import tensorflow as tf 
from tensorflow.keras.layers import Dense, Conv2D, Input, Lambda, concatenate
 
class ActorNetwork(tf.keras.Model):
    def __init__(self, actor_input_dims, action_dims):
        super(ActorNetwork, self).__init__()
        self.fc1 = Dense(64, activation="relu", input_shape=(actor_input_dims, ), kernel_initializer="he_uniform")
        self.fc2 = Dense(32, activation="relu", kernel_initializer="he_uniform")
        self.fc3 = Dense(32, activation="relu", kernel_initializer="he_uniform")
        
        self.out = Dense(action_dims, activation='softmax')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.out(x)
        return x 


class CriticNetwork(tf.keras.Model):
    def __init__(self, critic_input_dims, action_dims): 
        super(CriticNetwork, self).__init__()
        self.fc1 = Dense(64, activation="relu", kernel_initializer="he_uniform", input_shape=(critic_input_dims, ))
        self.fc2 = Dense(32, activation="relu", kernel_initializer="he_uniform")
        self.fc3 = Dense(32, activation="relu", kernel_initializer="he_uniform")
        self.out = Dense(1, activation='linear')

    def call(self, state, action):
        x = self.fc1(tf.concat([state, action], axis=1))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.out(x)
        return x 