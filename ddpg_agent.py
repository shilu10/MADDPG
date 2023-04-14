from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import numpy as np
import os 
import random 
from networks import *

class DDPGAgent:
  
    def __init__(self, actor_input_dims, critic_input_dims, n_actions, alpha, beta, soft_update, tau): 
        self.tau = tau 
        self.soft_update = soft_update
        self.fname = "models/"

        self.actor = ActorNetwork(actor_input_dims, n_actions)
        self.target_actor = ActorNetwork(actor_input_dims, n_actions)
        self.critic = CriticNetwork(critic_input_dims, 1)
        self.target_critic = CriticNetwork(critic_input_dims, 1)
        
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_target_networks()
        
    def ou_noise(self, x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
        return x + rho * (mu-x) * dt + sigma * np.sqrt(dt) * np.random.normal(size=dim)
        
    def get_action(self, state): 
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state = tf.reshape(state, (1, state.shape[0]))
        
        actions = self.actor(state)
        noise = tf.random.normal([n_actions])
        action = actions + noise

        return action.numpy()[0]
 
    def save_models(self, no):
        self.actor.save(self.fname + f"maddpg_actor_{no}")
        self.target_actor.save(self.fname + f"maddpg_target_actor_{no}")
        self.critic.save(self.fname  + f"maddpg_critic_{no}")
        self.target_critic.save(self.fname  + f"maddpg_target_critic_{no}")
        print("[+] Saving the models") 

    def load_models(self, no):
        self.actor = tf.keras.models.load_model(self.fname  + f"madddpg_actor_{no}") 
        self.target_actor = tf.keras.models.load_model(self.fname + f"maddpg_target_actor_{no}") 
        self.critic = tf.keras.models.load_model(self.fname +  f"maddpg_critic_{no}") 
        self.target_critic = tf.keras.models.load_model(self.fname + f"maddpg_target_critic_{no}") 
        print("[+] Loading the models")
  
    
    def update_target_networks(self):
        actor_weights = self.actor.get_weights()
        t_actor_weights = self.target_actor.get_weights()
        critic_weights = self.critic.get_weights()
        t_critic_weights = self.target_critic.get_weights()
        if self.soft_update: 
            for i in range(len(actor_weights)):
                t_actor_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * t_actor_weights[i]

            for i in range(len(critic_weights)):
                t_critic_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * t_critic_weights[i]

            self.target_actor.set_weights(t_actor_weights)
            self.target_critic.set_weights(t_critic_weights)
            
        else: 
            self.target_actor.set_weights(t_actor_weights)
            self.target_critic.set_weights(t_critic_weights)
  
