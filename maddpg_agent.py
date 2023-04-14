import tensorflow as tf 
import tensorflow.keras as keras 
import numpy as np 
import random 
import time, os 
from ddpg_agent import * 


class MADDPGAgent: 
    def __init__(self, actors_input_dims, critic_input_dims, batch_size, 
                                         n_agents, n_actions, tau, alpha, beta, soft_update, gamma): 
        self.actors_input_dims = actors_input_dims
        self.critic_input_dims = critic_input_dims
        self.n_agents = n_agents 
        self.n_actions = n_actions 
        self.tau = tau 
        self.gamma = gamma
        self.alpha = alpha 
        self.beta = beta
        self.soft_update = soft_update
        self.batch_size = batch_size
        
        self.ddpg_agents = []
        
        
    def initialize_agents(self): 
        for i in range(self.n_agents): 
            actor_input_dims = self.actors_input_dims[i]
            critic_input_dims = self.critic_input_dims + self.n_actions * self.n_agents
            agent = DDPGAgent(actor_input_dims, critic_input_dims, self.n_actions, 
                                              self.alpha, self.beta, self.soft_update, self.tau)
            
            self.ddpg_agents.append(agent)
            
            
    def get_actions(self, actors_state): 
        actions = []
        for agent_indx, agent in enumerate(self.ddpg_agents): 
            actor_state = actors_state[agent_indx]
            action = agent.get_action(actor_state)
            actions.append(action)
        
        return actions 
    
    
    def learn(self, env_obs, env_nxt_obs, actors_states, 
                              actors_nxt_state, actions, rewards, dones): 
        
        env_obs = tf.convert_to_tensor(env_obs, dtype=tf.float32)
        env_nxt_obs = tf.convert_to_tensor(env_nxt_obs, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)        
        dones = tf.convert_to_tensor(dones, dtype=tf.bool)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        
        
        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.ddpg_agents):
            new_states = tf.convert_to_tensor(actors_nxt_state[agent_idx], dtype=tf.float32)

            new_pi = agent.target_actor(new_states)
            all_agents_new_actions.append(new_pi)
            old_agents_actions.append(actions[agent_idx])

        next_actions = tf.concat([acts for acts in all_agents_new_actions], axis=1)
        old_actions = tf.concat([acts for acts in old_agents_actions], axis=1)
        
        for agent_idx, agent in enumerate(self.ddpg_agents):
            
            with tf.GradientTape() as tape: 
                next_obs_critic_value = agent.target_critic(env_nxt_obs, next_actions)
                next_obs_critic_value = tf.squeeze(next_obs_critic_value)

                next_obs_critic_value = tf.where(dones[:, 0], 0.0, next_obs_critic_value)

                curr_obs_critic_value = agent.critic(env_obs, old_actions)
                curr_obs_critic_value = tf.squeeze(curr_obs_critic_value)

                target = rewards[:, agent_idx] + self.gamma*next_obs_critic_value
                critic_loss = self.critic_loss(curr_obs_critic_value, target)
            
            critic_params = agent.critic.trainable_variables
            critic_grads = tape.gradient(critic_loss, critic_params)
            
            agent.critic.optimizer.apply_gradients(zip(critic_grads, critic_params))
            
            with tf.GradientTape() as tape: 
                all_agents_new_mu_actions = []
                for agent_idx, agent in enumerate(self.ddpg_agents): 
                    mu_states = tf.convert_to_tensor(actors_states[agent_idx])
                    pi = agent.actor(mu_states)
                    all_agents_new_mu_actions.append(pi)
                mu = tf.concat([acts for acts in all_agents_new_mu_actions], axis=1)
                
                q_value = agent.critic(env_obs, mu)
                q_value = tf.squeeze(q_value)
                actor_loss = self.actor_loss(q_value)
            
            actor_params = agent.actor.trainable_variables
            actor_grads = tape.gradient(actor_loss, actor_params)
            
            agent.actor.optimizer.apply_gradients(zip(actor_grads, actor_params))
            
            agent.update_target_networks()
        
        
    def save_models(self):
        for agent_indx, agent in enumerate(self.ddpg_agents): 
            agent.save_models(agent_indx)
    
    def load_models(self):
        pass
    
    def critic_loss(self, q_value, target_q_value): 
        mse = tf.keras.losses.MeanSquaredError()
        loss = mse(target_q_value, q_value)
        
        return loss
    
    def actor_loss(self, q_value):
        return -tf.reduce_mean(q_value)
