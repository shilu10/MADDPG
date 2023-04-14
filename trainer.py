from npy_append_array import NpyAppendArray
import numpy as np
import os 
import tensorflow as tf 
import tensorflow.keras as keras 
from telegram_bot import * 
from multiagent_experience_replay import * 
from maddpg_agent import * 
from ddpg_agent import * 
from video_recorder import * 
from writer import *
from utils import *

class Trainer:   
    def __init__(self, env, gamma, alpha, beta, tau, soft_update, 
                                 noe, max_steps, is_tg, tg_bot_freq_epi, record, 
                                 env_obs_dims, actors_input_dims, n_agents, n_actions, batch_size, mem_size): 
        self.env = env 
        self.target_score = 80
        self.noe = noe
        self.max_steps = max_steps
        self.is_tg = is_tg
        self.tg_bot_freq_epi = tg_bot_freq_epi
        self.record = record 
        self.writer = Writer("model_training_results.txt")
        self.recorder = RecordVideo("maddpg", "videos/", 20)
        self.maddpg_agent = MADDPGAgent(actors_input_dims, env_obs_dims, batch_size, n_agents, 
                                             n_actions, tau, alpha, beta, soft_update, gamma)
        
        self.maddpg_agent.initialize_agents()
        
        self.memory = MultiAgentReplayBuffer(mem_size, env_obs_dims, 
                                                 actors_input_dims, n_actions, n_agents, batch_size)
        
        
    def create_env_obs(self, actors_state): 
        env_obs = np.array([])
        for actor_state in actors_state:
            env_obs = np.append(env_obs, actor_state)

return np.array(env_obs)
    
    def train_rl_model(self): 
        learned = False
        avg_rewards = []
        best_reward = float("-inf")
        episode_rewards = []
        tot_steps = 0
        for episode in range(self.noe): 
            n_steps = 0 
            actors_state = env.reset()
            reward = 0 
            
            if record and episode % 50 == 0:
                img = self.env.render()
                self.recorder.add_image(img)

            for step in range(self.max_steps): 
                
                actors_actions = self.maddpg_agent.get_actions(actors_state)

                next_info = self.env.step(actors_actions)
                actors_next_state, reward_probs, dones, _ = next_info
          #      print(dones)
                reward += sum(reward_probs)
                
                env_obs = self.create_env_obs(actors_state)
                env_nxt_obs = self.create_env_obs(actors_next_state)

                self.memory.store_experiences(actors_state, env_obs, actors_actions, 
                                                     reward_probs, actors_next_state, env_nxt_obs, dones)
                
                if self.memory.is_sufficient() and tot_steps % 50 == 0: 
                    learned = True
                    actors_states_batch, env_obs_batch, actors_actions_batch, rewards_batch, \
                                actor_new_states_batch, env_next_obs_batch, terminal_batch = self.memory.sample_experiences()
                    

                    self.maddpg_agent.learn(env_obs_batch, env_next_obs_batch, actors_states_batch, 
                                                    actors_actions_batch, actors_actions_batch, rewards_batch, terminal_batch)

                actors_state = actors_next_state
                n_steps += 1   
                tot_steps += 1 
                
                # record
                if record and episode % 50 == 0:
                    img = self.env.render()
                    self.recorder.add_image(img)
                
                if np.any(dones): 
                    break
            
            episode_rewards.append(reward)
            avg_reward = np.mean(episode_rewards[-100:])
            avg_rewards.append(avg_reward)

            result = f"Episode: {episode}, Steps: {n_steps}, Reward: {reward}, Best reward: {best_reward}, Avg reward: {avg_reward}"
            self.writer.write_to_file(result)
            if episode % 100 == 0: 
                print(result)
            
            # Recording.
            if record and episode % 50 == 0:
                self.recorder.save(episode)
                
            # Saving Best Model
            if reward > best_reward and episode != 0 and learned: 
                best_reward = reward
                self.maddpg_agent.save_models()
                
            # Telegram bot
            if self.is_tg and episode % self.tg_bot_freq_epi == 0: 
                info_msg(episode+1, self.noe, reward, best_reward, "d")
                
            # Eatly Stopping
            if episode > 100 and np.mean(episode_rewards[-20:]) >= self.target_score: 
                break
                
        return episode_rewards, avg_rewards, best_reward
    
