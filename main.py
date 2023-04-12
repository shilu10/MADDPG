from make_env import make_env
import time
import signal
import time
import sys
import pickle
import os 

scenario = "simple_adversary"
env = make_env(scenario)
print(env)
record = False
gamma = 0.99
alpha = 0.01
beta = 0.01
batch_size = 64
tau = 0.01 
soft_update = False 
noe = 50000
max_steps = 30
is_tg = True 
tg_bot_freq_epi = 20
record = False 
mem_size = 100000

n_agents = env.n

actors_input_dims = [env.observation_space[dim].shape[0] for dim in range(env.n)]
env_obs_dims = sum(actors_dim)
n_actions = env.action_space[0].n
    
if __name__ == "__main__": 
    
    try: 
        
        if not os.path.exists("videos"): 
            os.mkdir("videos")

        if not os.path.exists("test_videos"):
            os.mkdir("test_videos")
            
        manage_memory()
        trainer = Trainer(env, gamma, alpha, beta, tau, soft_update, 
                                 noe, max_steps, is_tg, tg_bot_freq_epi, record, 
                                 env_obs_dims, actors_input_dims, n_agents, n_actions, batch_size, mem_size
                         )
        episode_rewards, avg_rewards, best_reward = trainer.train_rl_model()
        
        with open("ddpg_episode_rewards.obj", "wb") as f: 
            pickle.dump(episode_rewards, f)
        
        with open("ddpg_avg_rewards.obj", "wb") as f: 
            pickle.dump(avg_rewards, f)
        
        x = [i+1 for i in range(len(episode_rewards))]
        plot_learning_curve(x, episode_rewards, "maddpg_simple_adv")

       # model_path = "models/lunarlander_DQN_q_value/"

        #evaluator = Eval(env, action_space, model_path, "vanilla_dqn_lunarlander", 10)
        #evaluator.test()
        
    except Exception as error: 
        raise error
        
   # eval_model(env, "keras model", "videos/", fps=10)
