from make_env import make_env
import time
import signal
import time
import sys
import pickle
import os 
from trainer import *
import argparse

parser = argparse.ArgumentParser(prog='MADDPG', description='MADDPG Model trainer function')
parser.add_argument('--scenario', default="simple")
parser.add_argument('--record', default=False)
parser.add_argument('--gamma', default=0.95)
parser.add_argument('--alpha', default=0.01)
parser.add_argument('--beta', default=0.01)
parser.add_argument('--batch_size', default=64)
parser.add_argument('--tau', default=0.01)
parser.add_argument('--noe', default=1000)
parser.add_argument('--soft_update', default=False)
parser.add_argument('--max_steps', default=1000)
parser.add_argument('--is_tg', default=False)
parser.add_argument('--tg_bot_freq_epi', default=50)
parser.add_argument("--mem_size", default=100000)
parser.add_argument("--tg_token", default=None)
parser.add_argument("--tg_chat_id", default=None)
args = parser.parse_args()

if args.is_tg and (not args.tg_token or not args.tg_chat_id): 
    raise ValueError("if is_tg is True, then tg_token or tg_chat_id should not be None")

if args.is_tg and args.tg_token and args.tg_chat_id: 
    os.environ["TG_TOKEN"] = args.tg_token 
    os.environ["TG_CHAT_ID"] = args.tg_chat_id

env = make_env(args.scenario)
actors_input_dims = [env.observation_space[dim].shape[0] for dim in range(env.n)]
env_obs_dims = sum(actors_dim)
n_actions = env.action_space[0].n

def main(env, actors_input_dims, env_obs_dims, n_actions): 
    """
        This function, will run train the model of MADDPG for the given scenario.
        Params:
            env(dtype: make_env.ENV): this is gyn kike environment, but created using multi particle env.
            actors_input_dims(dtype: List): Input State Dimension of each agent, bcoz each agent will have a different input state dimension.
            env_obs_dims(dtype: Int): combined input dims of each agent, will be considered as a environment observation dimension.
            n_actions(dtype: Int): Number of actions in the environment.

        returns: 
            episode_rewards(dtype: List): Rewards, that are earned by the agent, while training.
            avg_rewards(dtype: List): Running Average of the rewards of the agent.
            best_reward(dtype: Float): The best reward, that is gained by the agent.
    """
    try: 
        if not os.path.exists("videos"): 
            os.mkdir("videos")

        if not os.path.exists("test_videos"):
            os.mkdir("test_videos")
            
        manage_memory()
        trainer = Trainer(env, args.gamma, args.alpha, args.beta, args.tau, args.soft_update, args.noe, 
                                          args.max_steps, args.is_tg, args.tg_bot_freq_epi, args.record, 
                                          env_obs_dims, actors_input_dims, args.n_agents, args.n_actions, 
                                          args.batch_size, args.mem_size
                                    )

        episode_rewards, avg_rewards, best_reward = trainer.train_rl_model()
        
        with open("ddpg_episode_rewards.obj", "wb") as f: 
            pickle.dump(episode_rewards, f)
        
        with open("ddpg_avg_rewards.obj", "wb") as f: 
            pickle.dump(avg_rewards, f)
        
        x = [i+1 for i in range(len(episode_rewards))]
        plot_learning_curve(x, episode_rewards, "maddpg_simple_adv")

       return episode_rewards, avg_rewards, best_reward

    except Exception as error: 
        raise error

if __name__ == "__main__": 
    episode_rewards, avg_rewards, best_reward = main(env, actors_input_dims, env_obs_dims, n_actions)