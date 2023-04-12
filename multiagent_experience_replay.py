import numpy as np

class MultiAgentReplayBuffer:
    def __init__(self, max_size, env_obs_dims, actor_dims, 
                                        n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.env_obs_memory = np.zeros((self.mem_size, env_obs_dims))
        self.env_next_obs_memory = np.zeros((self.mem_size, env_obs_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actors_state_memory = []
        self.actors_new_state_memory = []
        self.actors_action_memory = []

        for i in range(self.n_agents):
            self.actors_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actors_new_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actors_action_memory.append(
                            np.zeros((self.mem_size, self.n_actions)))


    def store_experiences(self, actor_states, env_obs, actors_actions, reward, 
                                                   actor_next_states, env_next_obs, done):

        index = self.mem_cntr % self.mem_size

        for agent_idx in range(self.n_agents):
            self.actors_state_memory[agent_idx][index] = actor_states[agent_idx]
            self.actors_new_state_memory[agent_idx][index] = actor_next_states[agent_idx]
            self.actors_action_memory[agent_idx][index] = actors_actions[agent_idx]

        self.env_obs_memory[index] = env_obs
        self.env_next_obs_memory[index] = env_next_obs
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_experiences(self):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        env_obs = self.env_obs_memory[batch]
        rewards = self.reward_memory[batch]
        env_next_obs = self.env_next_obs_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actor_new_states = []
        actors_actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actors_state_memory[agent_idx][batch])
            actor_new_states.append(self.actors_new_state_memory[agent_idx][batch])
            actors_actions.append(self.actors_action_memory[agent_idx][batch])
        
        #actor_states, actor_new_states, actors_actions = np.array(actor_states), np.array(actor_new_states), np.array(actors_action)
        return actor_states, env_obs, actors_actions, rewards, \
               actor_new_states, env_next_obs, terminal

    def is_sufficient(self):
        if self.mem_cntr >= self.batch_size:
            return True