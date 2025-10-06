import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

moves = np.array(['left', 'down', 'right', 'up'])

class OnPolicyMCControl():
    discount = 0.95
    epsilon = 0.10
    env = 0

    def __init__(self, env):
        self.env = env
        self.total_states = env.observation_space.n
        self.total_actions = env.action_space.n
        self.Q = np.random.rand(self.total_states, self.total_actions)
        self.returns = [[[] for _ in range(self.total_actions)] for _ in range(self.total_states)]
        self.pi = np.random.rand(self.total_states, self.total_actions)
        for i in range(self.total_states):
            self.pi[i] = self.pi[i] / np.sum(self.pi[i]) #normalize so everything adds up to one
        self.returns_over_time = [0]
        self.mean_returns = []
        self.iteration = 0
        self.iterations = []
        self.values = [0]
        self.episode = []

    def generate_episode(self): #follow current pi till the end state and record it
        done = False #if we're done with episode
        illegal_move = False
        curr_state, _ = self.env.reset() #FIXME
        # print("WE ARE STARTING IN STATE: ", curr_state)
        self.episode = []
        while not (done or illegal_move):
            move = np.random.choice(self.total_actions, p=self.pi[curr_state])
            # print("PICKING MOVE", move)
            self.episode.append(curr_state) #add S0
            self.episode.append(move) #add action
            curr_state, R, done, illegal_move, _ = self.env.step(move)
            self.episode.append(R) #append reward from move
            # if done:
                # print("WE ARE DONE")
        # print(self.episode)
        self.episode = np.array(self.episode)

    def inverse_loop(self):
        G = 0
        self.episode = self.episode.reshape(int(len(self.episode)/3), 3)
        for i, SAR in enumerate(reversed(self.episode)): #each iteration is time step loop
            state = int(SAR[0])
            action = int(SAR[1])
            reward = SAR[2]
            G = self.discount * G + reward
            self.iterations.append(self.iteration)
            self.values.append(G)
            self.iteration = self.iteration + 1
            if not self.exists(state, action, (self.episode.shape[0]-1) - i): #check if it existed at earlier timestep since we going in reverse order (only consider first occurance of SA pair)
                self.returns[state][action].append(G)
                self.Q[state][action] = np.average(self.returns[state][action])
                best_action = np.argmax(self.Q[state])
                for j in range(self.total_actions):
                    if j == best_action:
                        self.pi[state][j] = 1 - self.epsilon + (self.epsilon/self.total_actions)
                    else:
                        self.pi[state][j] = (self.epsilon/self.total_actions)
        self.returns_over_time.append(G)
        self.mean_returns.append(np.mean(self.returns_over_time))

    def exists(self, state, action, timestep):
        for i, row in enumerate(self.episode):
            if i == timestep:
                break
            if state == row[0] and action == row[1]:
                return True
        return False
    
    def graph(self):
        plt.plot(self.iterations, self.values, marker='o')  # line plot with markers
        plt.xlabel("Iteration")
        plt.ylabel("Mean Value of Value Function")
        plt.title("Value Iteration")
        plt.grid(True)
        plt.show()

    def graph_returns(self):
        plt.plot(range(len(self.mean_returns)), self.mean_returns)  # line plot with markers
        plt.xlabel("Iteration")
        plt.ylabel("Mean Value of Return")
        plt.title("Returns")
        plt.grid(True)
        plt.show()

class SARSA():
    alpha = 0.2
    discount = 0.95
    epsilon = 0.3
    epsilon_min = 0.01
    epsilon_decay = 0.99
    env = 0

    def __init__(self, env):
        self.env = env
        self.total_states = env.observation_space.n
        self.total_actions = env.action_space.n
        self.Q = np.random.rand(self.total_states, self.total_actions)/100
        self.Q[15] = 0
        self.pi = np.random.rand(self.total_states, self.total_actions)
        for i in range(self.total_states):
            self.pi[i] = self.pi[i] / np.sum(self.pi[i]) #normalize so everything adds up to one
        self.counts = np.zeros((self.total_states, self.total_actions))
        self.iteration = 0
        self.iterations = []
        self.avg_values = [[] for _ in range(16)]
        self.returns_over_time = [0]
        self.mean_returns = []

    def episode(self):
        curr_state, _ = self.env.reset()
        move = np.random.choice(self.total_actions, p=self.pi[curr_state])
        done = False
        illegal_move = False
        total_return = 0
        rewards = []
        while not (done or illegal_move):
            s_prime, R, done, illegal_move, _ = self.env.step(move)
            rewards.append(R)
            move_prime = np.random.choice(self.total_actions, p=self.pi[s_prime])
            self.Q[curr_state][move] = self.Q[curr_state][move] + self.alpha * (R + self.discount*self.Q[s_prime][move_prime] - self.Q[curr_state][move])
            self.counts[curr_state][move] += 1
            self.iterations.append(self.iteration)
            for i in range(16):
                self.avg_values[i].append(np.round(np.average(self.Q[i]), 5))
            self.iteration += 1
            curr_state = s_prime
            move = move_prime
            self.epsilon_greedy_policy()
        total_return = 0
        for r in reversed(rewards):
            total_return = self.discount * total_return + r
        self.returns_over_time.append(total_return)
        self.mean_returns.append(np.mean(self.returns_over_time))

    def epsilon_greedy_policy(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.pi = np.ones((self.total_states, self.total_actions)) * (self.epsilon / self.total_actions)
        best_actions = np.argmax(self.Q, axis=1)
        for s in range(self.total_states):
            self.pi[s, best_actions[s]] += (1.0 - self.epsilon)
    
    def graph(self):
        plt.plot(self.iterations, self.values, marker='o')  # line plot with markers
        plt.xlabel("Iteration")
        plt.ylabel("Mean Value of Value Function")
        plt.title("Value Iteration")
        plt.grid(True)
        plt.show()

    def graph_returns(self):
        plt.plot(range(len(self.mean_returns)), self.mean_returns)  # line plot with markers
        plt.xlabel("Iteration")
        plt.ylabel("Mean Value of Return")
        plt.title("Returns")
        plt.grid(True)
        plt.show()

class Q_learning():
    alpha = 0.2
    discount = 0.95
    epsilon = 0.3
    epsilon_min = 0.01
    epsilon_decay = 0.99
    env = 0

    def __init__(self, env):
        self.env = env
        self.total_states = env.observation_space.n
        self.total_actions = env.action_space.n
        self.Q = np.random.rand(self.total_states, self.total_actions)/100
        self.Q[15] = 0
        # print(self.Q)
        self.pi = np.random.rand(self.total_states, self.total_actions)
        for i in range(self.total_states):
            self.pi[i] = self.pi[i] / np.sum(self.pi[i]) #normalize so everything adds up to one
        self.counts = np.zeros((self.total_states, self.total_actions))
        self.iteration = 0
        self.iterations = []
        self.avg_values = [[] for _ in range(16)]
        self.returns_over_time = [0]
        self.mean_returns = []

    def episode(self):
        curr_state, _ = self.env.reset()
        done = False
        illegal_move = False
        total_return = 0
        rewards = []
        step = 0
        self.epsilon_greedy_policy()
        while not (done or illegal_move):
            move = np.random.choice(self.total_actions, p=self.pi[curr_state])
            s_prime, R, done, illegal_move, _ = self.env.step(move)
            # if s_prime == 15:
            #     print("we have reached the end")
            rewards.append(R)
            self.Q[curr_state][move] += self.alpha * (R + self.discount*np.max(self.Q[s_prime]) - self.Q[curr_state][move])
            self.counts[curr_state][move] += 1
            self.iterations.append(self.iteration)
            for i in range(16):
                self.avg_values[i].append(np.round(np.average(self.Q[i]), 5))
            self.iteration += 1
            curr_state = s_prime
            self.epsilon_greedy_policy()
        total_return = 0
        for r in reversed(rewards):
            total_return = self.discount * total_return + r
        self.returns_over_time.append(total_return)
        self.mean_returns.append(np.mean(self.returns_over_time))


    def epsilon_greedy_policy(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.pi = np.ones((self.total_states, self.total_actions)) * (self.epsilon / self.total_actions)
        best_actions = np.argmax(self.Q, axis=1)
        for s in range(self.total_states):
            self.pi[s, best_actions[s]] += (1.0 - self.epsilon)
    
    def graph(self):
        # plt.plot(self.iterations, self.values, marker='o')  # line plot with markers
        for i in range(16):
            plt.plot(self.iterations, self.avg_values[i], label=f"Line {i}")
            # print(self.avg_values[i])
        plt.xlabel("Iteration")
        plt.ylabel("Mean Value of Value Function")
        plt.title("Value Iteration")
        plt.grid(True)
        plt.show()

    def graph_returns(self):
        plt.plot(range(len(self.mean_returns)), self.mean_returns)  # line plot with markers
        plt.xlabel("Iteration")
        plt.ylabel("Mean Value of Return")
        plt.title("Returns")
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    custom_map = [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ]

    env = gym.make('FrozenLake-v1', desc=custom_map, map_name="4x4", is_slippery=False)
    np.set_printoptions(suppress=True, precision=4)
    # MCdonalds = OnPolicyMCControl(env)
    # while(max(MCdonalds.returns_over_time) == 0):
    #     MCdonalds = OnPolicyMCControl(env)
    #     for i in range(1000):
    #         MCdonalds.generate_episode()
    #         MCdonalds.inverse_loop()
    
    # # print(moves[np.argmax(np.round(MCdonalds.pi, 2), axis=1)].reshape(4,4))
    # MCdonalds.graph_returns()
    # print(MCdonalds.Q)

    # print(MCdonalds.pi)
    # print(MCdonalds.iteration)
    # MCdonalds.graph()

    salsa = SARSA(env)
    while(max(salsa.returns_over_time) == 0):
        salsa = SARSA(env)
        for i in range(2000):
            salsa.episode()

    # print(moves)
    # print(salsa.pi)
    print(np.sum(salsa.counts, axis=1).reshape(4,4), "\n")
    # salsa.graph_returns()
    print(moves[np.argmax(np.round(salsa.pi, 2), axis=1)].reshape(4,4), "\n")
    print(salsa.Q)

    agent_q = Q_learning(env)
    while(max(agent_q.returns_over_time) == 0):
        agent_q = Q_learning(env)
        for i in range(2000):
            agent_q.episode()

    # agent_q.graph_returns()
    print(moves[np.argmax(np.round(agent_q.pi, 2), axis=1)].reshape(4,4), "\n")
    print(np.sum(agent_q.counts, axis=1).reshape(4,4))
    print(agent_q.Q)