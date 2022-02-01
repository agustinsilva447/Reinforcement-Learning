import gym
import numpy as np

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2001
SHOW_EVERY = 1000
DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
q_table = np.random.uniform(low=-2, high=-1, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

epsilon = 1 
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

"""discrete_state = get_discrete_state(env.reset())
print(discrete_state)
print(q_table[discrete_state])
print(np.argmax(q_table[discrete_state]))"""

for episode in range(EPISODES):
    discrete_state = get_discrete_state(env.reset())
    done = False

    """    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False"""

    while not done:
        
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        """        if render:
            env.render()"""

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q    
        elif new_state[0] >= env.goal_position:
            print("We made it in episode {}.", episode)
            q_table[discrete_state + (action,)] = 0        

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value        

env.close()        