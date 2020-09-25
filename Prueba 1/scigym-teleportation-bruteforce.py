import scigym
import numpy as np

j = 0; reward = 0

while (reward == 0) and (j<1000) :
    env = scigym.make('teleportation-v0')
    if isinstance(env.action_space,scigym.Discrete):
        num_actions = env.action_space.n
    else:
        raise ValueError

    i = 0; j += 1
    done = False
    observation = env.reset()
    available = {}; available['available_actions'] = range(num_actions)

    while done==False:  
        action = np.random.choice(available['available_actions'])
        i += 1
        (observation, reward, done, available) = env.step(action)
        print("{}, {} --> Reward: {}. Done: {}. Available actions: {}.".format(j, i, reward, done, available['available_actions']), end=" ")
        
        if bool(available['available_actions']):
            print("Action {}: {}.".format(i, action))
        else:
            print("No more available actions.")

    if reward == 0:
        print("Target not met.")
    elif reward == 1:
        print("Congratulations!")