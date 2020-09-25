import scigym
import numpy as np

env = scigym.make('teleportation-v0')
if isinstance(env.action_space,scigym.Discrete):
    num_actions = env.action_space.n
else:
    raise ValueError

actions = range(num_actions)
observation = env.reset()
available = {}
available['available_actions'] = range(17)
i = 0
done = False
print("Number of possible actions:", actions)
print("Obsevation 0: {}".format(observation))

while done==False or i>100:  
    print("Choose an action:")
    action = input()
    if (action.isdigit()) and (int(action) in available['available_actions']):
        action = int(action)
        i += 1
        (observation, reward, done, available) = env.step(action)
        print("----------")
        print("Obsevation {}: {}".format(i, observation))
        print("Reward {}: {}".format(i, reward))
        print("Done {}: {}".format(i, done))
        print("Available actions {}: {}".format(i, available['available_actions']))
        if bool(available['available_actions']):
            print("Action {}: {}".format(i, action))
        else:
            print("No more available actions")
    else:
        print("----------")
        print("Actions not available")
        print("Available actions {}: {}".format(i, available['available_actions']))