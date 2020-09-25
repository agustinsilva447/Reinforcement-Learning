import scigym
import numpy as np

env = scigym.make('teleportation-v0')
if isinstance(env.action_space,scigym.Discrete):
    num_actions = env.action_space.n
else:
    raise ValueError

actions = range(num_actions)
observation = env.reset()
action = np.random.choice(actions)
i = 0
done = False
print("Cantidad de acciones:", actions)
print("Obsevation 0: {}".format(observation))
print("Action 0: {}".format(action))

while done==False or i>100:
    i += 1  
    (observation, reward, done, _) = env.step(action)
    print("----------")
    print("Obsevation {}: {}".format(i, observation))
    print("Reward {}: {}".format(i, reward))
    print("Done {}: {}".format(i, done))
    print("Info {}: {}".format(i, _['available_actions']))
    if bool(_['available_actions']):
        action = np.random.choice(_['available_actions'])
        print("Action {}: {}".format(i, action))
    else:
        print("No more available actions")