import scigym
import numpy as np

env = scigym.make('teleportation-v0')
if isinstance(env.action_space,scigym.Discrete):
    num_actions = env.action_space.n
else:
    raise ValueError

actions = range(num_actions)
observation = env.reset()
print("Cantidad de acciones:", actions)
print("Obsevation 0:", observation)
print("----------")

for i in range(1,5):
    action = np.random.choice(actions)
    (observation, reward, done, _) = env.step(action)
    print("Obsevation {}: {}".format(i, observation))
    print("Reward {}: {}".format(i, reward))
    print("Done {}: {}".format(i, done))
    print("Info {}: {}".format(i, _))
    print("Action {}: {}".format(i, action))
    print("----------")