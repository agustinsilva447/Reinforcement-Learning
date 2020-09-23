import scigym
import numpy as np

env = scigym.make('surfacecode-decoding-v0')

if isinstance(env.action_space,scigym.Discrete):
    num_actions = env.action_space.n
else:
    raise ValueError

actions = range(num_actions)

observation = env.reset()

action = np.random.choice(actions)

done = False
while not done:
    (observation, reward, done) = env.step(action)
    action = np.random.choice(actions)