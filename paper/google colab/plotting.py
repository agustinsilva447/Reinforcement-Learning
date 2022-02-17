import pickle                      
import matplotlib.pyplot as plt

start_mean_rewards = "/home/agustinsilva447/Escritorio/Github/Reinforcement-Learning/paper/google colab/mean_rewards-1645040777.pickle"
with open(start_mean_rewards, "rb") as f:
    mean_rewards = pickle.load(f)
print(mean_rewards.shape)

start_mean_rewards_avg = "/home/agustinsilva447/Escritorio/Github/Reinforcement-Learning/paper/google colab/mean_rewards_avg-1645040777.pickle"
with open(start_mean_rewards_avg, "rb") as f:
    mean_rewards_avg = pickle.load(f)
print(mean_rewards_avg.shape)

labels = [
    'e = 0.1, a = 0.1',
    'e = decay, a = 0.1', 
    'e = 0.1, a = 1/n', 
    'e = decay, a = 1/n', 
    'gradient ascent'
]

perf_ideal_1 = 37.4592
perf_ideal_2 = 16.4807
perf_network            = 100 * perf_ideal_1 / mean_rewards
perf_network[:,513:770] = 100 * perf_ideal_2 / mean_rewards[:,513:770]
perf_network_avg            = 100 * perf_ideal_1 / mean_rewards_avg
perf_network_avg[:,513:770] = 100 * perf_ideal_2 / mean_rewards_avg[:,513:770]

fig, axs = plt.subplots(2, 1, figsize=(30,20))
for i in range(len(labels)):
    axs[0].plot(perf_network[i], label=labels[i])
    axs[1].plot(perf_network_avg[i], label=labels[i])

axs[0].set_title("Learning quantum strategiesin a Network Routing Environment")
axs[0].set_ylabel("Total Time")
axs[1].set_ylabel("Mean Time")
axs[1].set_xlabel("Episodes")

#axs[0].legend()
axs[1].legend()
plt.show()