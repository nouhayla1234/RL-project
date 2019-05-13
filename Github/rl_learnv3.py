import gym
import gym_foo

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

import os
import numpy as np
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

# Changeable variables
islearning = True
number_steps = 3000
name_save = "deepq_pump_weird"



# Create log dir
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env_name = 'foo-v1'
env = gym.make(env_name)
# Logs will be saved in log_dir/monitor.csv
env = Monitor(env, log_dir, allow_early_resets=True)


best_mean_reward, n_steps = -np.inf, 0

list_y = []

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        print(load_results(log_dir))
        print(y)
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    # Returning False will stop training early
    return True


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()


def DQN_learn_save():
    model = DQN(MlpPolicy, env, exploration_fraction=0.3, verbose=1)
    print("Creating Model %s..." % name_save)
    print("Learning for %s steps..." % number_steps)
    model.learn(total_timesteps=number_steps, callback=callback)
    model.save(name_save)
    print("Model saved as " + name_save)
    plot_results(log_dir)
    return(model)


def DQN_load():
    model = DQN.load(name_save)
    print("Model %s loaded." % name_save)
    return(model)


def main():

    if islearning:
        model = DQN_learn_save()
    else:
        model = DQN_load()

    max_episodes = env.nb_tests #numbers of lines in csvData
    active_row = env.active_row
    csvData = env.csvData
    obs = env.reset()

    pbar = tqdm([i for i in range(max_episodes)])

    list_rewards = []
    list_rewards_ = []
    list_accuracy = []

    while active_row < max_episodes:
        env.randomize_demand_csv(active_row)
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

        list_rewards.append([rewards, action, obs, _states])
        list_rewards_.append(rewards)

        genetic_reward = float(csvData[active_row][0])

        list_accuracy.append(round(rewards/genetic_reward,2))

        pbar.update()
        pbar.set_description("Computing episode %i" % (active_row+1))
        active_row += 1
        env.reset()

    list_rewards_sorted = sorted(list_rewards, key=lambda tup: tup[0], reverse=True)

    #print(list_rewards_)
    print("Best reward : ", max(list_rewards_))
    print("Worst reward : ", min(list_rewards_))
    print("Average reward : ", np.mean(list_rewards_))

    
    title = "RL compared to the Genetic Algorithm test"
    # Create target Directory if don't exist
    result_dir = "accuracy_results"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    name_graph = result_dir + "/accuracy_score_%s.png" % name_save

    fig = plt.figure(title)
    plt.plot(list_accuracy)
    plt.xlabel('Test number')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.savefig(name_graph)
    print("Graph saved as %s." % name_graph[len(result_dir)+1:])
    plt.show()


if __name__ == '__main__':
    main()