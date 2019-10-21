import gym
import argparse

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.deepq.policies import CnnPolicy as DqnCnnPolicy
from stable_baselines.common.policies import CnnPolicy
from stable_baselines import DQN, A2C, SAC, PPO2
from stable_baselines.sac.policies import CnnPolicy as SacCnnPolicy
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
from envs.doom_env import DoomEnv
from skimage.transform import resize
import time
import os
import sys
import tensorflow as tf
import numpy as np
import json
import random
from shutil import copy2
from distutils.dir_util import copy_tree
from utils_for_log import custom_ts2xy

## Cv2 import done this way for computer with ROS
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append(ros_path)

### Global variables ###
best_mean_reward, n_steps = -np.inf, 0
best_mean_length = 100
last_total_steps = 0
log_path = "" #Initialized in run_experiment
checkpoint_path = "" #Initialized in run_experiment
SAVE_MODEL_EVERY_ENV_STEPS = 0 #Initialized in run_experiment


# Saves a copy of these files and directories when the training starts 
files_to_copy = ["results_plotter.py", "vizdoom_wrapper.py", 
"utils_for_log.py", "log_parser.py", "doom_baselines_agents.py", "doomturtle.py", "eval_doom_baselines.py",
"plot_eval_doom_baselines.py"]
directories_to_copy = ["envs/", "scenarios/", "stable_baselines/"]

def load_network_params(model, path, restore_value_function = False, load_last_fc = False):
    """ Loads parameters to new network from the source model. Currently used with PPO2.

        model: Model to which parameters should be loaded.
        path: (String) Path to .pkl file where parameters of the source model are saved
        resotre_value_function: (Boolean) Select if value function paremeters should be loaded
        load_last_fc: Load last fc-layer parameters. Set True if you want to use adapter method.

        """
    data, params = BaseRLModel._load_from_file(path)
    restores = []

    if restore_value_function:
        print("Restoring old network weights and value function")
    else:
        print("Restoring old network weights")

    param_list = list(filter(lambda param: "adapter" not in param.name, model.params))

    for param, loaded_p in zip(param_list, params):
        if "base_nn_param" in param.name:
            print(param.name)
            restores.append(param.assign(loaded_p))
        elif ("model/q" in param.name or "model/pi" in param.name) and load_last_fc:
            print(param.name)
            restores.append(param.assign(loaded_p))
        elif "vf" in param.name and restore_value_function:
            print(param.name)
            restores.append(param.assign(loaded_p))
           
    model.sess.run(restores) # Sets the parameters
    print("Succesfully loaded previous network weights from file.")

def load_dqn_network_params(model, path, restore_value_function = False, load_last_fc = False):
    """ Loads parameters to new network from the source model. Currently used with DQN.

    model: Model to which parameters should be loaded.
    path: (String) Path to .pkl file where parameters of the source model are saved
    resotre_value_function: (Boolean) Select if value function paremeters should be loaded
    load_last_fc: Load last fc-layer parameters. Set True if you want to use adapter method.
    """

    data, params = BaseRLModel._load_from_file(path)
    restores = []
    #text_file = open("Weights1.txt", "a")
    if restore_value_function:
        print("Restoring old network weights and value function")
    else:
        print("Restoring old network weights")

    param_list = list(filter(lambda param: "adapter" not in param.name, model.params))

    for param, loaded_p in zip(param_list, params):
        print("param name:",param.name)
        if "action_value" in param.name and "fully_connected" not in param.name:
            restores.append(param.assign(loaded_p))
        elif "action_value" in param.name and "fully_connected" in param.name and load_last_fc:
            restores.append(param.assign(loaded_p))
        if "state_value" in param.name and restore_value_function:
            restores.append(param.assign(loaded_p))

    print("Restores:")
    for restore in restores:
        print(restore)
    model.sess.run(restores)
    print("Succesfully loaded previous network weights from file.")

def get_log_info(timesteps):
    return np.cumsum(timesteps.l.values), timesteps.r.values, timesteps.l.values

def custom_callback(_locals, _globals):
    """ Custom callback to save best and latest models"""
    global n_steps, last_total_steps, best_mean_reward, best_mean_length, log_path, checkpoint_path, SAVE_MODEL_EVERY_ENV_STEPS
    ##Check if we have model with best reward and length
    x, rewards, lengths = get_log_info(load_results(log_path))
    if len(x) > 10 and n_steps > 10:

        # Checking if new best mean reward
        mean_reward = np.mean(rewards[-100:])
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            print("Saving new best reward model", round(mean_reward,2))
            _locals['self'].save(checkpoint_path + 'best_model_reward.pkl') #_' + str(time_step) + '

        #Checking if new best mean length
        mean_length = np.mean(lengths[-100:])
        if mean_length < best_mean_length:
            best_mean_length = mean_length
            print("Saving new best episode_length model:", round(mean_length,0))
            _locals['self'].save(checkpoint_path + 'best_model_episode_length.pkl')

        #Constant saving of the models for later evaluation
        if SAVE_MODEL_EVERY_ENV_STEPS != 0:
            total_steps = x[-1]
            if (total_steps - last_total_steps) > SAVE_MODEL_EVERY_ENV_STEPS:
                print("Saving checkpoint model:", total_steps)
                _locals['self'].save(checkpoint_path + ("snapshot_model_steps_%d.pkl" % total_steps))
                last_total_steps = total_steps
    
    n_steps += 1
    return True

def save_args(path, args):
    """ Saves arguments in json file. """
    json_path = path + "arguments.json"
    data = {}
    for arg in vars(args):
        data[arg] = getattr(args, arg)
    with open(json_path, 'w') as outfile:
        json.dump(data, outfile)

def copy_files(destination):
    """ Copies all the specified files in the beginning of the training to the experiment directory. """
    for file in files_to_copy:
        try:
            copy2(file, destination)
        except FileNotFoundError:
            print("Couldn't copy file", file)

    for directory in directories_to_copy:
        try:   
            copy_tree(directory, destination + "/" + directory)
        except Exception:
            print("Couldn't copy directory", directory)


def run_experiment(args):
    global log_path, checkpoint_path, SAVE_MODEL_EVERY_ENV_STEPS

    train = args.mode == "train"
    backend = "TensorFlow"

    if train or args.save_training: 
        save_training = True #Used for logging and turtlebot parameter saver
    else:
        save_training = False

    if args.environment == "":
        raise Exception("Choose environment with --environment")
    elif args.environment == "turtlebot":
        from envs.turtlebot_env import TurtlebotEnv

    ### Parameters ###
    if args.time_steps != 0:
        time_steps = args.time_steps
    elif args.environment == "sim-to-sim":
        time_steps = 200000
    elif args.environment == "turtlebot":
        time_steps = 20000
    else:
        time_steps = 1000000

    if args.number_of_checkpoints == 0:
        SAVE_MODEL_EVERY_ENV_STEPS = 0
    else:
        SAVE_MODEL_EVERY_ENV_STEPS = int(time_steps)/args.number_of_checkpoints

    print("Saving model between ", SAVE_MODEL_EVERY_ENV_STEPS, "steps")
    
    verbose = 1
    checkpoint_freq = 1
    batch_size = 128
    ent_coef = args.ent_coef

    ### Define action space ###
    if args.action_space != "":
        action_space = args.action_space
    elif args.algorithm == "DQN" and args.environment == "sim-to-sim" or args.environment == "turtlebot":
        action_space = "semi-continuous"
    elif args.environment == "sim-to-sim":
        action_space = "continuous"
    elif args.environment == "doom":
        action_space = "discrete"

    ### Paths and logs ###
    day_month = time.strftime("%d-%m - %H-%M-%S")

    if args.rootdir != "":
        rootdir = args.rootdir
    else:
        rootdir = "./saved_trainings/"

    if args.save_dir != "":
        save_directory = rootdir + "new_experiments/" + args.save_dir + "/" +args.algorithm + "_" + args.environment + " " + day_month + "/"
    else:
        save_directory = rootdir + "new_experiments/" + args.algorithm + "_" + args.environment + " " + day_month + "/"

    checkpoint_path = save_directory + "best_model/"
    log_path = save_directory + "/log/"
    codes_path = save_directory + "/codes/"

    if args.tensorboard_log:
        tensorboard_log_path = save_directory + "/tb_log/"
    else:
        tensorboard_log_path = None

    if save_training:
        ### Creating paths if not existing ###
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        if not os.path.exists(codes_path):
            os.makedirs(codes_path)

        ### Saving info and codes ###
        save_args(save_directory, args)
        copy_files(codes_path)
        #sys.stdout = open(save_directory + 'console_log.txt', 'w')

    ### Create env ###
    if args.environment == "sim-to-sim":
        env_non_vectorized = DoomEnv(action_space = action_space, visualize = args.visualize, backend = backend, sim_to_sim = 1, mode=args.mode, discretes=args.discretes)
    elif args.environment == "doom":
        env_non_vectorized = DoomEnv(action_space = action_space, visualize = args.visualize, backend = backend, sim_to_sim = 0, mode=args.mode, discretes=args.discretes, timeout=args.timeout, randomize=args.domain_randomization)
    elif args.environment == "turtlebot":
        env_non_vectorized = TurtlebotEnv(root_dir=save_directory, action_space = action_space, visualize = args.visualize, save_training=save_training)

    if save_training:
        ### Setting Monitor to log the training
        env_non_vectorized = Monitor(env_non_vectorized, log_path, allow_early_resets=True)

    ### Vectorizing env ###
    env = DummyVecEnv([lambda: env_non_vectorized])

    if args.replace:
        adapter_old_action_space = None #None value if we use replace layer instead of adapter layer
        freeze_last_fc = False
        load_last_fc = False
    else:
        adapter_old_action_space = gym.spaces.Discrete(4)
        freeze_last_fc = True
        load_last_fc = True

    ### Different reinforcement learning algorithms ###
    # DQN
    if args.algorithm == "DQN":
        if train:
            buffer_size = 50000
            train_freq = 4
            target_network_update_freq = 5000
            if args.environment == "sim-to-sim" or args.environment == "turtlebot":
                model = DQN(DqnCnnPolicy, 
                    env, 
                    verbose=verbose, 
                    learning_rate=args.dqn_learning_rate,
                    buffer_size = buffer_size,
                    batch_size = batch_size,
                    checkpoint_freq = checkpoint_freq,
                    checkpoint_path = checkpoint_path,
                    tensorboard_log = tensorboard_log_path,
                    train_freq = train_freq,
                    target_network_update_freq = target_network_update_freq,
                    freeze_base_nn=args.freeze_base_nn,
                    freeze_vf=args.freeze_vf,
                    freeze_last_fc=freeze_last_fc,
                    exploration_fraction = args.exploration_fraction,
                    policy_kwargs={"adapter_old_action_space": adapter_old_action_space}
                    )
                if args.load_params_from_previous_model:
                    load_dqn_network_params(model, args.source_model, args.restore_value_function, load_last_fc)
            elif args.environment == "doom":
                model = DQN(DqnCnnPolicy, 
                    env, 
                    verbose=verbose, 
                    learning_rate=args.dqn_learning_rate,
                    buffer_size = buffer_size,
                    batch_size = batch_size,
                    checkpoint_freq = checkpoint_freq,
                    checkpoint_path = checkpoint_path,
                    tensorboard_log = tensorboard_log_path,
                    train_freq = train_freq,
                    target_network_update_freq = target_network_update_freq,
                    exploration_fraction = args.exploration_fraction,
                    )
        else:
            model = DQN.load(args.model)

    # A2C
    elif args.algorithm == "A2C":
        if train:
            model = A2C(CnnPolicy, env, verbose=verbose, tensorboard_log=tensorboard_log_path, ent_coef=ent_coef)
        else:
            model = A2C.load(args.model)

    # PPO
    elif args.algorithm == "PPO":
        if train:
            if args.environment == "sim-to-sim":
                model = PPO2(CnnPolicy, 
                    env, 
                    verbose=verbose, 
                    tensorboard_log=tensorboard_log_path, 
                    ent_coef=ent_coef, 
                    freeze_base_nn=args.freeze_base_nn,
                    freeze_vf=args.freeze_vf,
                    freeze_last_fc=freeze_last_fc,
                    policy_kwargs={"adapter_old_action_space": adapter_old_action_space}
                    )
                if args.load_params_from_previous_model:
                    load_network_params(model, args.source_model, args.restore_value_function, load_last_fc)
            elif args.environment == "doom":
                model = PPO2(CnnPolicy, 
                    env, 
                    verbose=verbose, 
                    tensorboard_log=tensorboard_log_path, 
                    ent_coef=ent_coef,
                    )
        else:
            model = PPO2.load(args.model)

    # SAC
    elif args.algorithm == "SAC":
        if train:
            model = SAC(SacCnnPolicy, 
                env, 
                verbose=verbose,
                buffer_size = 20000,
                batch_size = batch_size,
                ent_coef = ent_coef,
                tensorboard_log=tensorboard_log_path)
        else:
            model = SAC.load(args.model)
    else:
        raise NotImplementedError("Unknown learning algorithm.")

    ### Train ###
    if train:
        model.learn(total_timesteps=time_steps, callback=custom_callback)
        model.save(save_directory + "doom_" + args.algorithm)

    ### Test ###
    else:
        testing_episodes = 200
        rewards = []
        lengths = []
        all_states = []
        all_rewards = []
        all_actions = []
        all_coordinates = []
        all_terminals = []

        obs = env.reset()
        save = args.save_parameters_in_test != ""

        if save:
            obs_saveable = (np.squeeze(obs[:,:,:,3]*255)).astype('uint8')
            all_states.append(obs_saveable)
            pos_x, pos_y, pos_z = env_non_vectorized.get_coordinates()
            all_coordinates.append(np.array([pos_x, pos_y, pos_z], dtype=np.float32))

        for i in range(0,testing_episodes):
            done = False
            j = 0
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)

                if save:
                    all_actions.append(action)
                    obs_saveable = (np.squeeze(obs[:,:,:,3]*255)).astype('uint8')
                    all_states.append(obs_saveable)
                    all_rewards.append(reward)
                    all_terminals.append(done)
                    pos_x, pos_y, pos_z = env_non_vectorized.get_coordinates()
                    all_coordinates.append(np.array([pos_x, pos_y, pos_z], dtype=np.float32))

                j += 1
                if done:
                    rewards.append(reward)
                    lengths.append(j)
                    print("Episode length:",j)

        if save:
            np.savez(args.save_parameters_in_test, states=all_states, actions=all_actions, rewards=all_rewards, coordinates=all_coordinates, terminals=all_terminals)

        print("Mean reward:",np.mean(rewards),", mean episode length:", np.mean(lengths))



if __name__ == "__main__":
    ## TODO: Most of these arguments could be specified by making a config file
    parser = argparse.ArgumentParser()

    # Common arguments
    parser.add_argument('--mode', choices=['train', 'test', 'log_from_tests'], default='test')
    parser.add_argument('--algorithm', choices=['DQN', 'A2C', 'PPO', 'SAC'], default='DQN')
    parser.add_argument('--environment', choices=['doom', 'sim-to-sim', 'turtlebot'], default='')
    parser.add_argument('--visualize', type=int, default=0)
    parser.add_argument('--model', type=str, default="") # Used when loading a model for testing
    parser.add_argument('--time_steps', type=int, default=0)
    parser.add_argument('--timeout', type=int, default=1000)
    parser.add_argument('--domain_randomization', type=int, default=1)

    # Action space
    parser.add_argument('--action_space', choices=['discrete', 'continuous', 'semi-continuous'], default='') #Default values for different algorithms/environments are defined later
    parser.add_argument('--discretes', type=str, default="wasd") # Define action space with wasd keys. I.e. "wsd"

    # Action space transfer
    parser.add_argument('--replace', type=int, default=1)
    parser.add_argument('--source_model', type=str, default="") # Used for action space transfer
    parser.add_argument('--restore_value_function', type=int, default=0)
    parser.add_argument('--freeze_base_nn', type=int, default=1) #Works only in sim-to-sim
    parser.add_argument('--freeze_vf', type=int, default=0)
    parser.add_argument('--load_params_from_previous_model', type=int, default=1)

    # PPO
    parser.add_argument('--ent_coef', type=float, default=0.001)

    # DQN
    parser.add_argument('--exploration_fraction', type=float, default=0.1)
    parser.add_argument('--dqn_learning_rate', type=float, default=0.0005)

    # Logging
    parser.add_argument('--save_training', type=int, default=0) # Default is 0 when testing and 1 when training
    parser.add_argument('--info', type=str, default="")
    parser.add_argument('--save_dir', type=str, default="") # Saves inside /saved_training/new_experiments/[save_dir]. by default, saves to /saved_training/ root.
    parser.add_argument('--number_of_checkpoints', type=int, default=50)
    parser.add_argument('--rootdir', type=str, default="") # Changes the /saved_training/ root directory for saved experiments
    parser.add_argument('--tensorboard_log', type=int, default=0)
    parser.add_argument('--save_parameters_in_test', type=str, default="")

    args = parser.parse_args()
    run_experiment(args)