import argparse
import numpy as np
import matplotlib
from matplotlib import pyplot
from glob import glob
import os
from collections import OrderedDict
import seaborn

seaborn.set()

parser = argparse.ArgumentParser("Plot set of evaluation results.")
parser.add_argument('results', help="Log files to plot", nargs="+")
parser.add_argument('--lengths', help="Plot lengths instead of rewards", action="store_true")
parser.add_argument('--plot_style', choices=['mean', 'paper'], default='mean')
parser.add_argument('output', help="Output for the graphics")

INTEROP_POINTS = 300
STEPS_FOR_SUMMARY = 5000

# If True, turn rewards into success rates
AS_SUCCESS_RATE = True

# How many eval-points should be included
# in the "final performance" summary
LAST_EVALS_TO_INCLUDE_IN_SUMMARY = 5

def read_data(filename):
    """ Read csv data into numpy array """
    return np.genfromtxt(filename, delimiter=",", 
                        names=True, skip_header=1)

def read_eval_curve(logs):
    """ 
    Load one evaluation curve from given files,
    which should be paths to snapshots of the model
    taken during training.

    Returns 
        - timesteps
        - rewards
        - episode_lengths
        - raw datas in ascending order

    Note: Assumes specific type of names
    """
    steps = list(map(lambda x: int(x.split(".pkl")[0].split("_steps_")[-1]), logs))
    datas = [read_data(x) for x in logs]

    # Turn {-1,1} into {0,1}
    if AS_SUCCESS_RATE:
        for i in range(len(datas)):
            datas[i]["r"] = np.clip(datas[i]["r"], 0, 1)

    mean_rewards = [x["r"].mean() for x in datas]
    mean_lengths = [x["l"].mean() for x in datas]

    # Sort ascending
    steps, mean_rewards, mean_lengths, datas = zip(*sorted(
        zip(steps, mean_rewards, mean_lengths, datas), 
        key=lambda x: x[0])
    )

    return np.array(steps), np.array(mean_rewards), np.array(mean_lengths), datas

def get_eval_curves(logdir, return_raw=False):
    """
    Go through given directory recursively,
    find all possible evaluation logs of snapshots, 
    load their data and return as N x D arrays.
    If return_raw==True, returns N x #Evals list of numpy arrays from monitor files

    Mainly designed for returning results of repeated experiments
    """

    all_steps = []
    all_rewards = []
    all_lengths = []
    all_datas = []
    num_snapshots = []
    for dirpath, dirnames, filenames in os.walk(logdir):
        snapshot_files = list(filter(lambda x: "snapshot_model" in x and ".csv" in x, filenames))
        # If we have enough snapshot files
        if len(snapshot_files) > 10:
            steps, rewards, lengths, datas = read_eval_curve(list(map(lambda x: os.path.join(dirpath, x), snapshot_files)))
            all_steps.append(steps)
            all_rewards.append(rewards)
            all_lengths.append(lengths)
            num_snapshots.append(len(steps))
            all_datas.append(datas)

    # Check if all runs had same number of experiments
    if not min(num_snapshots) == max(num_snapshots):
        if max(num_snapshots) - min(num_snapshots) > 2:
            print("[WARNING] Different number of snapshots with", logdir, "\n",
                  "         Cutting to shortest")
            print("Original number of snapshots:", num_snapshots)

        min_len = min(num_snapshots)
        for i in range(len(all_steps)):
            all_steps[i] = all_steps[i][:min_len]
            all_rewards[i] = all_rewards[i][:min_len]
            all_lengths[i] = all_lengths[i][:min_len]
            all_datas[i] = all_datas[i][:min_len]

    if return_raw:
        return all_datas

    # Linearly interpolate all points to new x-points
    new_steps = np.arange(min(min(x) for x in all_steps),
                          max(max(x) for x in all_steps),
                          INTEROP_POINTS)
    for i in range(len(all_steps)):
        all_rewards[i] = np.interp(new_steps, all_steps[i], all_rewards[i])
        all_lengths[i] = np.interp(new_steps, all_steps[i], all_lengths[i])
        all_steps[i] = new_steps

    return np.stack(all_steps), np.stack(all_rewards), np.stack(all_lengths) 

def get_mean_std_curves(logdir):
    """
    Returns means/stds of rewards/ep_lengths from
    multiple experiments (logdir contains directories, one per experiment run)
    """
    steps, rewards, lengths = get_eval_curves(logdir)

    # Episode steps do not match one-to-one, so we take
    # mean of them
    steps_mean = steps.mean(0)

    rewards_mean = rewards.mean(0)
    rewards_std = rewards.std(0)
    
    lengths_mean = lengths.mean(0)
    lengths_std = lengths.std(0)

    return steps_mean, rewards_mean, rewards_std, lengths_mean, lengths_std

def plot_experiment_curve(experiment_dir, 
                          plot_lengths=False, 
                          print_final_result=True,
                          plot_std=True,
                          plotter=pyplot):
    """
    Plot one experiment curve by taking means etc.
    If lengths==True, plot episode lengths instead of rewards
    If print_final_result==True, take average value from last ~5000 steps
    and print it out. 
    """

    steps, rewards, r_std, lengths, l_std = get_mean_std_curves(experiment_dir)

    if not plot_lengths:
        plotter.plot(steps, rewards)
        if plot_std:
            plotter.fill_between(steps, np.clip(rewards-r_std, -1, 1),
                                np.clip(rewards+r_std, -1, 1), alpha=0.2)
    else:
        plotter.plot(steps, lengths)
        if plot_std:
            plotter.fill_between(steps,
                                np.clip(lengths-l_std, 0, None),
                                np.clip(lengths+l_std, 0, None), 
                                alpha=0.2)

    if print_final_result:
        # Get raw data per repetition, get individual
        # rewards from final evaluations, and mean/std over them
        raw_datas = get_eval_curves(experiment_dir, return_raw=True)

        value_name = "l" if plot_lengths else "r"

        final_values = []
        # From all experiments, gather values from N last eval-points
        # for final summary
        for i in range(1,LAST_EVALS_TO_INCLUDE_IN_SUMMARY+1):
            final_values.append(np.concatenate([x[-i][value_name] for x in raw_datas]))
        final_values = np.stack(final_values)

        mean = 0
        std = 0
        # Compute mean and conf. interval separately 
        if not plot_lengths:
            # Turn into bernoulli
            final_values = np.clip(final_values, 0, 1)
            # this equals to success probability
            mean = final_values.mean()
            # 95% conf. interval (based on normal approximation)
            std = 1.96 * np.sqrt((mean*(1-mean))/len(final_values))
        else:
            print("NotImplementedError")
            return
            #raise NotImplementedError

        print(experiment_dir, "summary: %.3f ± %.3f" % (mean, std))
        # This was old summary calculator:
        #sum_idxs = steps >= (max(steps)-STEPS_FOR_SUMMARY)
        #values = lengths if plot_lengths else rewards
        #summary_values = values[sum_idxs].mean()
        # TODO is this correct std?
        #summary_std = values[sum_idxs].std()
        #print(experiment_dir, "summary: %.3f ± %.3f" % (summary_values, summary_std))

def main_means(args):
    _ = pyplot.figure(dpi=150)
    legends = []

    # For each experiment, plot one curve
    for experiment_dir in args.results:
        # Remove trailing /
        if experiment_dir.endswith("/"): experiment_dir = experiment_dir[:-1]
        plot_experiment_curve(experiment_dir, args.lengths)
        legends.append(os.path.basename(experiment_dir))

    pyplot.legend(legends)
    if args.lengths:
        pyplot.ylabel("Average episode lengths")
    else:
        pyplot.ylabel("Average episode reward")
        pyplot.ylim(-1.1, 1.1)
    pyplot.xlabel("Environment steps")
    
    pyplot.savefig(args.output)

# TODO hardcoded
# Maps code names to names we want in paper,
# and also defines which experiments we want to plot
INCLUDED_EXPERIMENTS = OrderedDict((
    ("baseline_no_loading", "Scratch"),
    ("sim-to-sim_qf_vf_trainable", "Fine-tune"),
    ("sim-to-sim_vf_trainable", "Loaded value function"),
    ("sim-to-sim", "Replace"),
    ("sim-to-sim_adapter", "Adapter"),
))

def get_filtered_experiments(directory):
    """
    Returns wanted experiments from the directory, and 
    renames them according to dictionary above.
    Returns OrderedDictionary of nice_name -> directory_name
    """
    experiments = glob(os.path.join(directory, "*"))
    filtered_experiments = []
    for included_experiment in INCLUDED_EXPERIMENTS.keys():
        for experiment in experiments:
            if experiment.strip("/").endswith(included_experiment):
                filtered_experiments.append([INCLUDED_EXPERIMENTS[included_experiment], experiment])
                break

    return OrderedDict(filtered_experiments)

def xtick_formatter(x, pos):
    """
    A formatter used in main_paper function to divide
    ticks by specific amount.
    From Stackoverflow #27575257
    """
    s = '%d' % int(x / 1000)
    return s

def main_paper(args):
    # Inputs is a set of baseline dirs, containing
    # number of experiments, and mean of each experiment is 
    # plotted

    # TODO assume fixed amount of plots (2x3)
    # For each baseline, plot one plot
    fig, axs = pyplot.subplots(2, 3, sharex="row", sharey=True)

    # TODO hardcoded formatter for x-axis to divide ticks
    # by specific amount
    x_formatter = matplotlib.ticker.FuncFormatter(xtick_formatter)

    legends = None
    for x in range(3):
        for y in range(2):
            ax = axs[y, x] 
            idx = y*3 + x
            experiments = get_filtered_experiments(args.results[idx])
            for experiment in experiments.values():
                plot_experiment_curve(experiment, plot_std=False, plotter=ax, 
                                      plot_lengths=args.lengths)
            # TODO hardcoded x-range
            if "DQN" in args.results[idx]:
                ax.set_xticks([0,25000,50000])
            else:
                ax.set_xticks([0,500000,1000000])
            ax.set_ylim(-1, 101)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.xaxis.set_major_formatter(x_formatter)

            if legends is None:
                legends = list(experiments.keys())

            # Hide ticks of other plots than leftmost ones
            if idx != 0 and idx != 3:
                ax.tick_params(axis="x", which="both", labelbottom=False)
            else:
                # Make tick labels smaller
                ax.tick_params(axis='both', which='both', labelsize=8, pad=-5)

            # Plot labels to specific subplots
            if idx == 3:
                ax.set_xlabel("Environment steps (thousands)", fontsize=9)
                ax.set_ylabel("PPO\nAverage episode length", fontsize=9)
            if idx == 0: 
                ax.set_ylabel("DQN\nAverage episode length", fontsize=9)

    fig.legend(legends, loc="lower right", ncol=3, prop={'size': 8})
    pyplot.tight_layout()
    pyplot.subplots_adjust(top=0.99, left=0.10, right=0.99, hspace=0.10, wspace=0.05)
    pyplot.savefig(args.output)


args = parser.parse_args()
if args.plot_style == "means":
    main_means(args)
else:
    main_paper(args)
