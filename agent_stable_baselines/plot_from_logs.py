import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
from itertools import cycle
from scipy.signal import savgol_filter
import argparse
import sys
from utils_for_log import fixedVectorLength, load_results, custom_ts2xy



def plot_logs(args):
    """ Plots the logs in specified folder. """
    multiple_files = []
    labels = []

    cycol = cycle('bgrmcky')
    colors = ['blue', 'green', 'red', 'magenta', 'cyan', 'black', 'yellow']

    rootdir = args.rootdir

    if args.plot_style == "multiple_means":
        dirs = next(os.walk(rootdir))[1]
        print(dirs)
        for directory in dirs:
            print(directory)
            files_array = []
            for subdir, dirs, files in os.walk(rootdir + directory):
                if "log" in subdir and "tb_log" not in subdir:
                    files_array.append(subdir)
            files_array.sort()
            multiple_files.append(files_array)
            print(files_array)
            labels.append(directory)
    else:
        files_array = []
        for subdir, dirs, files in os.walk(rootdir):
            if "log" in subdir and "tb_log" not in subdir:
                print(subdir)
                files_array.append(subdir)
        files_array.sort()
        multiple_files.append(files_array)

    j = 0
    for training_files in multiple_files:
        all_training_x_values = []
        all_training_y_values = []

        i = 0
        for file in training_files:
            print(file)
            x, y = custom_ts2xy(load_results(file), 'timesteps', args.mode)
            new_x = None
            new_y = None
            if args.smooth_algo == "fixed":
                new_x, new_y = fixedVectorLength(x, y, args.smooth);
            else:
                # Do average smoothing
                new_x = x
                new_y = np.convolve(np.ones((args.smooth,))/args.smooth, y, mode="valid")
                # Add missing points to the beginning
                missing_nums = len(y) - len(new_y)
                new_y = np.concatenate((np.cumsum(y[:missing_nums])/(np.arange(missing_nums)+1), new_y))

            if args.label != "":
                label = args.label
            else:
                label = "Training"

            if args.plot_style == "single":
                plt.plot(new_x,
                    new_y,
                    c=next(cycol),
                    label=file)
                plt.legend()
            else:
                all_training_x_values.append(new_x)
                all_training_y_values.append(new_y)
            print(len(new_x))
            print(len(new_y))
            i += 1

        if args.smooth_algo == "interp":
            # Do linear interpolation so points are at same point
            max_x = max(map(max, all_training_x_values))
            max_points = max(map(len, all_training_x_values))
            new_x = np.linspace(0, max_x, max_points)
            for i in range(len(all_training_y_values)):
                all_training_y_values[i] = np.interp(new_x, 
                                                     all_training_x_values[i],
                                                     all_training_y_values[i])

        if args.plot_style == "mean":
            training_means = np.mean(all_training_y_values, axis=0)
            training_stds = np.std(all_training_y_values, axis=0)

            print("training_means:", training_means)
            print("training_stds:", training_stds)

            plt.plot(new_x,
                    training_means,
                    c="blue",
                    label=label)
            plt.legend()
            plt.fill_between(new_x, training_means, np.subtract(training_means, training_stds), color=colors[j], alpha='0.25', edgecolor=None)
            plt.fill_between(new_x, training_means, np.add(training_means, training_stds), color=colors[j], alpha='0.25', edgecolor=None)

        elif args.plot_style == "multiple_means":
            training_means = np.mean(all_training_y_values, axis=0)
            training_stds = np.std(all_training_y_values, axis=0)

            print("training_means:", training_means)
            print("training_stds:", training_stds)

            plt.plot(new_x,
                    training_means,
                    c=next(cycol),
                    label=labels[j])
            plt.legend()

            plt.fill_between(new_x, training_means, np.subtract(training_means, training_stds), color=colors[j], alpha='0.25', edgecolor=None)
            plt.fill_between(new_x, training_means, np.add(training_means, training_stds), color=colors[j], alpha='0.25', edgecolor=None)

        j += 1


    last_x_tick = round(max(new_x),-3)
    print(last_x_tick)

    if args.mode == "reward":
        plt.xlabel("Training steps" + " (thousands)")
        plt.ylabel("Average episode reward")
        plt.axis((0,last_x_tick,-1,1.05))
    elif args.mode == "episode_length":
        plt.xlabel("Training steps" + " (thousands)")
        plt.ylabel("Average episode length (steps)")
        plt.axis((0,last_x_tick,0,105))

    x_axis_ticks = np.arange(0, last_x_tick+1, last_x_tick/8)
    plt.xticks(x_axis_ticks, [int(i) for i in x_axis_ticks/1000])

    plt.grid(alpha=0.5)
    plt.title(args.title)

    if args.save != "":
        plt.savefig(args.save + ".pdf", bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default="")
    parser.add_argument('--mode', choices=['reward', 'episode_length'], default='reward')
    parser.add_argument('--title', type=str, default="")
    parser.add_argument('--label', type=str, default="")
    parser.add_argument('--rootdir', type=str, default='./plots/')
    parser.add_argument('--plot_style', choices=['mean', 'single', 'multiple_means'], default='mean')
    parser.add_argument('--smooth', type=int, default=50)
    parser.add_argument('--smooth_algo', type=str, choices=["fixed", "interp"], default="interp")
    args = parser.parse_args()
    plot_logs(args)