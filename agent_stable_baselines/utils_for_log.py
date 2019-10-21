import numpy as np
import os
import time
import csv
import json
import uuid
from glob import glob
import pandas

def fixedVectorLength(x, y, interval):
    """Takes in a two vectors, which have to be already sorted by x-values. Calculates 
    mean for datapoints between given interval. For example with interval = 1000,
    the first new x value will be 1000 and mean of y is calculated from the x values 
    within range 500-1500. Thus first 500 and final 500 values are excluded. """

    new_x = []
    new_y = []
    lower_bound = 0
    mean_y = []
    iterator = interval/2
    for i in range(0,len(x)):
        if x[i] > lower_bound+interval:
            if len(mean_y) > 0:
                new_y.append(np.mean(mean_y))
                new_x.append(iterator)
            mean_y = []
            mean_y.append(y[i])
            iterator += interval
            lower_bound = lower_bound+interval
        elif x[i] > lower_bound:
            mean_y.append(y[i])
    if len(mean_y) > 0:
        new_y.append(np.mean(mean_y))
        new_x.append(iterator)
    return new_x, new_y

def custom_ts2xy(timesteps, xaxis, yaxis = 'reward'):
    X_TIMESTEPS = 'timesteps'
    X_EPISODES = 'episodes'
    X_WALLTIME = 'walltime_hrs'
    Y_REWARD = 'reward'
    Y_EPISODE_LENGTH = 'episode_length'

    if xaxis == X_TIMESTEPS:
        x_var = np.cumsum(timesteps.l.values)
    elif xaxis == X_EPISODES:
        x_var = np.arange(len(timesteps))
    elif xaxis == X_WALLTIME:
        x_var = timesteps.t.values / 3600.
    else:
        raise NotImplementedError
    if yaxis == Y_REWARD:
        y_var = timesteps.r.values
    elif yaxis == Y_EPISODE_LENGTH:
        y_var = timesteps.l.values
    else:
        raise NotImplementedError
    return x_var, y_var

def load_results(path):
    """
    Load results from a given file
    :param path: (str) the path to the log file
    :return: (Pandas DataFrame) the logged data
    """
    # get both csv and (old) json files
    monitor_files = (glob(os.path.join(path, "*monitor.json")) + glob(os.path.join(path, "*monitor.csv")))
    print(monitor_files)
    if not monitor_files:
        raise Exception("Not a valid Monitor file")
    data_frames = []
    headers = []
    for file_name in monitor_files:
        with open(file_name, 'rt') as file_handler:
            if file_name.endswith('csv'):
                first_line = file_handler.readline()
                assert first_line[0] == '#'
                header = json.loads(first_line[1:])
                data_frame = pandas.read_csv(file_handler, index_col=None)
                headers.append(header)
            elif file_name.endswith('json'):  # Deprecated json format
                episodes = []
                lines = file_handler.readlines()
                header = json.loads(lines[0])
                headers.append(header)
                for line in lines[1:]:
                    episode = json.loads(line)
                    episodes.append(episode)
                data_frame = pandas.DataFrame(episodes)
            else:
                assert 0, 'unreachable'
            data_frame['t'] += header['t_start']
        data_frames.append(data_frame)
    data_frame = pandas.concat(data_frames)
    data_frame.sort_values('t', inplace=True)
    data_frame.reset_index(inplace=True)
    data_frame['t'] -= min(header['t_start'] for header in headers)
    # data_frame.headers = headers  # HACK to preserve backwards compatibility
    return data_frame