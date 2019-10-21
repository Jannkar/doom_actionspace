import time
import os
import sys
import numpy as np
from rl.callbacks import Callback
import json

# This is done if ROS is installed with Python 2.7. Importing just cv2 would import it from this ROS installation,
# but now instead we want to use Python 3 version of cv2
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
    import cv2
    sys.path.append(ros_path)
else:
    import cv2

class VideoRecorder():
    """ Class for recording videos """
    save_location = "./videos/"
    day_month = ""
    video_out = None

    def __init__(self, width, height):
        self.video_buffer = []
        self.day_month = time.strftime("%d-%m - %H-%M-%S")
        if not os.path.exists(self.save_location):
            os.makedirs(self.save_location)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_out = cv2.VideoWriter(self.save_location + "video " + self.day_month+".avi"
            , fourcc, 35.0, (width, height))
        print("Starting to record video")

    def record_video(self, img):
        self.video_buffer.append(img)
        if len(self.video_buffer) > 350: # Save once per 10 seconds
            self.save_video()

    def save_video(self):
        for img in self.video_buffer:
            img = (img*255).astype(np.uint8)

            if len(img.shape) > 2: #more than one channel
                self.video_out.write(img[...,::-1]) # Flip RGB to BGR
            else:
                self.video_out.write(img)
        self.video_buffer = []

    def finish_video(self):
        self.save_video()
        self.video_out.release()


class CustomTestLogger(Callback):
    def __init__(self, filepath, interval=None):
        self.filepath = filepath
        self.interval = interval
        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dict that maps from episode to metrics array.
        #self.metrics = {}
        #self.starts = {}
        self.data = {}

    def on_train_begin(self, logs):
        """ Initialize model metrics before training """
        #self.metrics_names = self.model.metrics_names

    def on_train_end(self, logs):
        """ Save model at the end of training """
        self.save_data()

    def on_episode_begin(self, episode, logs):
        #print(episode)
        """ Initialize metrics at the beginning of each episode """
        #assert episode not in self.metrics
        #assert episode not in self.starts
        #self.metrics[episode] = []
        #self.starts[episode] = timeit.default_timer()

    def on_episode_end(self, episode, logs):
        """ Compute and print metrics at the end of each episode """ 
        data = list(logs.items())
        data += [('episode', episode)]
        for key, value in data:
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

        if self.interval is not None and episode % self.interval == 0:
             self.save_data()

    def on_step_end(self, step, logs):
        """ Append metric at the end of each step """
        #self.metrics[logs['episode']].append(logs['metrics'])

    def save_data(self):
        """ Save metrics in a json file """
        if len(self.data.keys()) == 0:
            return

        #Sort everything by episode.
        assert 'episode' in self.data
        sorted_indexes = np.argsort(self.data['episode'])
        sorted_data = {}
        for key, values in self.data.items():
            assert len(self.data[key]) == len(sorted_indexes)
            # We convert to np.array() and then to list to convert from np datatypes to native datatypes.
            # This is necessary because json.dump cannot handle np.float32, for example.
            sorted_data[key] = np.array([self.data[key][idx] for idx in sorted_indexes]).tolist()

        # Overwrite already open file. We can simply seek to the beginning since the file will
        # grow strictly monotonously.
        with open(self.filepath, 'w') as f:
            json.dump(sorted_data, f)