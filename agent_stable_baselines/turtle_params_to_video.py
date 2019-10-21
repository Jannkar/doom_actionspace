import numpy as np
import os
import io
from PIL import Image
import pickle
from utils import VideoRecorder
import argparse

def save_video(args):
    """ Records a video from saved Turtlebot parameters. Note that parameters are usually
        saved with the frame_skip on, which is why the video appears to be very fast. 
        Saves the video to /videos/ directory. """

    rootdir = args.directory
    files_array = []

    for subdir, dirs, files in os.walk(rootdir):
        episodes = []
        for file in files:
            filename = os.path.join(file)
            if ".pkl" in filename: 
                path = subdir + "/" + filename
                episodes.append(path)
        if len(episodes) > 0:
            episodes.sort(key=lambda x: os.path.getmtime(x))
            files_array.append(episodes)

    for episodes in files_array:
        with open(episodes[0], "rb") as f:
            parameters1 = pickle.load(f)

        img = np.asarray(Image.open(io.BytesIO(parameters1[0][0])))
        recorder = VideoRecorder(img.shape[1], img.shape[0]);
        for episode in episodes:

            print("Recording episode: ", episode)
            with open(episode, "rb") as f:
                parameters = pickle.load(f)
            i = 0
            for image in parameters[0]:
                img = np.asarray(Image.open(io.BytesIO(image))).astype('uint8')
                recorder.record_video(img/255)
                i += 1
        recorder.finish_video()
        print(" ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help="Give the directory, which contains all the saved Turtlebot episodes")
    args = parser.parse_args()
    save_video(args)