# Experiment code for paper "From Video Game to Real Robot: The Transfer between Action Spaces"

Train agent in Doom game, which can then be transferred to Turtlebot robot with different action space. 
Further information how to run the experiments can be found under `agent_stable_baselines`.

1. [Requirements](#Requirements)
1. [Experiments](#Experiments)
1. [Turtlebot initialization and controls](#turtlebot-initialization-and-controls)
1. [Scenarios](#Scenarios)

<a name="Requirements"></a>
# Requirements
* ViZDoom with dependencies (https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md)
* Python (tested on 3.5.2)
* Recommended: ROS Kinetic Kame, if you are going to use Turtlebot (http://wiki.ros.org/kinetic/Installation)
* Recommended: Ubuntu Xenial (16.04 LTS) operating system for ROS Installation. (Tested only with this OS)

At least the following python Libraries:
* Tensorflow
* OpenCV
* Open AI Gym
* Rospy (Only if you use Turtlebot)

<a name="Experiments"></a>
# Experiments
These experiments were done to perform action space transfer, while transferring agent from simulation to reality. 

Action space transfer is done with sim-to-sim experiments, to compare replace and adapter methods. In replace method, the output layer of the network is simply replaced with the new output layer, which corresponds the new action space. Other network weights are frozen and loaded from source model. In adapter method, instead of replacing the last layer, the new layer is added behind the previous layer, to "map" the old actions to new actions. Again, other weights than the output layer's weights are frozen and loaded from the source model. Similarly to these sim-to-sim experiments, the action space transfer was then performed with the Turtlebot robot.

To experiment with different algorithms (especially with PPO), the Stable Baselines codes were implemented. It also provides improved performance for the DQN algorithm. Same experiments were ran for the PPO as previously for DQN. Also some new experiments were made, to for instance see how loading and freezing the value function parameters affects the time to learn new action space. 

<a name="Turtlebot initialization"></a>
# Turtlebot initialization and controls

On the desktop computer, run command: 
`roscore`

On Turtlebot (can be done via ssh), to initialize the subscribers and publishers, run command:   
`roslaunch turtlebot3_bringup turtlebot3_doom.launch`

To communicate with the Turtlebot synchronously while using stable baselines, navigate to `python_scripts/` on Turtlebot and run:  
`python turtlebot_receiver.py`

Robot controls during the training (Controlled from cv2 screen):  
* wasd keys - Manual controlling between episodes
* space - Stops the robot, when controlling manually between episodes
* enter - Starts the next episode
* p - Gives reward during the training
 
Remember to use `--visualize 1` argument, otherwise controls will not work properly.

<a name="Scenarios"></a>
# Scenarios
Three available scenarios for ViZDoom:

* Find_the_object.wad - Task in the map is to find a tall red goal pillar and touch it. Agent starts from middle of the room, looking to random direction. Map is finished when the pillar is touched.

* Find_the_object_randomized.wad - Same as the previous one, but now the agent's view height and room textures are randomized. 

* Find_the_object_randomized_with_two_pillars.wad - Same as the previous one, but now there is also a green pillar in the room. However, touching the green pillar does not yield reward.

# License

Code original to this project is shared under MIT license. Code under `agent_stable_baselines/stable_baselines` is shared under MIT license with original copy from following GitHub repository: https://github.com/hill-a/stable-baselines
