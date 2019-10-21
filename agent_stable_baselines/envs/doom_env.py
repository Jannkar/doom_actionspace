from vizdoom import *
import numpy as np
from gym import Env
from random import choice
from time import sleep
import vizdoom_wrapper as vzw
from collections import deque
from gym.spaces import Box, Discrete
from skimage.transform import resize
from random import *
from itertools import product

class DoomEnv(Env):
    """ OpenAI Gym environment for ViZDoom """
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    action_space = None
    width = 80 
    height = 60 
    observation_space = Box(low=0, high=1, shape=(height, width, 4), dtype=np.float32)
    prev_observation = None
    prev_reward = None
    visualize = False
    image_shape = (height, width)
    window_length = 4
    past_images = deque(maxlen=window_length)
    

    def __init__(self, action_space, visualize = False, backend = "TensorFlow", sim_to_sim = 0, mode="train", discretes = "wasd", timeout = 1000, randomize=0):
        self.visualize = visualize
        self.backend = backend
        self.mode = mode

        if backend == "PyTorch":
            observation_space = Box(low=0, high=1, shape=(4, self.height, self.width), dtype=np.float32)

        # Initiate past_images deque
        for i in range(0,self.window_length):
            self.past_images.append(np.zeros(self.image_shape))

        # Define some actions. Each list entry corresponds to declared buttons:
        # TURN_RIGHT, TURN_LEFT, MOVE_FORWARD, MOVE_BACKWARD
        TURN_RIGHT = [True, False, False, False]
        TURN_LEFT = [False, True, False, False]
        MOVE_FORWARD = [False, False, True, False]
        MOVE_BACKWARD = [False, False, False, True]

        if sim_to_sim:
            MODULES = [vzw.ReshapeNormalize(self.width,self.height)]
        else:
            if randomize:
                MODULES = [vzw.ReshapeNormalize(self.width,self.height),
                   vzw.RandomNoise(),
                   vzw.RandomFov(),
                   vzw.RandomGamma(),
                   vzw.RandomBobbing(),
                   ]
            else:
                MODULES = [vzw.ReshapeNormalize(self.width,self.height)]

        self.game = DoomGame()

        #Sets the action space
        if action_space == "discrete" and discretes == "wasd":
            self.action_space = Discrete(4)
            self.actions = [TURN_RIGHT, TURN_LEFT, MOVE_FORWARD, MOVE_BACKWARD]
            #elf.nb_actions = len(self.action_space)
            self.game.add_available_button(Button.TURN_RIGHT)
            self.game.add_available_button(Button.TURN_LEFT)
            self.game.add_available_button(Button.MOVE_FORWARD)
            self.game.add_available_button(Button.MOVE_BACKWARD)
            print("actions:", self.actions)
        elif discretes != "wasd":
            self.actions = []
            if "w" in discretes:
                #self.actions.append(MOVE_FORWARD)
                self.game.add_available_button(Button.MOVE_FORWARD)
            if "a" in discretes:
                #self.actions.append(TURN_LEFT)
                self.game.add_available_button(Button.TURN_LEFT)
            if "s" in discretes:
                #self.actions.append(MOVE_BACKWARD)
                self.game.add_available_button(Button.MOVE_BACKWARD)
            if "d" in discretes:
                #self.actions.append(TURN_RIGHT)
                self.game.add_available_button(Button.TURN_RIGHT)

            #creating actions list, maybe could be better way to do this
            for i in range(self.game.get_available_buttons_size()):
                arr = []
                for j in range(self.game.get_available_buttons_size()):
                    if i == j:
                        arr.append(True)
                    else:
                        arr.append(False)
                self.actions.append(arr)
            print(self.actions)

            self.action_space = Discrete(len(self.actions))

        elif action_space == "continuous":
            self.action_space = Box(-1, 1, shape=(2,), dtype=np.float32)
            print(self.action_space.shape)
            #self.nb_actions = self.action_space.shape

            self.game.add_available_button(Button.TURN_LEFT_RIGHT_DELTA)
            self.game.add_available_button(Button.MOVE_FORWARD_BACKWARD_DELTA)

        elif action_space == "semi-continuous":
            #Creating 25 actions
            tuple_actions = list(product([-5,-1,0,1,5], repeat=2))
            self.actions = [list(x) for x in tuple_actions]
            for i in range(len(self.actions)):
                self.actions[i][1] *= 5

            #minus the action where agent just stays still
            idx = self.actions.index([0,0])
            self.actions.pop(idx)

            self.action_space = Discrete(len(self.actions))

            self.game.add_available_button(Button.TURN_LEFT_RIGHT_DELTA)
            self.game.add_available_button(Button.MOVE_FORWARD_BACKWARD_DELTA)
            print("actions:", self.actions)
        else:
            raise AttributeError("Unknown action_space definition");

        print("action_space:", self.action_space)
        
        # Sets path to additional resources wad file which is basically your scenario wad.
        # If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
        if sim_to_sim:
            self.game.set_doom_scenario_path("scenarios/Find_the_object.wad")
        else:
            self.game.set_doom_scenario_path("scenarios/Find_the_object_randomized.wad")

        # Sets map to start (scenario .wad files can contain many maps).
        self.game.set_doom_map("map01")

        # Sets resolution. Default is 320X240
        self.game.set_screen_resolution(ScreenResolution.RES_160X120) #160X120

        # Sets the screen buffer format. Not used here but now you can change it. Default is CRCGCB.
        self.game.set_screen_format(ScreenFormat.CRCGCB) #RGB24

        # Enables depth buffer.
        self.game.set_depth_buffer_enabled(False)

        # Enables labeling of in game objects labeling.
        self.game.set_labels_buffer_enabled(False)

        # Enables buffer with top down map of the current episode/level.
        self.game.set_automap_buffer_enabled(False)

        # Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
        self.game.set_render_hud(False)
        self.game.set_render_minimal_hud(False)  # If hud is enabled
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(True)
        self.game.set_render_decals(False)  # Bullet holes and blood on the walls
        self.game.set_render_particles(False)
        self.game.set_render_effects_sprites(False)  # Smoke and blood
        self.game.set_render_messages(False)  # In-game messages
        self.game.set_render_corpses(False)
        self.game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items

        # Adds game variables that will be included in state.
        self.game.add_available_game_variable(GameVariable.AMMO2)

        # Causes episodes to finish after 1000 tics (actions)
        self.game.set_episode_timeout(timeout)

        # Makes episodes start after 10 tics (~after raising the weapon)
        #self.game.set_episode_start_time(10)

        # Makes the window appear (turned on by default)
        self.game.set_window_visible(visualize)

        # Turns on the sound. (turned off by default)
        self.game.set_sound_enabled(False)

        # Sets the livin reward (for each move) to -1
        self.game.set_living_reward(0)

        # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
        self.game.set_mode(Mode.PLAYER)

        #self.game.add_game_args("+vid_forcesurface 1")

        # Enables engine output to console.
        #game.set_console_enabled(True)

        # Initialize the game. Further configuration won't take any effect from now on.
        self.gameex = vzw.VizdoomWrapper(self.game, MODULES, verbose=False)
        self.gameex.init()

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        frame_skip = 10

        sleep_time = 0.028*frame_skip
        actions_multiplier = 25 #multiplier for continuous actions, which are between [-1,1]

        if self.visualize and self.mode == "test":
            sleep(sleep_time)

        if isinstance(self.action_space, Box): #Continuous
            left_right_delta = action[0]*actions_multiplier
            forward_backward_delta = action[1]*actions_multiplier
            reward = self.gameex.make_action([left_right_delta, forward_backward_delta], frame_skip)
        elif isinstance(self.action_space, Discrete):
            reward = self.gameex.make_action(self.actions[action], frame_skip)
        else:
            print(self.action_space)
            raise NotImplementedError("Other action spaces than space and discrete are not yet implemented")
        
        done = self.gameex.is_episode_finished()
        state = self.gameex.get_state()
        
        if state != None: # State can be none if episode is finished
            observation = self.process_observation(state.screen_buffer)
            self.prev_observation = observation
        else:
            observation = self.prev_observation

        # There is a bug in the scenario, when the reward 1.0 is sometimes given one state off. This is a workaround to fix this.
        if not done:
            self.prev_reward = reward
            info = {"Game running": "True"}
            if reward == 1.0: #Changing reward to 0, if not done. Next time WILL be done, so using then prev reward
               reward = 0
        else:
            if reward == 1.0 or self.prev_reward == 1.0:
                reward = 1.0
                info = {"Episode finished": "True"}
            else:
                reward = -1.0
                info = {"Timed_out": "True"}

        if done and self.mode == "test":
            print("R: ",reward, ". Done: ", done, ".Info: ", info)
        if done and reward == 0:
            raise ValueError("Terminal reward was 0")
        return observation, reward, done, info

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        """
        self.gameex.new_episode()

        # Reset past states deque
        for i in range(0,self.window_length):
            self.past_images.append(np.zeros(self.image_shape))
        
        state = self.gameex.get_state()
        observation = self.process_observation(state.screen_buffer)

        return observation

    def render(self, mode='human'):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self.gameex.close()


    def process_observation(self, observation):
        # Green Channel
        processed_observation = observation[:,:,1]

        self.past_images.append(processed_observation)

        stacked_images = np.asarray(self.past_images)

        if self.backend == "TensorFlow":
            #Transposing array for TensorFlow
            transposed_array = np.transpose(stacked_images, (1, 2, 0))
            return transposed_array
        elif self.backend == "PyTorch":
            return stacked_images
        else:
            raise ValueError("Unknown backend type.")

    def get_coordinates(self):
        pos_x = self.game.get_game_variable(POSITION_X)
        pos_y = self.game.get_game_variable(POSITION_Y)
        pos_z = self.game.get_game_variable(POSITION_Z)
        return pos_x, pos_y, pos_z