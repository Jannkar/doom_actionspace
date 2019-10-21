#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  vizdoom_wrappers.py
#  Wrappers around Vizdoom, similar to what OpenAI Universe has 
#  (e.g. image resizing/cropping, adding noise, stacking)
#  Author: Anssi Kanervisto

import numpy as np
# TODO use cv2 resize or skimage resize instead
from scipy.misc import imresize
import random as r
import vizdoom as vz
from collections import namedtuple
import math as m

# TODO consider inherting DoomGame...
# TODO how all possible actions are provided to the script?
# TODO move all class-constants to some common config file etc
# TODO modify resize module so that it always returns [height, widht, channel],
#      even when number of channels == 1

# Resolution tuples are [width, height], but images are shaped [height, width]

class VizdoomWrapper():
    """ Main class holding modules for randomization and other settings
    
    NOTE: All actions to DoomGame should be done via this class after 
    this has been created!
    
    `modules` is a list of randomizer modules (below), each of which
    define function for getting state and making an action. The modules are 
    called in the order they are in the list 
    
    `verbose` specifies if modules should print out their values and other
    logging info"""
    
    def __init__(self, doomgame, modules, verbose=False):
        """ DoomGame settings should not change after this point """
        self.doomgame = doomgame
        self.modules = modules
        self.verbose = verbose
        
        self.in_resolution = [self.doomgame.get_screen_width(),
                              self.doomgame.get_screen_height()]
        self.in_channels   = self.doomgame.get_screen_channels()
        
        # Initialize modules
        last_res = self.in_resolution
        last_channel = self.in_channels
        for module in self.modules:
            module.initialize(self.doomgame, last_res, last_channel, self.verbose)
            last_res = module.out_resolution
            last_channel = module.out_channels
        
        self.out_resolution = self.modules[-1].out_resolution
        self.out_channels   = self.modules[-1].out_channels
        
        # TODO depthmap
        
    def get_state(self):
        state = self.doomgame.get_state()
        
        # State can be None (e.g. at the end of episode)
        if state is None:
            return state
        
        ret_state = ExState(screen_buffer=state.screen_buffer,
                            depth_buffer=state.depth_buffer,
                            game_variables=state.game_variables)
        
        if ret_state.screen_buffer is None:
            return None
        
        # Go through the modules
        for module in self.modules:
            ret_state = module.get_state(ret_state)
        return ret_state
        
    def make_action(self, act, num_steps):
        # Go modules in reverse order, modifying the actions
        for module in self.modules[::-1]:
            act, num_steps = module.make_action(act, num_steps)
        # Do action
        reward = self.doomgame.make_action(act, num_steps)
        # Calculate rewards
        for module in self.modules[::-1]:
            reward = module.compute_reward(reward)
        return reward
    
    def advance_action(self):
        self.doomgame.advance_action()
        
    def get_last_action(self):
        return self.doomgame.get_last_action()
    
    def get_last_reward(self):
        return self.doomgame.get_last_reward()
    
    def get_total_reward(self):
        return self.doomgame.get_total_reward()
    
    def get_last_action(self):
        return self.doomgame.get_last_action()
    
    def send_game_command(self, cmd):
        return self.doomgame.send_game_command(cmd)
    
    def get_available_buttons(self):
        return self.doomgame.get_available_buttons()
    
    def get_available_game_variables(self):
        return self.doomgame.get_available_game_variables()
    
    def is_episode_finished(self):
        return self.doomgame.is_episode_finished()
    
    def close(self):
        self.doomgame.close()
    
    def init(self):
        self.doomgame.init()
        for module in self.modules:
            module.new_episode(self.doomgame)
    
    def new_episode(self):
        self.doomgame.new_episode()
        for module in self.modules:
            module.new_episode(self.doomgame)
        
    def is_episode_finished(self):
        return self.doomgame.is_episode_finished()

class ExState():
    """ An extension to "state" provided by Vizdoom get_state """
    # TODO add depth/map/label buffers etc
    
    def __init__(self, screen_buffer=None, depth_buffer=None, 
                 game_variables=None):
        self.screen_buffer = screen_buffer
        self.depth_buffer = depth_buffer
        self.game_variables = game_variables
        

class Module():
    """ Baseclass for modules in the wrapper """
    def __init__(self):
        """ Proper initializion is done in `initialize`, where doom game 
        has been created.
        Random variables can be initialized here (e.g. strength of noise)"""
        self.in_resolution = None
        self.in_channels = None
        self.out_resolution = None
        self.out_channels = None
        
        self.verbose = None
        
    def initialize(self, doomgame, in_res, in_channels, verbose):
        """ Initialize the module with proper. Additionally this can modify
        DoomGame if need to be """
        self.in_resolution = in_res
        self.in_channels = in_channels
        # By default, image shape is not changed
        self.out_resolution = in_res
        self.out_channels = in_channels
        
        self.verbose = verbose

    def get_state(self, state):
        """ Modifies state (in-place)"""
        return state
        
    def make_action(self, act, num_steps):
        """ Modifies actions (in-place)"""
        return act, num_steps
    
    def new_episode(self, doomgame):
        """ Called after new episode starts (including first episode) """
        return 
    
    def compute_reward(self, reward):
        """ Modify reward signal. Called after "make_action".
        Receives current reward, returns modified reward """
        return reward

class ReshapeNormalize(Module):
    """ Reshapes the image and normalizes to [0,1]
    
    Note1: Assumes either CRCGCB or GRAY8 modes!
    Note2: Image is converted to [height, width, channels]"""
    
    INTERP = "bilinear"
    
    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height
        self.imresize_shape = (self.height, self.width)
        
    def initialize(self, doomgame, in_res, in_channels, verbose):
        super().initialize(doomgame, in_res, in_channels, verbose)
        self.out_resolution = (self.width, self.height)
        self.out_channels = in_channels
        # Check aspect ratios:
        if in_res[0]/in_res[1] != self.width/self.height:
            print("[ReshapeNormalize] Warning: Output and input have different"+
                  " aspect ratios")
        
    def get_state(self, state):
        # Handle screen buffer
        origtype = state.screen_buffer.dtype
        img = state.screen_buffer.astype(np.float32)
        imgwidth = img.shape[1] if self.in_channels > 1 else img.shape[0]
        imgheight = img.shape[2] if self.in_channels > 1 else img.shape[1]
        if imgheight != self.height and imgwidth != self.width:
            if self.in_channels == 1:
                img = imresize(img, self.imresize_shape, 
                           ReshapeNormalize.INTERP, "F")
            else:
                
                new_image = np.zeros((self.height, self.width,self.out_channels),
                                     dtype=np.float32)
                for i in range(self.in_channels):
                    new_image[...,i] = imresize(img[i], [self.height, self.width], 
                                                ReshapeNormalize.INTERP, "F")
                img = new_image
                
                """ This causes variation in the brightness of pixels
                img = img.transpose([1,2,0])
                img = imresize(img, [self.height, self.width, self.out_channels], 
                               ReshapeNormalize.INTERP)
                """
                        
        if origtype == np.uint8:
            img = img/255
        else:
            raise ValueError("[ReshapeNormalize] Unknown datatype for "+
                "normalization: %s" % str(origtype))
        state.screen_buffer = img
        
        # Handle depth map if one available
        if state.depth_buffer is not None:
            depth = state.depth_buffer.astype(np.float32)
            if depth.shape[0] != self.height and depth.shape[1] != self.width:
                depth = imresize(depth, self.imresize_shape, 
                                 ReshapeNormalize.INTERP, "F")
            # Depthmap is always uint8
            depth = depth/255
            state.depth_buffer = depth
        return state

class RandomNoise(Module):
    """ Adds white/gaussian noise to the image
    
    Note: Assumes image pixels are in [0,1]
    
    strength: for uniform, this is the max value 
              for gaussian, N(0,1) is multiplied by [this_value]/3
    gaussian: If true, always use gaussian noise. 
              If none, pick white/gauss randomly
    shared_among_channels: If true, same noise per pixel is applied to all 
                           channels"""
    
    # TODO salt/pepper noise?
    
    MAX_STRENGTH = 0.2
    
    def __init__(self, strength=None, gaussian=None, shared_among_channels=None):
        super().__init__()
        self.strength = strength
        self.gaussian = gaussian
        self.shared = shared_among_channels
        
        self.randomize_gauss = self.gaussian is None
        if self.gaussian is None:
            self.gaussian = r.randint(0,1)
        
        self.randomize_strength = strength is None
        if self.strength is None:
            self.strength = r.random()*RandomNoise.MAX_STRENGTH
        
        self.randomize_shared = self.shared is None
        if self.shared is None:
            self.shared = r.randint(0,1)
        
    def get_state(self, state):
        img = state.screen_buffer
        noise = None
        if self.gaussian:
            if self.shared:
                noise = np.random.randn(*img.shape[:2])*(self.strength/3)
            else:
                noise = np.random.randn(*img.shape)*(self.strength/3)
        else:
            if self.shared:
                noise = np.random.random(img.shape[:2])*self.strength
            else:
                noise = np.random.random(img.shape)*self.strength
        if len(img.shape) == 3 and len(noise.shape) < 3:
            # Fix shape dimensions so broadcasting works
            noise = np.expand_dims(noise, -1)
        # Add noise and make sure you we do not go outside range
        img = img+noise
        img = np.clip(img, 0, 1)
        state.screen_buffer = img
        return state
    
    def new_episode(self, doomgame):
        # Pick new randoms
        if self.randomize_strength:
            self.strength = r.random()*RandomNoise.MAX_STRENGTH

        if self.randomize_gauss:
            self.gaussian = r.randint(0,1)

        if self.randomize_shared:
            self.shared = r.randint(0,1)
            
        if self.verbose:
            print("[RandomNoise] Strength: %f" % self.strength)
            print("[RandomNoise] Gaussian: %d" % self.gaussian)
            print("[RandomNoise] SharedCh: %d" % self.shared)
        
class RandomGamma(Module):
    """ Randomizes gamma in game (using zdoom commands) """
    
    GAMMA_RANGE = [0.6, 1.5]
    
    def __init__(self, gamma=None):
        super().__init__()
    
    def new_episode(self, doomgame):
        rgamma = r.uniform(*RandomGamma.GAMMA_RANGE)
        doomgame.send_game_command("gamma %f" % rgamma) 
        
        if self.verbose:
            print("[RandomGamma] Gamma: %f" % rgamma)

class RandomBobbing(Module):
    """ Randomizes headbobbing (up/down movement while moving)"""
    
    # Default is 0.25
    BOB_RANGE = [0.0,0.5]
    
    def __init__(self, bob_strength=None):
        super().__init__()
        self.strength = bob_strength
        
    def new_episode(self, doomgame):
        rbob = self.strength
        if rbob is None:
            rbob = r.uniform(*RandomBobbing.BOB_RANGE)
        doomgame.send_game_command("movebob %f" % rbob) 
        
        if self.verbose:
            print("[RandomBobbing] Movebob: %f" % rbob)
        
class RandomMirroring(Module):
    """ Randomly mirrors the image left and right """
    def __init__(self, mirror=None):
        super().__init__()
        # If None, pick at random if next image should be mirrored
        self.mirror = mirror
        self.randomize_mirror = self.mirror is None
        
        self.left_btn_i = None
        self.right_btn_i = None
        
        self.tleft_btn_i = None
        self.tright_btn_i = None
        
        self.delta_btn_i = None
    
    def initialize(self, doomgame, in_res, in_channels, verbose):
        super().initialize(doomgame, in_res, in_channels, verbose)
        
        # For compact indexing
        index = lambda l,v: l.index(v) if v in l else None 
        buttons = doomgame.get_available_buttons()
        
        # Get buttons we need to flip
        self.left_btn_i = index(buttons, vz.Button.MOVE_LEFT)
        self.right_btn_i = index(buttons, vz.Button.MOVE_RIGHT)
        
        self.tleft_btn_i = index(buttons, vz.Button.TURN_LEFT)
        self.tright_btn_i = index(buttons, vz.Button.TURN_RIGHT)
        
        self.delta_btn_i = index(buttons, vz.Button.TURN_LEFT_RIGHT_DELTA)
        
        # Sanity checking
        if ((self.left_btn_i is None) + (self.right_btn_i is None)) == 1:
            # One button defined, one not
            raise ValueError("[RandomMirroring] "+
                "Only one of the following buttons was used: "+
                "MOVE_LEFT and MOVE_RIGHT")
        if ((self.tleft_btn_i is None) + (self.tright_btn_i is None)) == 1:
            # One button defined, one not
            raise ValueError("[RandomMirroring] "+
                "Only one of the following buttons was used: "+
                "TURN_LEFT and TURN_RIGHT")
    
    def get_state(self, state):
        if self.mirror: 
            state.screen_buffer = np.fliplr(state.screen_buffer)
            if state.depth_buffer is not None:
                state.depth_buffer = np.fliplr(state.depth_buffer)
        return state
    
    def _swap_values(self, l, i1, i2):
        """ Swaps values of indexes i1 and i2 in list l """
        temp = l[i1]
        l[i1] = l[i2]
        l[i2] = temp
    
    def make_action(self, act, num_steps):
        # Flip buttons
        if self.mirror:
            # Move left/right
            if self.left_btn_i is not None:
                self._swap_values(act, self.left_btn_i, self.right_btn_i)
            # Turn left/right
            if self.tleft_btn_i is not None:
                self._swap_values(act, self.tleft_btn_i, self.tright_btn_i)
            # Delta
            if self.delta_btn_i is not None:
                act[self.delta_btn_i] = -act[self.delta_btn_i]
        
        return act, num_steps
        
    def new_episode(self, doomgame):
        # Randomize mirroring if not explicitly set
        self.mirror = r.randint(0,1) if self.randomize_mirror else self.mirror
        
        if self.verbose:
            print("[RandomMirroring] Mirror: %d " % self.mirror)
        
class RandomMap(Module):
    """ Randomizes the map played """
    def __init__(self, list_of_maps):
        super().__init__()
        self.list_of_maps = list_of_maps
        
    def new_episode(self, doomgame):
        randmap = r.sample(self.list_of_maps, 1)[0]
        doomgame.set_doom_map(randmap)
        
        if self.verbose:
            print("[RandomMap] Map: %s " % randmap)
        
class RandomFov(Module):
    """ Randomizes the field of view (zdoom commands)"""
    
    FOV_RANGE = [50,120]
    
    def __init__(self, fov=None):
        super().__init__()
    
    def new_episode(self, doomgame):
        rfov = r.uniform(*RandomFov.FOV_RANGE)
        doomgame.send_game_command("fov %f" % rfov)
        
        if self.verbose:
            print("[RandomFov] FOV: %f " % rfov)
        
class RandomBot(Module):
    """ Adds bots to the game """
    
    BOTS_RANGE = [2,8]

    def __init__(self, num_bots=None):
        super().__init__()
        self.num_bots = num_bots
    
    def new_episode(self, doomgame):
        add_bots = (self.num_bots if self.num_bots is not None else 
                    r.randint(*RandomBot.BOTS_RANGE))
        doomgame.send_game_command("removebots")
        for i in range(add_bots):
            self.doom.send_game_command("addbot")
    
class RandomUpDown(Module):
    """ Change viewpoint so that it randomly looks bit down or up """
    
    # Max speed of looking up or down (DELTA speed)
    MAX_SPEED = 12

    def __init__(self, amount=None, up=None):
        super().__init__()
        self.amount = amount
        self.up = up
        self.random_amount = self.amount is None
        self.random_dir = self.up is None
        
        self.up_down_index = None
        # if UP_DOWN_DELTA was already defined and we did not need to add it
        self.up_down_index_predefined = False
        
    def initialize(self, doomgame, in_res, in_channels, verbose):
        super().initialize(doomgame, in_res, in_channels, verbose)
        # Add UP_DOWN_DELTA if does not exist yet
        buttons = doomgame.get_available_buttons()
        if not vz.Button.LOOK_UP_DOWN_DELTA in buttons:
            doomgame.add_available_button(vz.Button.LOOK_UP_DOWN_DELTA)
            self.up_down_index = len(buttons)
        else:
            self.up_down_index = buttons.index(vz.Button.LOOK_UP_DOWN_DELTA)
            self.up_down_index_predefined = True
    
    def make_action(self, act, num_steps):
        # If UP_DOWN_DELTA was not defined before, set it to zero here
        # (i.e. "it does not exist")
        if (not self.up_down_index_predefined and 
                len(act) >= self.up_down_index+1):
            act[self.up_down_index] = 0
        return act, num_steps
    
    def new_episode(self, doomgame):
        # 0 = down, 1 = up
        if self.random_dir:
            self.up = r.randint(0,1)
        if self.random_amount:
            self.amount = r.randint(0,RandomUpDown.MAX_SPEED)
        self.amount = self.amount if self.up else -self.amount
        
        # Create action for looking up/down
        act = [0 for i in range(self.up_down_index)]
        act.append(self.amount)
        doomgame.make_action(act, 1)
        doomgame.get_state()
        
        
class StuckPenalty(Module):
    """ Adds penalty to reward if player is stuck (tries to move, but 
        does not move)"""
    
    DEFAULT_STUCK_PENALTY = -0.1
    # Threshold for how much movement is considered "stuck" (below this number)
    STUCK_THRESHOLD       = 1.0 
    MOVE_BUTTONS = [vz.Button.MOVE_LEFT,vz.Button.MOVE_RIGHT,
                    vz.Button.MOVE_FORWARD, vz.Button.MOVE_BACKWARD,
                    vz.Button.MOVE_LEFT_RIGHT_DELTA,
                    vz.Button.MOVE_FORWARD_BACKWARD_DELTA]
                     

    def __init__(self, stuck_penalty = None):
        super().__init__()
        self.stuck_penalty = StuckPenalty.DEFAULT_STUCK_PENALTY
        if stuck_penalty is not None:
            self.stuck_penalty = stuck_penalty
        self.button_is = []
        self.attempted_moving = False
        self.last_x = None
        self.last_y = None
        
        self.doomgame = None
        
    def initialize(self, doomgame, in_res, in_channels, verbose):
        super().initialize(doomgame, in_res, in_channels, verbose)
        self.doomgame = doomgame
        
        buttons = doomgame.get_available_buttons()
        for i in range(len(StuckPenalty.MOVE_BUTTONS)):
            if StuckPenalty.MOVE_BUTTONS[i] in buttons:
                self.button_is.append(buttons.index(
                                                StuckPenalty.MOVE_BUTTONS[i]))
    
    def make_action(self, act, num_steps):
        self.attempted_moving = False
        for button_i in self.button_is:
            if act[button_i] != 0:
                self.attempted_moving = True
                self.last_x = self.doomgame.get_game_variable(
                                vz.GameVariable.POSITION_X)
                self.last_y = self.doomgame.get_game_variable(
                                vz.GameVariable.POSITION_Y)
                break
        return act, num_steps
    
    def compute_reward(self, reward):
        if self.attempted_moving:
            new_x = self.doomgame.get_game_variable(
                                    vz.GameVariable.POSITION_X)
            new_y = self.doomgame.get_game_variable(
                                    vz.GameVariable.POSITION_Y)
            if ((abs(new_x-self.last_x)+abs(new_y-self.last_y)) <= 
                        StuckPenalty.STUCK_THRESHOLD):
                reward = reward + self.stuck_penalty
        return reward
    
    def new_episode(self, doomgame):
        self.attempted_moving = False
        self.last_x = None
        self.last_y = None
        
class SpeedReward(Module):
    """ Adds bonus points for movement """
    
    def __init__(self):
        super().__init__()
        self.doomgame = None
        self.last_pos = None
        
    def initialize(self, doomgame, in_res, in_channels, verbose):
        super().initialize(doomgame, in_res, in_channels, verbose)
        self.doomgame = doomgame
        self.last_pos = None

    def compute_reward(self, reward):
        cur_x = self.doomgame.get_game_variable(vz.GameVariable.POSITION_X)
        cur_y = self.doomgame.get_game_variable(vz.GameVariable.POSITION_Y)
        
        if self.last_pos is not None:
            dist = (cur_x - self.last_pos[0])**2 + (cur_y - self.last_pos[1])**2
            reward += ((m.sqrt(dist)/4000)-(0.005))

        self.last_pos = [cur_x, cur_y]

        return reward
    
    def new_episode(self, doomgame):
        self.last_pos = None

class CollisionPenalty(Module):
    """ Adds penalty to reward if something is too close to player (scan line 
        in middle)
        NOTE: Assumes: Depth map normalized to [0,1]"""
    
    # If min(depth_map) < 0 -> collision
    COLLISION_THRESHOLD = 0.070
    # If collision, add this to reward
    COLLISION_REWARD = -1
    
    def __init__(self):
        super().__init__()
        self.reward_plus = 0
    
    def get_state(self, state):
        if state.depth_buffer is not None:
            # Take middle line
            min_dist = state.depth_buffer[state.depth_buffer.shape[0]//2, :].min()
            if min_dist > 1:
                print("[CollisionPenalty] Looks like depth is not normalized [0,1]")
            if min_dist < CollisionPenalty.COLLISION_THRESHOLD:
                self.reward = CollisionPenalty.COLLISION_REWARD
                if self.verbose:
                    print("[CollisionPenalty] Collision penalty! Dist: %f" % 
                          min_dist)
            else:
                self.reward = 0
        else:
            raise ValueError("[CollisionPenalty] No depth map available")
        return state
    
    def compute_reward(self, reward):
        return reward+self.reward_plus
    
