import rospy
from geometry_msgs.msg import Twist
from math import radians
from sensor_msgs.msg import CompressedImage, Image, BatteryState
from std_msgs.msg import String
from time import sleep, time
import sys
from PIL import Image, ImageFile
import numpy as np
import io
import threading

class DoomTurtle():
    """ 
    Class to initialize connection with Turtlebot and communicate with it.
    Provides simple functions for sending actions for the Turtlebot and for
    receiving data from it
    """
    def __init__(self):
        print("Trying to initialize turtlebot connection")
        rospy.init_node('doomturtle', anonymous=False)
        print("Connection initialized")
        r = rospy.Rate(5)   # 5 HZ

        # Publish commands to control motors asynchronously
        self.manual_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=None)

        # Publish commands to control motors synchronously
        self.cmd_vel = rospy.Publisher('/control_turtle', Twist, queue_size=None)
        
        self.image = 0
        self.image_received = False

        # Subscribe to see battery state
        battery_topic = "/battery_state"
        self.battery_sub = rospy.Subscriber(battery_topic, BatteryState, self.battery_callback)
        self.battery_state = ""

        # Subscribe to the return information when command was done
        self.cmd_return = rospy.Subscriber("/control_turtle_return", CompressedImage, self.action_done_callback, queue_size=None)
        self.action_done = False

        rospy.sleep(1) # give DoomTurtle time to initialize the connection
        rospy.on_shutdown(self.stop)

    def move_forward(self):
        """Move forward with 0.2 m/s. Max is 0.26m/s. """
        move_cmd = Twist()
        move_cmd.linear.x = 0.2
        self.cmd_vel.publish(move_cmd)

    def move_backward(self):
        """Move backward with 0.2 m/s. Max is 0.26m/s. """
        move_cmd = Twist()
        move_cmd.linear.x = -0.2
        self.cmd_vel.publish(move_cmd)

    def turn_left(self):
        """Turn with 45 deg/s. Max is 104 deg/s. """
        turn_cmd = Twist()
        turn_cmd.linear.x = 0
        turn_cmd.angular.z = radians(45) #45 deg/s in radians/s
        self.cmd_vel.publish(turn_cmd)

    def turn_right(self):
        """Turn with 45 deg/s. Max is 104 deg/s"""
        turn_cmd = Twist()
        turn_cmd.linear.x = 0
        turn_cmd.angular.z = radians(-45) #45 deg/s in radians/s
        self.cmd_vel.publish(turn_cmd)

    def move_delta(self, turn_delta, move_delta):
        move_cmd = Twist()
        move_cmd.linear.x = move_delta
        move_cmd.angular.z = radians(turn_delta)
        self.cmd_vel.publish(move_cmd)

    def async_action_delta(self, turn_delta, move_delta):
        move_cmd = Twist()
        move_cmd.linear.x = move_delta
        move_cmd.angular.z = radians(turn_delta)
        self.manual_cmd.publish(move_cmd)

    def stop(self):
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)

    def action_done_callback(self, data):
        """ Called when Turtlebot informs that the action was done """
        self.action_done = True
        self.image = data

    def make_action(self, action):
        """ Sends synchronoush action command to Turtlebot and waits for the return image"""
        self.move_delta(action[0], action[1])
        # Wait for the results
        start_time = time()
        self.action_done = False
        while not self.action_done:
            if (time()-start_time) > 10: 
                print("[WARNING] No reply from turtlebot after executing action")
                start_time = time()
            sleep(0.01)
        return self.process_image(self.image)

    def make_async_action(self, action):
        self.async_action_delta(action[0], action[1])

    def battery_callback(self, data):
        self.battery_state = round((data.percentage-1) * 1000)

    def process_image(self, image):
        byteio = io.BytesIO(image.data)
        img = Image.open(byteio, "r")
        img_array = np.asarray(img, dtype="int32")
        return img_array

    def get_state(self):
        screen_buffer = self.get_image()
        #depth_buffer = self.get_depth() #TODO: depth buffer not yet implemented
        return screen_buffer

    def get_battery_state(self):
        return self.battery_state