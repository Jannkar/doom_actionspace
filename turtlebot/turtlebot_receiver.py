import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from math import radians
from sensor_msgs.msg import CompressedImage, Image, BatteryState
from time import sleep

""" Short script to allow synchronous communication between the Turtlebot and server. Run this code on the Turtlebot."""

TIME_PER_COMMAND = (1.0/35.0)*15

print("Trying to initialize turtlebot connection")
rospy.init_node('turtlebot', anonymous=False)
print("Connection initialized")
r = rospy.Rate(5)   # 5 HZ

cmd_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=None)
# This returns the info when we are done
return_publisher = rospy.Publisher('/control_turtle_return', CompressedImage, queue_size=None)

global image_data

def control_callback(data):
	global image_data
	cmd_publisher.publish(data)
	sleep(TIME_PER_COMMAND)
	cmd_publisher.publish(Twist()) #TODO stop
	return_publisher.publish(image_data) #TODO tell computer when we are done

def image_callback(data):
	global image_data
	image_data = data

rospy.Subscriber('/control_turtle', Twist, control_callback, queue_size=None)
img_topic = "/camera/rgb/image_color/compressed"
image_sub = rospy.Subscriber(img_topic, CompressedImage, image_callback, queue_size=None)

rospy.sleep(1) # give DoomTurtle short time to initialize the connection
rospy.spin()