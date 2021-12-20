#!/usr/bin/env python
#
# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
"""
This module contains a local planner to perform
low-level waypoint following based on PID controllers.
"""

from collections import deque
from genpy.message import check_type
import rospy
import math
import numpy as np
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion
from carla_waypoint_types.srv import GetWaypoint
from carla_msgs.msg import CarlaEgoVehicleControl
from vehicle_pid_controller import VehiclePIDController  # pylint: disable=relative-import
from misc import distance_vehicle  # pylint: disable=relative-import
import carla
import carla_ros_bridge.transforms as trans

class Obstacle:
    def __init__(self):
        self.id = -1 # actor id
        self.vx = 0.0 # velocity in x direction
        self.vy = 0.0 # velocity in y direction
        self.vz = 0.0 # velocity in z direction
        self.ros_transform = None # transform of the obstacle in ROS coordinate
        self.carla_transform = None # transform of the obstacle in Carla world coordinate
        self.bbox = None # Bounding box w.r.t ego vehicle's local frame

class MyLocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a trajectory of waypoints that is
    generated on-the-fly. The low-level motion of the vehicle is computed by using two PID
    controllers, one is used for the lateral control and the other for the longitudinal
    control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice.
    """

    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, role_name, opt_dict=None):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param role_name: name of the actor
        :param opt_dict: dictionary of arguments with the following semantics:

            target_speed -- desired cruise speed in Km/h

            sampling_radius -- search radius for next waypoints in seconds: e.g. 0.5 seconds ahead

            lateral_control_dict -- dictionary of arguments to setup the lateral PID controller
                                    {'K_P':, 'K_D':, 'K_I'}

            longitudinal_control_dict -- dictionary of arguments to setup the longitudinal
                                         PID controller
                                         {'K_P':, 'K_D':, 'K_I'}
        """
        self.target_route_point = None
        self._current_waypoint = None
        self._vehicle_controller = None
        self._waypoints_queue = deque(maxlen=20000)
        self._buffer_size = 5
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        self._vehicle_yaw = None
        self._current_speed = None
        self._current_pose = None
        self._obstacles = []
        self._changing_lane = False
        self._lane_delta = 0
        self._lane_change_history = []
        self._buffer_updated = False
        self._passing = False

        # get world and map for finding actors and waypoints
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        self.world = client.get_world()
        self.map = self.world.get_map()        

        self._target_point_publisher = rospy.Publisher(
            "/next_target", PointStamped, queue_size=1)

        rospy.wait_for_service('/carla_waypoint_publisher/{}/get_waypoint'.format(role_name))
        self._get_waypoint_client = rospy.ServiceProxy(
            '/carla_waypoint_publisher/{}/get_waypoint'.format(role_name), GetWaypoint)

        # initializing controller
        self._init_controller(opt_dict)

    def get_obstacles(self, location, range):
        """
        Get a list of obstacles that are located within a certain distance from the location.
        
        :param      location: queried location
        :param      range: search distance from the queried location
        :type       location: geometry_msgs/Point
        :type       range: float or double
        :return:    None
        :rtype:     None
        """
        self._obstacles = []
        actor_list = self.world.get_actors()
        for actor in actor_list:
            if "role_name" in actor.attributes:
                if actor.attributes["role_name"] == 'autopilot' or actor.attributes["role_name"] == "static":
                    carla_transform = actor.get_transform()
                    ros_transform = trans.carla_transform_to_ros_pose(carla_transform)
                    x = ros_transform.position.x
                    y = ros_transform.position.y
                    z = ros_transform.position.z 
                    distance = math.sqrt((x-location.x)**2 + (y-location.y)**2)
                    if distance < range:
                        # print("obs distance: {}").format(distance)
                        ob = Obstacle()
                        ob.id = actor.id
                        ob.carla_transform = carla_transform
                        ob.ros_transform = ros_transform
                        ob.vx = actor.get_velocity().x
                        ob.vy = actor.get_velocity().y
                        ob.vz = actor.get_velocity().z
                        ob.bbox = actor.bounding_box # in local frame
                        # print("x: {}, y: {}, z:{}").format(x, y, z)
                        # print("bbox x:{} y:{} z:{} ext: {} {} {}".format(ob.bbox.location.x, ob.bbox.location.y, ob.bbox.location.z, ob.bbox.extent.x, ob.bbox.extent.y, ob.bbox.extent.z))
                        self._obstacles.append(ob)

    def check_obstacle(self, point, obstacle):
        """
        Check whether a point is inside the bounding box of the obstacle

        :param      point: a location to check the collision (in ROS frame)
        :param      obstacle: an obstacle for collision check
        :type       point: geometry_msgs/Point
        :type       obstacle: object Obstacle
        :return:    true or false
        :rtype:     boolean   
        """
        carla_location = carla.Location()
        carla_location.x = point.x
        carla_location.y = -point.y
        carla_location.z = point.z
        
        vertices = obstacle.bbox.get_world_vertices(obstacle.carla_transform)
        
        vx = [v.x for v in vertices]
        vy = [v.y for v in vertices]
        vz = [v.z for v in vertices]
        return carla_location.x >= min(vx) and carla_location.x <= max(vx) \
                and carla_location.y >= min(vy) and carla_location.y <= max(vy) \
                and carla_location.z >= min(vz) and carla_location.z <= max(vz) 

    def check_obstacles(self, position):
        for obstacle in self._obstacles:
            if self.check_obstacle(position, obstacle):
                return True
        return False 

    def check_waypoint_obstacles(self, position):
        self.get_obstacles(position, 70.0)
        return self.check_obstacles(position)

    def get_coordinate_lanemarking(self, position):
        """
        Helper to get adjacent waypoint 2D coordinates of the left and right lane markings 
        with respect to the closest waypoint
        
        :param      position: queried position
        :type       position: geometry_msgs/Point
        :return:    left and right waypoint in numpy array
        :rtype:     tuple of geometry_msgs/Point (left), geometry_msgs/Point (right)
        """
        # get waypoints along road
        current_waypoint = self.get_waypoint(position)
        if current_waypoint.is_junction:
            return None, None
        waypoint_xodr = self.map.get_waypoint_xodr(current_waypoint.road_id, current_waypoint.lane_id, current_waypoint.s)
        
        # find two orthonormal vectors to the direction of the lane
        yaw = math.pi - waypoint_xodr.transform.rotation.yaw * math.pi / 180.0
        v = np.array([1.0, math.tan(yaw)])
        norm_v = v / np.linalg.norm(v)
        right_v = np.array([-norm_v[1], norm_v[0]])
        left_v = np.array([norm_v[1], -norm_v[0]])
        
        # find two points that are on the left and right lane markings
        half_width = current_waypoint.lane_width / 2.0
        left_waypoint = np.array([current_waypoint.pose.position.x, current_waypoint.pose.position.y]) + half_width * left_v
        right_waypoint = np.array([current_waypoint.pose.position.x, current_waypoint.pose.position.y]) + half_width * right_v
        ros_left_waypoint = Point()
        ros_right_waypoint = Point()
        ros_left_waypoint.x = left_waypoint[0]
        ros_left_waypoint.y = left_waypoint[1]
        ros_right_waypoint.x = right_waypoint[0]
        ros_right_waypoint.y = right_waypoint[1]
        return ros_left_waypoint, ros_right_waypoint

    def get_adjacent_lane_points(self, position):
        current_waypoint = self.get_waypoint(position)
        lane_width = current_waypoint.lane_width
        left_lane, right_lane = self.get_coordinate_lanemarking(position)
        if left_lane is None or right_lane is None:
            return None, None
        left_disp = np.array([left_lane.x - position.x, left_lane.y - position.y])
        left_disp = left_disp / np.linalg.norm(left_disp)
        left_point = Point()
        left_point.x = position.x + lane_width * left_disp[0]
        left_point.y = position.y + lane_width * left_disp[1]
        left_left, left_right = self.get_coordinate_lanemarking(left_point)
        if left_left is None or left_right is None:
            return None, None
        left_midpoint = self.get_mid_waypoint(left_left, left_right)

        right_disp = np.array([right_lane.x - position.x, right_lane.y - position.y])
        right_disp = right_disp / np.linalg.norm(right_disp)
        right_point = Point()
        right_point.x = position.x + lane_width * right_disp[0]
        right_point.y = position.y + lane_width * right_disp[1]
        right_left, right_right = self.get_coordinate_lanemarking(right_point)
        if right_left is None or right_right is None:
            return None, None
        right_midpoint = self.get_mid_waypoint(right_left, right_right)

        return left_midpoint, right_midpoint

    def get_waypoint(self, location):
        """
        Helper to get waypoint from a ros service
        """
        try:
            response = self._get_waypoint_client(location)
            return response.waypoint
        except (rospy.ServiceException, rospy.ROSInterruptException) as e:
            if not rospy.is_shutdown:
                rospy.logwarn("Service call failed: {}".format(e))

    def odometry_updated(self, odo):
        """
        Callback on new odometry
        """
        self._current_speed = math.sqrt(odo.twist.twist.linear.x ** 2 +
                                        odo.twist.twist.linear.y ** 2 +
                                        odo.twist.twist.linear.z ** 2) * 3.6

        self._current_pose = odo.pose.pose
        quaternion = (
            odo.pose.pose.orientation.x,
            odo.pose.pose.orientation.y,
            odo.pose.pose.orientation.z,
            odo.pose.pose.orientation.w
        )
        _, _, self._vehicle_yaw = euler_from_quaternion(quaternion)

    def _init_controller(self, opt_dict):
        """
        Controller initialization.

        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        args_lateral_dict = {
            'K_P': 1.95,
            'K_D': 0.01,
            'K_I': 1.4}
        args_longitudinal_dict = {
            'K_P': 0.2,
            'K_D': 0.05,
            'K_I': 0.1}

        # parameters overload
        if opt_dict:
            if 'lateral_control_dict' in opt_dict:
                args_lateral_dict = opt_dict['lateral_control_dict']
            if 'longitudinal_control_dict' in opt_dict:
                args_longitudinal_dict = opt_dict['longitudinal_control_dict']

        self._vehicle_controller = VehiclePIDController(args_lateral=args_lateral_dict,
                                                        args_longitudinal=args_longitudinal_dict)

    def set_global_plan(self, current_plan):
        """
        set a global plan to follow
        """
        self.target_route_point = None
        self._waypoint_buffer.clear()
        self._waypoints_queue.clear()
        for elem in current_plan:
            self._waypoints_queue.append(elem.pose)

    def get_mid_waypoint(self, point1, point2):
        midpoint = Point()
        midpoint.x = (point1.x + point2.x) / 2.0
        midpoint.y = (point1.y + point2.y) / 2.0
        midpoint.z = (point1.z + point2.z) / 2.0

        return midpoint

    def check_front_obstacle(self):
        buffer = []
        collision = False
        for i in range(5):
            pose = self._waypoint_buffer.popleft()
            buffer.append(pose)
            left_lane_point, right_lane_point = self.get_coordinate_lanemarking(pose.position)
            waypoint = self.get_waypoint(pose.position)
            collision = collision or self.check_waypoint_obstacles(waypoint.pose.position)
            if left_lane_point is None or right_lane_point is None:
                continue    
            left_midpoint = self.get_mid_waypoint(left_lane_point, pose.position)
            right_midpoint = self.get_mid_waypoint(right_lane_point, pose.position)
            left_lane_waypoint = self.get_waypoint(left_midpoint)
            right_lane_waypoint = self.get_waypoint(right_midpoint)
            collision = collision or self.check_waypoint_obstacles(left_lane_waypoint.pose.position)
            collision = collision or self.check_waypoint_obstacles(right_lane_waypoint.pose.position)
        for i in range(5):
            self._waypoint_buffer.appendleft(buffer.pop())

        return collision

    def check_adjacent_lanes_obstacles(self):
        buffer = []
        result = [False, False]
        for i in range(5):
            pose = self._waypoint_buffer.popleft()
            buffer.append(pose)
            waypoint = self.get_waypoint(pose.position)
            left_point, right_point = self.get_adjacent_lane_points(waypoint.pose.position)
            if left_point is None or right_point is None:
                continue
            left_waypoint = self.get_waypoint(left_point)
            right_waypoint = self.get_waypoint(right_point)

            result[0] = result[0] or self.check_waypoint_obstacles(left_waypoint.pose.position)
            result[1] = result[1] or self.check_waypoint_obstacles(right_waypoint.pose.position)
            print(result)
        for i in range(5):
            self._waypoint_buffer.appendleft(buffer.pop())
        
        return result

    def change_lane_left(self):
        buffer = []
        target_buffer = []
        for i in range(5):
            pose = self._waypoint_buffer.popleft()
            buffer.append(pose)
            if i > 0:
                waypoint = self.get_waypoint(pose.position)
                left_point, right_point = self.get_adjacent_lane_points(waypoint.pose.position)
                if left_point is None or right_point is None:
                    target_buffer.append(waypoint.pose)
                    continue
                left_waypoint = self.get_waypoint(left_point)
                target_buffer.append(left_waypoint.pose)
        for i in range(4):
            self._waypoint_buffer.appendleft(target_buffer.pop())

        midpoint = self.get_mid_waypoint(self._current_waypoint.pose.position, self._waypoint_buffer[0].position)
        waypoint = self.get_waypoint(midpoint)
        self._waypoint_buffer.appendleft(waypoint.pose)

        self._changing_lane = True
        self._lane_delta += 1
        self._lane_change_history.append(1)


    def change_lane_right(self):
        buffer = []
        target_buffer = []
        for i in range(5):
            pose = self._waypoint_buffer.popleft()
            buffer.append(pose)
            if i > 0:
                waypoint = self.get_waypoint(pose.position)
                left_point, right_point = self.get_adjacent_lane_points(waypoint.pose.position)
                if left_point is None or right_point is None:
                    target_buffer.append(waypoint.pose)
                    continue
                right_waypoint = self.get_waypoint(right_point)
                target_buffer.append(right_waypoint.pose)
        for i in range(4):
            self._waypoint_buffer.appendleft(target_buffer.pop())

        midpoint = self.get_mid_waypoint(self._current_waypoint.pose.position, self._waypoint_buffer[0].position)
        waypoint = self.get_waypoint(midpoint)
        self._waypoint_buffer.appendleft(waypoint.pose)

        self._changing_lane = True
        self._lane_delta -= 1
        self._lane_change_history.append(-1)

         
    def can_return(self):
        last_lane_change = self._lane_change_history[-1]
        left_point, right_point = self.get_adjacent_lane_points(self._current_waypoint.pose.position)
        if left_point is None or right_point is None:
            return False
        if last_lane_change == -1:
            left_waypoint = self.get_waypoint(left_point)
            passed = not self.check_waypoint_obstacles(left_waypoint.pose.position) 
        else:
            right_waypoint = self.get_waypoint(right_point)
            passed = not self.check_waypoint_obstacles(right_waypoint.pose.position)
        result = self.check_adjacent_lanes_obstacles()
        return passed and not result[(last_lane_change+1)/2]

    def return_lane(self):
        last_lane_change = self._lane_change_history[-1]
        if last_lane_change == -1:
            self.change_lane_left()
        else:
            self.change_lane_right()

        self._lane_change_history.pop()
        self._lane_change_history.pop()

        self._passing = False
        

    def keep_straight(self):
        if self._lane_delta == 0:
            return
        pose = self._waypoint_buffer.pop()
        waypoint = self.get_waypoint(pose.position)
        if self._lane_delta > 0:
            for i in range(abs(self._lane_delta)):
                left_point, right_point = self.get_adjacent_lane_points(waypoint.pose.position)
                if left_point is None or right_point is None:
                    continue
                left_waypoint = self.get_waypoint(left_point)
                waypoint = left_waypoint
        else:
            for i in range(abs(self._lane_delta)):
                left_point, right_point = self.get_adjacent_lane_points(waypoint.pose.position)
                if left_point is None or right_point is None:
                    continue
                right_waypoint = self.get_waypoint(right_point)
                waypoint = right_waypoint
        self._waypoint_buffer.append(waypoint.pose)
        

    def changing_lane(self):
        self.keep_straight()
        self._changing_lane = False

    def check_side(self):
        last_lane_change = self._lane_change_history[-1]
        left_point, right_point = self.get_adjacent_lane_points(self._current_waypoint.pose.position)
        if left_point is None or right_point is None:
            return
        if last_lane_change == -1:
            if self.check_waypoint_obstacles(left_point):
                print("PASSED LEFT"))
                self._passing = False
        else:
            if self.check_waypoint_obstacles(right_point):
                self._passing = False
                print("PASSED RIGHT")


    def run_step(self, target_speed, current_speed, current_pose):
        """
        Execute one step of local planning which involves running the longitudinal
        and lateral PID controllers to follow the waypoints trajectory.
        """
        if not self._waypoint_buffer and not self._waypoints_queue:
            control = CarlaEgoVehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False

            rospy.loginfo("Route finished.")
            return control, True

        #   Buffering the waypoints
        if len(self._waypoint_buffer) < 5:
            self._buffer_updated = True
            for i in range(self._buffer_size - len(self._waypoint_buffer)):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break
        else:
            self._buffer_updated = False

        # current vehicle waypoint
        self._current_waypoint = self.get_waypoint(current_pose.position)

        # get a list of obstacles surrounding the ego vehicle
        self.get_obstacles(current_pose.position, 70.0)

        # Example 1: get two waypoints on the left and right lane marking w.r.t current pose
        # left, right = self.get_coordinate_lanemarking(current_pose.position)
        # print("\x1b[6;30;33m------Example 1------\x1b[0m")
        # print("Left: {}, {}; right: {}, {}".format(left.x, left.y, right.x, right.y))
        
        # # Example 2: check obstacle collision
        # print("\x1b[6;30;33m------Example 2------\x1b[0m")
        # point = Point()
        # point.x = 100.0
        # point.y = 100.0
        # point.z = 1.5
        # for ob in self._obstacles:
        #     print("id: {}, collision: {}".format(ob.id, self.check_obstacle(point, ob)))
        
        # target waypoint
        if len(self._waypoint_buffer) >= 5 and self._buffer_updated:
            if self._changing_lane:
                self.changing_lane()
                print("CHANGING LANE")
            else:
                self.keep_straight()
                if self._passing:
                    self.check_side()

                elif self._lane_delta != 0 and not self._passing:
                    if self.can_return():
                        self.return_lane()
                        print("RETURNING LANE")

                elif not self._changing_lane and self.check_front_obstacle():
                    print("OBSTACLE DECTECTED FRONT")
                    result = self.check_adjacent_lanes_obstacles()
                    if not result[0]:
                        self.change_lane_left()
                        self._passing = True
                        print("CHANGE LANE LEFT")
                    elif not result[1]:
                        self.change_lane_right()
                        print("CHANGE LANE RIGHT")
                        self._passing = True
                    else:
                        print("NO WHERE TO GO!")
                        target_speed = 10





            # target_route_point = self._waypoint_buffer.popleft()
            # target_waypoint = self.get_waypoint(target_route_point.position)
            # future_route_point = self._waypoint_buffer.popleft()
            # future_waypoint = self.get_waypoint(future_route_point.position)
            # far_route_point = self._waypoint_buffer.popleft()
            # far_waypoint = self.get_waypoint(far_route_point.position)

            # # print("Distance to next waypoint:  ", math.sqrt((target_route_point.position.x-current_pose.position.x)**2 + (target_route_point.position.y-current_pose.position.y)**2 + (target_route_point.position.z-current_pose.position.z)**2))
            
            # far_left, far_right = self.get_coordinate_lanemarking(far_waypoint.pose.position)
            # far_left_waypoint = self.get_waypoint(far_left)
            # far_right_waypoint = self.get_waypoint(far_right)
            # future_left, future_right = self.get_coordinate_lanemarking(future_waypoint.pose.position)
            # future_left_waypoint = self.get_waypoint(future_left)
            # future_right_waypoint = self.get_waypoint(future_right)

            # if self._current_waypoint.lane_id == far_waypoint.lane_id:
            #     if self.check_waypoint_obstacles(future_waypoint.pose.position):
            #         if not self.check_waypoint_obstacles(far_left_waypoint.pose.position):
            #             self._waypoint_buffer.appendleft(far_left_waypoint.pose)
            #             self._waypoint_buffer.appendleft(future_left_waypoint.pose)
            #             target_route_point = self.get_mid_waypoint(current_pose, future_left_waypoint.pose)
            #             target_waypoint = self.get_waypoint(target_route_point.position)
            #             self._waypoint_buffer.appendleft(target_waypoint.pose)
            #         elif not self.check_waypoint_obstacles(future_right_waypoint.pose.position):
            #             self._waypoint_buffer.appendleft(far_right_waypoint.pose)
            #             self._waypoint_buffer.appendleft(future_right_waypoint.pose)
            #             target_route_point = self.get_mid_waypoint(current_pose, future_right_waypoint.pose)
            #             target_waypoint = self.get_waypoint(target_route_point.position)
            #             self._waypoint_buffer.appendleft(target_waypoint.pose)
            #         else:
            #             print("FUCK! NOWHERE TO GO!!!!")
            
            # else:
            #     if self.check_waypoint_obstacles(far_waypoint.pose.position):
            #         if self._current_waypoint.lane_id == far_left_waypoint.lane_id:
            #             if not self.check_waypoint_obstacles(far_left_waypoint.pose.position):
            #                 self._waypoint_buffer.appendleft(far_left_waypoint.pose)
            #                 self._waypoint_buffer.appendleft(future_left_waypoint.pose)
            #                 target_route_point = self.get_mid_waypoint(current_pose, future_left_waypoint.pose)
            #                 target_waypoint = self.get_waypoint(target_route_point.position)
            #                 self._waypoint_buffer.appendleft(target_waypoint.pose)
            #             else:
            #                 print("FUCK! NOWHERE TO GO!!!!")
            #         elif self._current_waypoint.lane_id == far_right_waypoint.lane_id:
            #             if not self.check_waypoint_obstacles(far_right_waypoint.pose.position):
            #                 self._waypoint_buffer.appendleft(far_right_waypoint.pose)
            #                 self._waypoint_buffer.appendleft(future_right_waypoint.pose)
            #                 target_route_point = self.get_mid_waypoint(current_pose, future_right_waypoint.pose)
            #                 target_waypoint = self.get_waypoint(target_route_point.position)
            #                 self._waypoint_buffer.appendleft(target_waypoint.pose)
            #             else:
            #                 print("FUCK! NOWHERE TO GO!!!!")
            #         else:
            #             print("WTF???? WHERE AM I?!")


        self.target_route_point = self._waypoint_buffer[0]
        target_point = PointStamped()
        target_point.header.frame_id = "map"
        target_point.point.x = self.target_route_point.position.x
        target_point.point.y = self.target_route_point.position.y
        target_point.point.z = self.target_route_point.position.z
        self._target_point_publisher.publish(target_point)
        
        # move using PID controllers
        control = self._vehicle_controller.run_step(
            target_speed, current_speed, current_pose, self.target_route_point)

        # purge the queue of obsolete waypoints
        max_index = -1

        sampling_radius = target_speed * 1 / 3.6  # 1 seconds horizon
        min_distance = sampling_radius * self.MIN_DISTANCE_PERCENTAGE

        for i, route_point in enumerate(self._waypoint_buffer):
            if distance_vehicle(
                    route_point, current_pose.position) < min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()

        return control, False
