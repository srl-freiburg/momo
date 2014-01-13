#!/usr/bin/env python

__author__ = 'Billy Okal <okal@cs.uni-freiburg.de>'
__version__ = '0.1'
__license__ = 'BSD'


import rospy
import numpy as np
from pedsim_msgs.msg import AgentState
from pedsim_msgs.msg import AllAgentsState
from nav_msgs.msg import Path, GridCells, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import ast

import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
path = os.path.abspath(os.path.join(BASE_DIR, "python"))
sys.path.append(path)

import momo


class Params(object):
    pass


def param(name, default=None):
    if default != None:
        return rospy.get_param("/" + rospy.get_name() + "/" + name, default)
    else:
        return rospy.get_param("/" + rospy.get_name() + "/" + name)


def get_params():
    result = Params()
    result.target_id = param("target_id")
    result.feature_type = param("feature_type")
    result.feature_params = ast.literal_eval(param("feature_params"))
    result.weights = np.array(
        ast.literal_eval(param("weights")), dtype=np.float64)
    result.goal = np.array(
        ast.literal_eval(param("goal")), dtype=np.float64)
    result.goal_threshold = param("goal_threshold")
    result.speed = param("speed")
    result.cell_size = param("cell_size")
    result.x1 = param("x1")
    result.x2 = param("x2")
    result.y1 = param("y1")
    result.y2 = param("y2")
    result.max_msg_age = param("max_msg_age")
    return result


class MomoROS(object):

    LOOKAHEAD = 1
    OBSTACLES = None
    GOAL_REACHED = False

    """ ROS interface for momo """

    def __init__(self):
        self.params = get_params()
        self._build_compute_objects(
            self.params.feature_type,
            self.params.feature_params,
            self.params.x1,
            self.params.x2,
            self.params.y1,
            self.params.y2,
            self.params.cell_size
        )

        # publishers
        self.pub_plan = rospy.Publisher('planned_path', Path)
        self.pub_cost = rospy.Publisher('costmap', OccupancyGrid)
        self.pub_goal_status = rospy.Publisher('goal_status', String)
        self.pub_agent_state = rospy.Publisher('robot_state', AgentState)

        # subscribers
        rospy.Subscriber("AllAgentsStatus", AllAgentsState, self.callback_agent_status)
        rospy.Subscriber("static_obstacles", GridCells, self.callback_obstacles)


    def publish_path(self, plan):
        p = []
        for item in plan:
            pose = PoseStamped()
            pose.pose.position.x = item[0]
            pose.pose.position.y = item[1]
            pose.header.stamp = rospy.Time.now()
            p.append(pose)

        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "world"
        path.poses = p
        self.pub_plan.publish(path)

    def publish_costmap(self, costs, cell_size):
        # cc = np.sum( costs, axis=0 )
        # cc = costs[0] * 1.0
        # cc *= 100.0 / np.max( cc )
        # cc = cc.astype( np.int8 )
        # w, h = cc.shape
        # c = np.reshape( cc, w * h )

        cc = costs[0] * 1.0

        # np.savetxt('cmap.txt', cc)

        # cc *= 100.0 / np.max( cc )
        cc *= 100.0 / np.max( cc )
        cc = cc.astype( np.int8 )
        w, h = cc.shape
        c = np.reshape( cc, w * h )

        # np.savetxt('cmap.txt', cc)

        ocg = OccupancyGrid()
        ocg.header.stamp = rospy.Time.now()
        ocg.header.frame_id = "world"
        ocg.data = c
        ocg.info.resolution = cell_size
        ocg.info.width = h
        ocg.info.height = w
        self.pub_cost.publish( ocg )

    def publish_goal_status(self):
        if self.GOAL_REACHED is True:
            self.pub_goal_status.publish('Arrived')
        else:
            self.pub_goal_status.publish('Travelling')

    def publish_robot_state(self, target_id, x, y, vx, vy):
        a = AgentState()
        a.id = target_id
        a.position.x = x
        a.position.y = y
        a.velocity.x = vx
        a.velocity.y = vy
        self.pub_agent_state.publish(a)


    def _build_compute_objects(self, feature_type, feature_params,
                               x1, x2, y1, y2, cell_size):
        # Build planning objects
        self.convert = momo.convert(
            {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, cell_size)
        self.features = momo.features.__dict__[feature_type]
        self.compute_features = self.features.compute_features(
            self.convert, **feature_params)
        self.compute_costs = momo.features.compute_costs(self.convert)
        self.planner = momo.planning.dijkstra()

    def plan(self, weights, feature_type, feature_params,
             x1, y1, x2, y2, cell_size, robot, other, goal, speed):

        # Compute features and costs
        f = self.compute_features(speed, other)
        costs = self.compute_costs(f, weights)
        # rospy.loginfo('cost size %d %d %d' % (costs.shape))
        # costs[:, :, :] = 1.0

        # bring in obstacles
        if self.OBSTACLES is not None:
            for obs in self.OBSTACLES:
                # costs[:, obs[1], obs[0]] = 50
                costs[:, obs[1] / cell_size, obs[0] / cell_size] = 10.0

        # Plan
        current = self.convert.from_world2(robot)
        grid_goal = self.convert.from_world2(goal)

        cummulated, parents = self.planner(costs, grid_goal)
        path = self.planner.get_path(parents, current)

        world_path = []
        for p in path:
            world_path.append(self.convert.to_world2(p, speed))

        interpolated_path = []
        i = 0
        current = robot * 1.0

        while True:
            current_cell = self.convert.from_world2(current)
            next_cell = self.convert.from_world2(world_path[i])
            if (current_cell[:2] == next_cell[:2]).all():
                i += 1

            if i > self.LOOKAHEAD + 1 or i >= len(world_path):
                break
            current[2:] = world_path[i][:2] - current[:2]
            current[2:] = speed * current[2:] / np.linalg.norm(current[2:])
            current[:2] += current[2:]
            interpolated_path.append(current * 1.0)

        self.publish_path(world_path)
        self.publish_costmap(costs, cell_size)
        self.publish_goal_status()

        return interpolated_path

    def callback_agent_status(self, data):
        parms = get_params()
        if rospy.get_rostime().to_sec() - data.header.stamp.to_sec() > parms.max_msg_age:
            return

        other = []
        robot = None

        for a in data.agent_states:
            v = np.array(
                [a.position.x, a.position.y,
                 a.velocity.x, a.velocity.y],
                dtype=np.float64)
            # TODO - change to agent type
            if a.id == parms.target_id:
                robot = v
            else:
                other.append(v)
        other = np.array(other)

        path = self.plan(
            parms.weights, parms.feature_type, parms.feature_params,
            parms.x1, parms.y1, parms.x2, parms.y2, parms.cell_size,
            robot, other, parms.goal, parms.speed
        )

        distance = 0.0
        if len(path) > self.LOOKAHEAD:
            distance = np.linalg.norm(robot[:2] - parms.goal[:2])

        if distance > parms.goal_threshold:
            self.publish_robot_state(parms.target_id, robot[0], robot[1],
                path[self.LOOKAHEAD][2], path[self.LOOKAHEAD][3])
        else:
            self.GOAL_REACHED = True
            self.publish_robot_state(parms.target_id, 0, 0, 0, 0)

    def callback_obstacles(self, data):
        self.OBSTACLES = []
        for cell in data.cells:
            # TODO - get cell size in here
            self.OBSTACLES.append([int(cell.x), int(cell.y)])
            # self.OBSTACLES.append([int(cell.x - 0.5), int(cell.y - 0.5)])
        


def run(args):
    rospy.init_node('irl_features')
    m = MomoROS()

    # start up
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"


if __name__ == '__main__':
    run(sys.argv)
