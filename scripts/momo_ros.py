#!/usr/bin/env python

__author__ = 'Billy Okal <okal@cs.uni-freiburg.de>'
__version__ = '0.1'
__license__ = 'BSD'


import sys
import os
import math
import ast

import numpy as np

import rospy
from pedsim_msgs.msg import AgentState
from pedsim_msgs.msg import AllAgentsState
from nav_msgs.msg import Path, GridCells, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


# must be set before calling import momo
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
path = os.path.abspath(os.path.join(BASE_DIR, "python"))
sys.path.append(path)

import momo


class Params(object):
    pass


def param(name, default=None):
    if default is not None:
        return rospy.get_param("/pedsim" + "/" + name, default)
    else:
        return rospy.get_param("/pedsim" + "/" + name)


# TODO - clean this up into a dict of parameters
def get_params():
    result = Params()
    result.target_type = param("target_type")
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
    result.update_type = param("update_type")
    result.sensor_range = param("sensor_range")
    return result


class MomoROS(object):

    """
    ROS interface for momo

    """

    def __init__(self):
        self.LOOKAHEAD = 1
        self.OBSTACLES = None
        self.GOAL_REACHED = False

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

        self.costs = None

        # publishers
        self.pub_plan = rospy.Publisher('planned_path', Path)
        self.pub_cost = rospy.Publisher('costmap', OccupancyGrid)
        self.pub_goal_status = rospy.Publisher('goal_status', String)
        self.pub_agent_state = rospy.Publisher('robot_state', AgentState)

        # subscribers
        rospy.Subscriber("dynamic_obstacles", AllAgentsState,
                         self.callback_agent_status)
        rospy.Subscriber("static_obstacles", GridCells,
                         self.callback_obstacles, queue_size=1)

    def publish_path(self, plan):
        p = list()
        if plan is not None:
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
        """ publish_costmap( costs, cell_size )

        Publish the costmap derived from the learned weights

        """
        cc = costs[0] * 1.0
        # cc = (costs[0] + costs[1] + costs[2] + costs[3] + costs[4] + costs[5] + costs[6] + costs[7]) * (1.0/8.0)

        cc *= 1000.0 / np.max(cc)
        cc = cc.astype(np.int8)
        w, h = cc.shape
        c = np.reshape(cc, w * h)

        ocg = OccupancyGrid()
        ocg.header.stamp = rospy.Time.now()
        ocg.header.frame_id = "world"
        ocg.data = c
        ocg.info.resolution = cell_size
        ocg.info.width = h
        ocg.info.height = w
        self.pub_cost.publish(ocg)

    def publish_goal_status(self):
        if self.GOAL_REACHED is True:
            self.pub_goal_status.publish('Arrived')
            os.system('rosnode kill --all')
        else:
            self.pub_goal_status.publish('Travelling')

    def publish_robot_state(self, target_type, x, y, vx, vy):
        a = AgentState()
        a.type = target_type
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

    def feature_at_cell(self, features, cell):
        """
        Raw binary feature in each direction at a particular cell
        """
        return features[:, cell[0], cell[1], :]

    def cost_at_cell(self, costs, cell):
        """
        Cost in each direction at a particular cell
        """
        return costs[:, cell[0], cell[1]]

    def within_grid(self, cell):
        """
        Check if a cell is within the grid
        """
        if (cell[0] >= self.params.x1 and cell[0] <= self.params.x2) and \
                (cell[1] >= self.params.y1 and cell[1] <= self.params.y2):
            return True
        else:
            return False

    def distance_between(self, cella, cellb):
        return math.sqrt((cella[0] - cellb[0]) ** 2 + (cella[1] - cellb[1]) ** 2)

    def get_cells_in_range(self, robot, radius):
        """
        Get only the cells in the local radius of the robot

        TODO - add direction
        """
        local_cells = list()

        for i in xrange(int(robot[0]) - int(radius), int(robot[0]) + int(radius)):
            for j in xrange(int(robot[1]) - int(radius), int(robot[1]) + int(radius)):
                if self.within_grid((i, j)) is True and self.distance_between(robot, (i, j)) < radius:
                    local_cells.append((i, j))

        return local_cells

    def plan(self, weights, feature_type, feature_params,
             x1, y1, x2, y2, cell_size, robot, other, goal, speed):

        # Compute features and costs
        f = self.compute_features(speed, other)

        # update costs based on oracle/local switch
        if self.params.update_type == "oracle":
            self.costs = self.compute_costs(f, weights)
        else:
            temp_costs = self.compute_costs(f, weights)
            self.costs = np.zeros(shape=temp_costs.shape)
            lc = self.get_cells_in_range(robot, self.params.sensor_range)
            # print lc
            if len(lc) > 0:
                for cell in lc:
                    self.costs[:, cell[1], cell[0]
                               ] = temp_costs[:, cell[1], cell[0]]

        # for visualization (different thresholds for obstacles)
        viscosts = self.costs.copy()

        # bring in obstacles
        if self.OBSTACLES is not None:
            for obs in self.OBSTACLES:
                self.costs[:, obs[1] / cell_size, obs[0] / cell_size] = 1000.0
                viscosts[:, obs[1] / cell_size, obs[0] / cell_size] = 40.0

        # Plan
        current = self.convert.from_world2(robot)
        grid_goal = self.convert.from_world2(goal)

        cummulated, parents = self.planner(self.costs, grid_goal)
        path = self.planner.get_path(parents, current)

        world_path = []
        interpolated_path = []

        if path is not None:
            for p in path:
                world_path.append(self.convert.to_world2(p, speed))

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
        self.publish_costmap(viscosts, cell_size)
        self.publish_goal_status()

        return interpolated_path

    def callback_agent_status(self, data):
        # self.params = get_params()
        travel_time = rospy.get_rostime().to_sec() - data.header.stamp.to_sec()
        if travel_time > self.params.max_msg_age:
            rospy.loginfo('Skipping data which (%f)s late' % (travel_time))
            return

        other = []
        robot = None

        for a in data.agent_states:
            v = np.array(
                [a.position.x, a.position.y,
                 a.velocity.x, a.velocity.y],
                dtype=np.float64)
            if a.type == self.params.target_type:
                robot = v
            else:
                other.append(v)
        other = np.array(other)

        path = self.plan(
            self.params.weights, self.params.feature_type, self.params.feature_params,
            self.params.x1, self.params.y1, self.params.x2, self.params.y2, self.params.cell_size,
            robot, other, self.params.goal, self.params.speed
        )

        # f = self.compute_features(self.params.speed, other)
        # costs = self.compute_costs(f, self.params.weights)
        # self.publish_costmap(costs, self.params.cell_size)

        distance = 0.0
        if len(path) > self.LOOKAHEAD:
            distance = np.linalg.norm(robot[:2] - self.params.goal[:2])

        if distance > self.params.goal_threshold:
            self.publish_robot_state(
                self.params.target_type, robot[0], robot[1],
                path[self.LOOKAHEAD][2], path[self.LOOKAHEAD][3])
        else:
            self.GOAL_REACHED = True
            self.publish_robot_state(
                self.params.target_type, robot[0], robot[1], 0, 0)

    def callback_obstacles(self, data):
        self.OBSTACLES = []
        for cell in data.cells:
            # TODO - get cell size in here
            self.OBSTACLES.append([int(cell.x), int(cell.y)])


def run(args):
    rospy.init_node('momo_node')
    MomoROS()

    # start up
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down momo node"


if __name__ == '__main__':
    run(sys.argv)
