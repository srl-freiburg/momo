#!/usr/bin/env python
import rospy
import numpy as np
from pedsim_msgs.msg import AgentState
from pedsim_msgs.msg import AllAgentsState
from pedsim_srvs.srv import SetAgentState
import ast

import sys
import os

BASE_DIR = os.path.abspath( os.path.join( os.path.dirname( __file__ ), ".." ) )
path     = os.path.abspath( os.path.join( BASE_DIR, "python" ) )
sys.path.append( path )

import momo

class Params( object ): pass

def param( name, default = None ):
  if default != None:
    return rospy.get_param( "/" + rospy.get_name() + "/"  + name, default )
  else:
    return rospy.get_param( "/" + rospy.get_name() + "/"  + name )

def get_params():
  result = Params()
  result.target_id = param( "target_id" )
  result.feature_type = param( "feature_type" )
  result.feature_params = ast.literal_eval( param( "feature_params" ) )
  result.weights = np.array( ast.literal_eval( param( "weights" ) ), dtype = np.float64 )
  result.goal = np.array( ast.literal_eval( param( "goal" ) ), dtype = np.float64 )
  result.goal_threshold = param( "speed" )
  result.speed = param( "goal_threshold" )
  result.cell_size = param( "cell_size" )
  result.x1   = param( "x1" )
  result.x2   = param( "x2" )
  result.y1   = param( "y1" )
  result.y2   = param( "y2" )
  result.max_msg_age = param( "max_msg_age" )
  return result

def plan( weights, feature_type, feature_params, x1, y1, x2, y2, cell_size, robot, other, goal, speed ):
  # Build planning objects
  convert = momo.convert( { "x1": x1, "y1": y1, "x2": x2, "y2": y2 }, cell_size )
  features = momo.features.__dict__[feature_type]
  compute_features = features.compute_features( convert, **feature_params )
  compute_costs = momo.features.compute_costs( convert )
  planner = momo.planning.dijkstra()

  # Compute features and costs
  f = compute_features( speed, other )
  costs = compute_costs( f, weights )

  # Plan

  current = convert.from_world2( robot )
  goal = convert.from_world2( goal )

  cummulated, parents = planner( costs, goal )
  path = planner.get_path( parents, current )

  world_path = []
  for p in path:
    world_path.append( convert.to_world2( p, speed ) )

  interpolated_path = []
  i = 0
  current = robot * 1.0


  while True:
    current_cell = convert.from_world2( current )
    next_cell = convert.from_world2( world_path[i] )
    if ( current_cell[:2] == next_cell[:2] ).all():
      i += 1
    if not i < len( world_path ):# or i > 1:
      sys.stderr.write( "Breaking with i:= (%d, %d)\n" % (i,  len(world_path) ) )
      break
    current[2:] = world_path[i][:2] - current[:2] 
    current[2:] = speed * current[2:] / np.linalg.norm( current[2:] )
    current[:2] += current[2:]
    interpolated_path.append( current * 1.0 )

  # if len( interpolated_path ) > 1:
    # sys.stderr.write( "Current: %f, %f\n" % ( robot[0], robot[1] ) )
    # sys.stderr.write( "Path[0]: %f, %f\n" % ( interpolated_path[1][0], interpolated_path[1][1] ) )
  return interpolated_path


def set_agent_state( target_id, x, y, vx, vy ):
    a = AgentState()
    a.id = target_id
    a.position.x = x
    a.position.y = y
    a.velocity.x = vx
    a.velocity.y = vy

    rospy.loginfo( "Waiting to send command: %f, %f, %f, %f" % ( x, y, vx, vy ) )
    rospy.wait_for_service( "SetAgentState" )
    try:
      set_agent_status = rospy.ServiceProxy( "SetAgentState", SetAgentState )
      result = set_agent_status( a )
    except rospy.ServiceException, e:
      rospy.logerr( "Service call failed: %s" % e )
    rospy.loginfo( "Command sent" )

def callback( data ):
  parms = get_params()
  if rospy.get_rostime().to_sec() - data.header.stamp.to_sec() > parms.max_msg_age: 
    return

  other = []
  robot  = None

  for a in data.agent_states:
    v = np.array( [a.position.x, a.position.y, a.velocity.x, a.velocity.y], dtype = np.float64 )
    if a.id == parms.target_id:
      robot = v
    else:
      other.append( v )
  other = np.array( other )

  path = plan( 
    parms.weights, parms.feature_type, parms.feature_params, 
    parms.x1, parms.y1, parms.x2, parms.y2, parms.cell_size, 
    robot, other, parms.goal, parms.speed 
  )

  distance = 0.0
  if len( path ) > 2:
    distance = np.linalg.norm( robot[:2] - path[2][:2] )

  if distance > parms.goal_threshold:
    set_agent_state( parms.target_id, robot[0], robot[1], path[1][2], path[1][3] )
  else:
    set_agent_state( parms.target_id, 0, 0, 0, 0 )



def listener():
  rospy.init_node( 'irl_features' )
  p = get_params()
  rospy.Subscriber( "AllAgentsStatus", AllAgentsState, callback )
  rospy.spin()


if __name__ == '__main__':
  listener()
