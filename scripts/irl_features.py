#!/usr/bin/env python
import rospy
import numpy as np
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
  result.target_id = param( "target_id", 1 )
  result.feature_type = param( "feature_type", "smoke0" )
  result.feature_params = ast.literal_eval( param( "feature_params" ) )
  result.weights = np.array( ast.literal_eval( param( "weights" ) ), dtype = np.float64 )
  result.goal = np.array( ast.literal_eval( param( "goal" ) ), dtype = np.float64 )
  result.cell_size = param( "cell_size", 1 )
  result.x1   = param( "x1", 0.0 )
  result.x2   = param( "x2", 0.0 )
  result.y1   = param( "y1", 41.0 )
  result.y2   = param( "y2", 41.0 )
  return result


def callback( data ):
  parms = get_params()

  # Build planning objects
  convert = momo.convert( { "x1": parms.x1, "y1": parms.y1, "x2": parms.x2, "y2": parms.y2 }, parms.cell_size )
  features = momo.features.__dict__[parms.feature_type]
  compute_features = features.compute_features( convert, **parms.feature_params )
  compute_costs = momo.features.compute_costs( convert )
  planner = momo.planning.dijkstra()

  # Compute features and costs
  other = []
  robot  = None
  for a in data.agent_states:
    v = np.array( [a.position.x, a.position.y, a.velocity.x, a.velocity.y], dtype = np.float64 )
    if a.id == parms.target_id:
      robot = v
    else:
      other.append( v )
  other = np.array( other )

  speed = np.linalg.norm( robot[2:] )
  f = compute_features( speed, other )
  costs = compute_costs( f, parms.weights )

  # Plan

  current = convert.from_world2( robot )
  goal = convert.from_world2( parms.goal )


  rospy.loginfo( "Origin: %f, %f; Goal: %f %f" % ( current[0], current[1], goal[0], goal[1] ) )


  cummulated, parents = planner( costs, goal )
  path = planner.get_path( parents, current )

  result = []
  for p in path:
    result.append( convert.to_world2( p, speed ) )

  rospy.loginfo( "0: %f, %f; -1: %f, %f" % ( path[0][0], path[0][1], path[-1][0], path[-1][1] ) )



def listener():
  rospy.init_node( 'irl_features' )
  p = get_params()
  rospy.Subscriber( "AllAgentsStatus", AllAgentsState, callback )
  rospy.spin()


if __name__ == '__main__':
  listener()
