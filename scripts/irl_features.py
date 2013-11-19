#!/usr/bin/env python
import rospy
import numpy as np
from pedsim_msgs.msg import AllAgentsState
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
  result.y1   = param( "y2", 41.0 )
  result.y2   = param( "y2", 41.0 )
  rospy.loginfo( result.goal )
  return result


def callback( data ):
  parms = get_params()

  # Build planning objects
  convert = momo.convert( { "x1": parms.x1, "y1": parms.y1, "x2": parms.x2, "y2": parms.y2 }, cell_size )
  features = momo.features.__dict__[parms.feature_type]
  compute_features = features.compute_features( convert, **parms.feature_params )
  compute_costs = momo.features.compute_costs( convert )
  planner = momo.planning.dijkstra()

  # Compute features and costs
  other = []
  robot  = None
  for a in data.data.agent_states:
    v = np.array( [a.position.x, a.position.y, a.velocity.x, a.velocity.y], dtype = np.float64 )
    if a.id == target_id:
      robot = v
    else:
      other.append( v )
  other = np.array( other )

  f = compute_features( np.linalg.norm( robot[2:] ), other )
  costs = compute_costs( f, parms.weights )

  # Plan

  current = convert.from_world2( robot )
  goal = convert.from_world2( parms.goal )


  cummulated, parents = self.planner( costs, goal )
  path = self.planner.get_path( parents, current )

  rospy.loginfo( path )


  #rospy.loginfo( data.data )


def listener():
  rospy.init_node( 'irl_features' )
  p = get_params()
  rospy.Subscriber( "AllAgentsStatus", AllAgentsState, callback )
  rospy.spin()


if __name__ == '__main__':
  listener()
