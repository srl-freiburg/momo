import numpy as np
import sys
import time
import momo
from momo.learning.max_ent.compute_cummulated import *
from math import *

def learn( feature_module, convert, frame_data, ids, radius, h ):
  feature_length = feature_module.FEATURE_LENGTH

  compute_costs = feature_module.compute_costs( convert )
  planner = momo.irl.planning.forward_backward( convert, compute_costs )
  compute_features = feature_module.compute_features( convert, radius )
  accum = compute_cummulated()

  sys.stderr.write( "Initializing weight vector\n" )
  observed_integral = []
  grid_paths = []
  sum_obs = np.zeros( feature_length, np.float64 )
  count = 0.0
  for fd in frame_data:
    observed_integral.append( {} )
    grid_paths.append( {} )
    for o_id in ids:
      sys.stderr.write( "." )
      states = fd[o_id]["states"]
      frames = fd[o_id]["frames"]
      obs, path = compute_observed( feature_module, convert, states, frames, radius )
      observed_integral[-1][o_id] = obs
      grid_paths[-1][o_id] = path
      sum_obs += obs[-1]
      count += len( path )

  sys.stderr.write( "[OK]\n" )

  # Initialize weight vector
  w  = np.ones( feature_length ).astype( np.float64 )
  for i in xrange( feature_length ):
    w[i] = exp( - ( sum_obs[i] + 1.0 ) / ( count + 2.0 ) )
    #w[i] = 1.0 / count
  w /= np.sum( w )

  sys.stderr.write( "count: %f\n" % count )
  sys.stderr.write( "observed:" + str( sum_obs ) + "\n" )
  sys.stderr.write( "w:" + str( w ) + "\n" )

  gammas = np.ones( feature_length, np.float64 ) * 0.5
  old_gradient = None


  gamma = 0.5
  decay = 0.95
  min_w = None
  min_e = 1E6

  np.set_printoptions( precision = 8, suppress = True )

  sys.stderr.write( "Entering main loop\n" )
  for times in xrange( 5 ):

    sum_obs = np.zeros( feature_length, np.float64 )
    sum_exp = np.zeros( feature_length, np.float64 )

    frame_idx = 0

    for fd in frame_data:
      obs_integral = observed_integral[frame_idx]
      gp = grid_paths[frame_idx]
      frame_idx += 1
      for o_id in ids:
        states = fd[o_id]["states"]
        frames = fd[o_id]["frames"]
        l = len( states )
        for i in xrange( max( l - h, 1 ) ):
          expected, cummulated, costs =\
            momo.learning.max_ent.compute_expectations( 
              states[i:], frames[i:], w, h,
              convert, compute_costs, planner, compute_features, accum
            )
          observed = obs_integral[o_id][min( i + h, l - 1 )] * 1
          if i > 0:
            observed -= obs_integral[o_id][i - 1]
          sum_obs += observed
          sum_exp += expected

          if np.any( np.isnan( expected ) ):
            sys.stderr.write( "x" )
            continue
          if np.sum( observed ) != 0 and np.sum( expected ) != 0:
            gradient = observed / np.sum( observed ) - expected / np.sum( expected )
            sys.stderr.write( "." )
          else:
            gradient = observed * 0.
            sys.stderr.write( "x" )
          error = np.linalg.norm( gradient )
          #momo.plot.gradient_descent_step( cummulated, costs, gp[o_id], error )
        sys.stderr.write( "\n" )


    
    s_obs = sum_obs / np.sum( sum_obs )
    s_exp = sum_exp / np.sum( sum_exp )
    w = w / np.sum( w )
    gradient = s_obs - s_exp
    error = np.linalg.norm( gradient )
    print "Result:", w, "Error:", error
    print "Observed", s_obs
    print "Expected", s_exp
    print "Gradient", gradient
    print times, error



    
    if old_gradient != None:
      for i in xrange( feature_length ):
        if gradient[i] * old_gradient[i] > 0:
          gammas[i] *= 1.2
        elif gradient[i] * old_gradient[i] < 0:
          gammas[i] *= 0.5
          gradient[i] = 0
    old_gradient = gradient
    print "gammas", gammas
      

    for i in xrange( feature_length ):
      w[i] *= exp( - gammas[i] * gradient[i] )
    w /= np.sum( w )


    #if np.sum( sum_obs ) != 0 and np.sum( sum_exp ) != 0:
      #gradient = sum_obs / np.sum( sum_obs ) - sum_exp / np.sum( sum_exp )
    #error = np.linalg.norm( gradient )
    if error < min_e:
      min_e = error
      min_w = w
    #if error < 0.05:
      #break
    #for i in xrange( feature_length ):
      #w[i] *= exp( -gamma * decay ** times * gradient[i] )
      ##w[i] *= exp( -gamma * gradient[i] )
    #w /= np.sum( w )

  print min_w
  return min_w


def compute_observed( feature_module, convert, states, frames, radius ):
  l = len( states )
  grid_path = [convert.from_world2( s ) for s in states]
  repr_path = [convert.to_world2( convert.from_world2( s ), np.linalg.norm( s[2:] ) ) for s in states]
  result = []
  for i in xrange( len( states ) ):
    result.append( feature_module.compute_feature( states[i], frames[i], radius ) )
    if i > 0:
      result[i] += result[i- 1]
  return result, grid_path

