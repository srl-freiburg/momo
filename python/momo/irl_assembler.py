import momo
import sys
import numpy as np
import cPickle
from math import *

class irl_assembler( object ):
  def __init__( self, features, learning, radius, h, theta = None, data = None, ids = None, delta = 0.25 ):
    self.features = features
    self.learning = learning
    self.radius = float( radius )
    self.h = int( h )
    self.theta  = theta
    self.delta  = delta

    if data != None:
      self.learn( data, ids )

  def learn( self, data, ids ):
    sys.stderr.write( "Preprocessing data..." )
    frame_data = []
    self.convert = momo.convert( reduce( lambda x, y: x + y, data ), self.delta )
    for d in data:
      frame_data.append( self.__convert.preprocess_data( d, ids ) )
    sys.stderr.write( "[OK]\n" )

    if ids == None or len( ids ) == 0:
      l = len( frame_data.keys() ) / 2
      ids = range( l, l + 5 )

    sys.stderr.write( "Running learning algorithms\n" )
    self.theta = self.learning.learn( 
      self.features, self.__convert, frame_data, ids, 
      radius = self.radius, h = self.h
    )
    sys.stderr.write( "Learning done\n" )

  def set_convert( self, convert ):
    self.__convert = convert
    self.compute_costs = momo.features.compute_costs( self.__convert )
    self.planner = momo.irl.planning.dijkstra( self.__convert, self.compute_costs )

  convert = property( None, set_convert )

  def plan( self, start, goal, velocity, frames, dynamic = False ):
    if not dynamic:
      return self.planner( start, goal, velocity, frames[0], self.theta )[0]
    else:
      result = []
      count  = 0
      p = self.planner( start, goal, velocity, frames[0], self.theta )[0]
      while len( p ) > 0:
        count += 1
        result.append( p[0] )
        if len( p ) == 2:
          result.append( p[1] )
          break
        else:
          p = self.planner( p[1], goal, velocity, frames[count], self.theta )[0]
      return result

  def feature_sum( self, states, frames ):
    result = np.array( [0.] * self.features.FEATURE_LENGTH )
    states = [self.__convert.to_world2( self.__convert.from_world2( s ), np.linalg.norm( s[2:] ) ) for s in states]
    for i in xrange( len( states ) ):
      result += self.features.compute_feature( states[i], frames[i], self.radius )
    return result

  def save( self, stream ):
    print "Saved", self.features.__name__, self.learning.__name__
    cPickle.dump( [self.features.__name__, self.learning.__name__, self.radius, self.h, self.theta], stream )

  @staticmethod
  def load( stream ):
    features, learning, radius, h, theta = cPickle.load( stream )
    features = momo.features.__dict__[features.split( "." )[-1]]
    learning = momo.learning.__dict__[learning.split( "." )[-1]]
    result = irl_assembler( features, learning, radius, h, theta )
    return result

