import numpy as np
from math import *
from __common__ import *
import momo
from momo.features import *

class compute_features( object ):
  def __init__( self, convert ):
    self.convert = convert

  def __call__( self, speed, frame ):
    features = np.ones( (8, self.convert.grid_height, self.convert.grid_width, FEATURE_LENGTH ), dtype=np.float32 )
    return features

