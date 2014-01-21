import pyopencl as cl
import numpy as np
from math import *
import momo
from momo.features import *

import sys

class compute_features( momo.opencl.Program ):
  def __init__( self, convert, radius ):
    momo.opencl.Program.__init__( self )
    self.kernel = self.loadProgram( momo.BASE_DIR + "/opencl/test.cl" )

    self.convert = convert
    self.radius  = radius

  def __call__( self, frame ):
    mf = cl.mem_flags
    features = np.zeros( (8, self.convert.grid_height, self.convert.grid_width ), dtype=np.float32 )

    f = self.convert.rebase_frame( frame ).astype( np.float32 )

    frame_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = f )
    feature_buffer = cl.Buffer( self.context, mf.WRITE_ONLY, features.nbytes )

    self.kernel.computeFeatures( 
      self.queue, features.shape, None, 
      np.float32( self.convert.delta ), np.float32( self.radius ),
      np.int32( self.convert.grid_width ), np.int32( self.convert.grid_height ),
      np.int32( frame.shape[0] ), frame_buffer, 
      feature_buffer 
    )

    cl.enqueue_copy( self.queue, features, feature_buffer )
    return features

