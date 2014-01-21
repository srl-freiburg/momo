#pragma OPENCL EXTENSION cl_khr_fp64: enable

void computeFeature( 
  float2 position, float radius,
  uint frameSize, __constant float4 * frame, 
  float * feature 
) {
  *feature = 0.0;
  for ( int i = 0; i < frameSize; ++i ) {
    float2 otherX = frame[i].lo;
    float2 relX = otherX - position;
    float dist = length( relX );
    if ( dist < radius ) {
      *feature = 1.0;
    }
  }
}

__kernel void computeFeatures( 
  float delta, float radius, 
  uint width, uint height,
  uint frameSize, __constant float4 * frame, 
  __global float * features
) {
  unsigned int direction = get_global_id( 0 );
  unsigned int row       = get_global_id( 1 );
  unsigned int column    = get_global_id( 2 );

  float2 position = (float2)( column * delta, row * delta );
  float f;
  
  computeFeature( position, radius, frameSize, frame, &f );

  int base =  direction * width * height 
            + row * width
            + column;
             
  features[base] = f;
}
