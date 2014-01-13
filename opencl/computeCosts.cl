#pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void computeCosts( 
  uint width, uint height, uint featureLength,
  __constant float2 * directions, 
  __global float * features,
  __constant float * theta, __global float * costs
) {
  unsigned int direction = get_global_id( 0 );
  unsigned int row       = get_global_id( 1 );
  unsigned int column    = get_global_id( 2 );

  float2 dir = directions[direction];

  int base =  direction * width * height * featureLength 
            + row * width * featureLength
            + column * featureLength;  

  float cost = 0;
  for ( int i = 0; i < featureLength; i++ ) {
    cost += features[base + i] * theta[i];
  }
  costs[direction * width * height + row * width + column] = cost;
}
