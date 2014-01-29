#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define FEATURE_LENGTH 4
float __constant angles[3] = { -1, -0.7071067811865475, 0.7071067811865475 };
float2 __constant directions[8] = {
  (float2) (  1.0f,  0.0f ), 
  (float2) (  1.0f,  1.0f ), 
  (float2) (  0.0f,  1.0f ), 
  (float2) ( -1.0f,  1.0f ), 
  (float2) ( -1.0f,  0.0f ), 
  (float2) ( -1.0f, -1.0f ), 
  (float2) (  0.0f, -1.0f ), 
  (float2) (  1.0f, -1.0f )
};


uint maxIdx( float value, __constant float * reference, uint length )
{
  uint result = 0;
  for ( uint i = 0; i < length; ++i ) {
    if ( value >= reference[i] ) {
      result = i;
    }
  }
  return result;
}

void computeFeature( 
  float2 position, float2 velocity, float radius,
  uint frameSize, __constant float4 * frame, 
  float lambda,
  float * feature 
) {
  for ( int i = 0; i < FEATURE_LENGTH; i++ ) {
    feature[i] = 0.;
  }
  for ( int i = 0; i < frameSize; i++ ) {
    float2 otherX = frame[i].lo;
    float2 otherV = frame[i].hi;

    float2 xRel  = otherX - position;
    float xLen   = length( xRel );

    float2 n = normalize( xRel );
    float2 e = normalize( otherV );
    float cosPhi = dot( -n, e );  
    float force  = ( lambda + 0.5 * ( 1 - lambda ) * ( 1 + cosPhi ) ) * exp( ( 2 * radius - xLen ) );
    if ( force > 0.5 ) {
      feature[maxIdx( cosPhi, angles, 3 )] += 1;
    }
  }
  feature[3] = 1.0;
}

__kernel void computeFeatures( 
  float speed, float delta, float radius, 
  uint width, uint height,
  uint frameSize, __constant float4 * frame, 
  float lambda,
  __global float * features
) {
  unsigned int direction = get_global_id( 0 );
  unsigned int row       = get_global_id( 1 );
  unsigned int column    = get_global_id( 2 );

  float2 dir      = normalize( directions[direction] );
  float2 velocity = dir * speed; 
  float2 position = (float2)( column * delta, row * delta );
  float f[FEATURE_LENGTH];
  
  computeFeature( position, velocity, radius, frameSize, frame, lambda, f );

  int base =  direction * width * height * FEATURE_LENGTH 
            + row * width * FEATURE_LENGTH
            + column * FEATURE_LENGTH;
             
  for ( int i = 0; i < FEATURE_LENGTH; i++ ) {
    features[base + i] = f[i];
  }
}
