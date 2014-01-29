#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define FEATURE_LENGTH 10

float __constant densities[3] = { 0.0, 2.0, 5.0 };
float __constant speeds[3] = { 0.0, 0.015, 0.025 };
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
  float * feature 
) {
  int density = 0;
  float angSum = 0.0;
  float magSum = 0.0;

  for ( int i = 0; i < FEATURE_LENGTH; i++ ) {
    feature[i] = 0.;
  }
  for ( int i = 0; i < frameSize; i++ ) {
    float2 other = frame[i].lo;
    float2 xRel  = other - position;
    float xLen   = length( xRel );
    if ( xLen < radius ) {
      density += 1;
      float2 vRel = frame[i].hi - velocity;
      float vLen = length( vRel );
      float a = dot( vRel / vLen, xLen / xLen );
      angSum += a;
      magSum += vLen;
    }
  }



  if ( density > 0 ) {
    feature[maxIdx( density, densities, 3)] = 1;

    float speed = magSum / density;
    feature[3 + maxIdx( speed, speeds, 3 )] = 1;

    float cosine = angSum / density;
    feature[6 + maxIdx( cosine, angles, 3 )] = 1;
  }
  feature[9] = 1.0;
}

__kernel void computeFeatures( 
  float speed, float delta, float radius, 
  uint width, uint height,
  uint frameSize, __constant float4 * frame, 
  __global float * features
) {
  unsigned int direction = get_global_id( 0 );
  unsigned int row       = get_global_id( 1 );
  unsigned int column    = get_global_id( 2 );

  float2 dir      = normalize( directions[direction] );
  float2 velocity = dir * speed; 
  float2 position = (float2)( column * delta, row * delta );
  float f[9];
  
  computeFeature( position, velocity, radius, frameSize, frame, f );

  int base =  direction * width * height * FEATURE_LENGTH 
            + row * width * FEATURE_LENGTH
            + column * FEATURE_LENGTH;
             
  for ( int i = 0; i < FEATURE_LENGTH; i++ ) {
    features[base + i] = f[i];
  }
}
