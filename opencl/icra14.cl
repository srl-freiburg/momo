#pragma OPENCL EXTENSION cl_khr_fp64: enable

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
  int binCount[3] = { 0, 0, 0 };
  float binSum[3] = { 0.0, 0.0, 0.0 };

  for ( int i = 0; i < 12; i++ ) {
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
      uint i  = maxIdx( a, angles, 3 );
      binCount[i] += 1;
      binSum[i] += vLen;
    }
  }


  if ( density > 0 ) {
    feature[maxIdx( density * 1.0, densities, 3 )] = 1;

    for ( int angle = 0; angle < 3; ++angle ) {
      if ( binCount[angle] >= 0 ) {
        float l = binSum[angle] / binCount[angle];
        feature[3 + angle * 3 + maxIdx( l, speeds, 3 )] = 1;
      }
    }
  }
}

__kernel void computeFeatures( 
  float speed, float delta, float radius, 
  uint width, uint height, uint featureLength,
  uint frameSize, __constant float4 * frame, 
  __global float * features
) {
  unsigned int direction = get_global_id( 0 );
  unsigned int row       = get_global_id( 1 );
  unsigned int column    = get_global_id( 2 );

  float2 dir      = normalize( directions[direction] );
  float2 velocity = dir * speed; 
  float2 position = (float2)( column * delta, row * delta );
  float f[12];
  
  computeFeature( position, velocity, radius, frameSize, frame, f );

  int base =  direction * width * height * featureLength 
            + row * width * featureLength
            + column * featureLength;
             
  for ( int i = 0; i < featureLength; i++ ) {
    features[base + i] = f[i];
  }
}
