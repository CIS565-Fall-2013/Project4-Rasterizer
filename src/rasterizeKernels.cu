// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"
#include "glm/glm.hpp"
#include "util.h"
#include "variables.h"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

glm::vec3* framebuffer = 0;
fragment* depthbuffer = 0;
float* device_vbo = 0;   //pre-transformed
float* device_vbo_t = 0; //post-transformed
float* device_cbo = 0;
int* device_ibo = 0;
triangle* primitives = 0;

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a){
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

//Writes a given fragment to a fragment buffer at a given location
__host__ __device__ void writeToDepthbuffer(int x, int y, fragment frag, fragment* depthbuffer, ushort width, ushort height){
  if( x < width && y < height){

    depthbuffer[(y*width) + x] = frag;
  }
}

//Reads a fragment from a given location in a fragment buffer
__host__ __device__ fragment getFromDepthbuffer(int x, int y, fragment* depthbuffer, ushort width, ushort height){
  if(x<width && y<height)
  {

    return depthbuffer[(y*width) + x];
  }
  else
  {
    fragment f;
    return f;
  }
}

//Writes a given pixel to a pixel buffer at a given location
__host__ __device__ void writeToFramebuffer(int x, int y, glm::vec3 value, glm::vec3* framebuffer, ushort width, ushort height){
  if(x<width && y<height){
 
    framebuffer[(y*width) + x] = value;
  }
}

//Reads a pixel from a pixel buffer at a given location
__host__ __device__ glm::vec3 getFromFramebuffer(int x, int y, glm::vec3* framebuffer, ushort width, ushort height){
  if(x<width && y<height){

    return framebuffer[(y*width) + x];
  }else{
    return glm::vec3(0,0,0);
  }
}

//Kernel that clears a given pixel buffer with a given color
__global__ void clearImage( ushort width, ushort height, glm::vec3* image, glm::vec3 color){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * width);
    if(x<width && y<height)
    {
      image[index] = color;
    }
}

//Kernel that clears a given fragment buffer with a given fragment
__global__ void clearDepthBuffer( ushort width, ushort height, fragment* buffer, fragment frag){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * width);
    if(x<width && y<height)
    {
      fragment f = frag;
      f.position.x = x;
      f.position.y = y;
      buffer[index] = f;
    }
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, ushort width, ushort height, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * width);
  
  if(x<width && y<height){

      glm::vec3 color;      
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;     
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//TODO: Implement a vertex shader
__global__ void vertexShadeKernel(float* vbo, float* vbo_t, ushort width, ushort height, int vbosize, VertUniform uniform)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  glm::vec4 pos;
  if( index < vbosize/3 )
  {
      pos.x = vbo[ 3*index ];
      pos.y = vbo[ 3*index+1 ];
      pos.z = vbo[ 3*index+2 ];
      pos.w = 1.0f;

      pos = uniform.viewingMat * pos;
      pos = uniform.projMat * pos;

      //Perspective divide
      pos.x /= pos.w;
      pos.y /= pos.w;
      pos.z /= pos.w;

      //convert to window coordinate
      pos.x = width * ( pos.x + 1.0f ) /2.0f;
      pos.y = height * ( pos.y + 1.0f ) / 2.0f;
      //memcpy( &vbo[3*index], &pos[0], sizeof(float)*3 );
      vbo_t[3*index] = pos.x;
      vbo_t[3*index+1] = pos.y;
      vbo_t[3*index+2] = pos.z;
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;

  triangle pri;
  size_t size;
  glm::vec3 orient;
  if(index<primitivesCount)
  {
      size = sizeof(float)*3;
      memcpy( &pri.p[0][0], &vbo[3*ibo[3*(index)]   ], size );
      memcpy( &pri.p[1][0], &vbo[3*ibo[3*(index)+1] ], size );
      memcpy( &pri.p[2][0], &vbo[3*ibo[3*(index)+2] ], size );
      primitives[index] = pri;
  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, ushort width, ushort height)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  triangle pri;
  int y1, y2, y3, x1, x2, x3;

  int Dx12, Dx23, Dx31;
  int Dy12, Dy23, Dy31;
  int minx, maxx, miny, maxy;
  int C1, C2, C3;
  float Cy1, Cy2, Cy3;
  int Cx1, Cx2, Cx3;

  char base1, base2, base3;
  float zval;
  float x_alpha, z_alpha;
  float x_l, x_r, z_l, z_r, w_l, w_r;
  glm::vec2 t_l, t_r;
  glm::vec3 color_l, color_r;
  x_l = 0;
  if(index<primitivesCount)
  {
      pri = primitives[index];
      y1 = pri.p[0].y + 0.5f;
      x1 = pri.p[0].x + 0.5f;

      y2 = pri.p[1].y + 0.5f;
      x2 = pri.p[1].x + 0.5f;

      y3 = pri.p[2].y + 0.5f;
      x3 = pri.p[2].x + 0.5f;

      //delta
      Dx12 = x2 - x1;
      Dx23 = x3 - x2;
      Dx31 = x1 - x3;

      Dy12 = y2 - y1;
      Dy23 = y3 - y2;
      Dy31 = y1 - y3;

      //Bounding coordinate
      minx = min( min( x1, x2 ),x3 );
      maxx = max( max( x1, x2 ), x3 );
      miny = min( min( y1, y2 ), y3 );
      maxy = max( max( y1, y2 ), y3 );
      if( minx < 0 ) minx = 0;
      if( miny < 0 ) miny = 0;
      if( maxx >= width ) maxx = width;
      if( maxy >= height ) maxy = height;

      //constant part of half-edge functions
      
      C1 = Dy12*x1 - Dx12*y1; //derived from line equation (X1-X2)*(y-Y1) - (Y1-Y2)*(x-X1) = 0
      C2 = Dy23*x2 - Dx23*y2; //derived from line equation (X2-X3)*(y-Y2) - (Y2-Y3)*(x-X2) = 0
      C3 = Dy31*x3 - Dx31*y3; //derived from line equation (X3-X1)*(y-Y3) - (Y3-Y1)*(x-X3) = 0

      //Correct for fill convention
      if( Dy12 < 0 || ( Dy12 == 0 && Dx12 > 0 ) )
          C1 += 1;
      if( Dy23 < 0 || ( Dy23 == 0 && Dx23 > 0 ) )
          C2 += 1;
      if( Dy31 < 0 || ( Dy31 == 0 && Dx31 > 0 ) )
          C3 += 1;

      Cy1 = C1 + Dx12 * miny - Dy12 * minx;
      Cy2 = C2 + Dx23 * miny - Dy23 * minx;
      Cy3 = C3 + Dx31 * miny - Dy31 * minx;

      for( int y = miny; y < maxy; ++y )
      {
          Cx1 = Cy1;
          Cx2 = Cy2;
          Cx3 = Cy3;
          for( int x = minx; x < maxx; ++x )
          {
              if( Cx1 > 0 && Cx2 > 0 && Cx3 > 0 )
              {
                  //interpolate attributes using barycentic interpolation
               //   if( pri.p[0].y - pri.p[1].y != 0 && pri.p[0].y - pri.p[2].y != 0 )
               //   {
               //       base1 = 0; base2 = 1; base3 = 2;
               //   }
               //   else if( pri.p[1].y - pri.p[0].y != 0 && pri.p[1].y - pri.p[2].y != 0 )
               //   {
               //       base1 = 1; base2 = 0; base3 = 2;
               //   }
               //   else
               //   {
               //       base1 = 2; base2 = 0; base3 =1;
               //   }

               //   //interpolate Z value
               //   x_alpha = ( pri.p[base1].x - pri.p[base2].x ) / ( pri.p[base1].y - pri.p[base2].y );
               //   x_l = ( y - pri.p[base2].y ) * x_alpha + pri.p[base2].x;

               //   x_alpha = ( pri.p[base1].x - pri.p[base3].x ) / ( pri.p[base1].y - pri.p[base3].y );
               //   x_r = ( y - pri.p[base3].y ) * x_alpha + pri.p[base3].x;

               //   z_alpha = ( pri.p[base1].z - pri.p[base2].z ) / ( pri.p[base1].y - pri.p[base2].y );
               //   z_l = ( y - pri.p[base2].y ) * z_alpha + pri.p[base2].z;

               //   z_alpha = ( pri.p[base1].z - pri.p[base3].z ) / ( pri.p[base1].y - pri.p[base3].y );
               //   z_r = ( y - pri.p[base3].y ) * z_alpha + pri.p[base3].z;

               //   zval = ( ( z_l - z_r ) / (float)( x_l - x_r )) * ( x - x_r ) + z_r; 
               //   //if( zval > depthbuffer[y*width + x]. )
               //   //    continue;

               //   x_alpha = ( y - pri.p[base2].y ) / ( pri.p[base2].y - pri.p[base1].y );
               //   x_l = pri.p[base2].x + x_alpha *( pri.p[base2].x - pri.p[base1].x );
               //   w_l = 1.0f / pri.p[base2].z + x_alpha * ( 1.0f/pri.p[base2].z - 1.0f/pri.p[base1].z );

               //   t_l = pri.t[base2] / pri.p[base2].z + x_alpha * ( pri.t[base2] / pri.p[base2].z- pri.t[base1] / pri.p[base1].z );
               //   color_l = pri.c[base2] / pri.p[base2].z + x_alpha * ( pri.c[base2] / pri.p[base2].z - pri.c[base1] / pri.p[base1].z );

	              //x_alpha = ( y - pri.p[base3].y ) / ( pri.p[base3].y - pri.p[base1].y );
	              //x_r = pri.p[base3].x + x_alpha * ( pri.p[base3].x - pri.p[base1].x );
	              //w_r = 1.0f/pri.p[base3].z + x_alpha * ( 1.0f/pri.p[base3].z - 1.0f/pri.p[base1].z );

	              //t_r = pri.t[base3] / pri.p[base3].z + x_alpha * ( pri.t[base3] / pri.p[base3].z - pri.t[base1] / pri.p[base1].z );
	              //color_r = pri.c[base3] / pri.p[base3].z + x_alpha * ( pri.c[base3] / pri.p[base3].z - pri.c[base1] /pri.p[base1].z );

               //   x_alpha = ( x-x_r )/(x_l-x_r);
                  //unsigned int assume = __float_as_int(depthbuffer[x+(width*y)].color.x);
                  //atomicCAS( (int*)&depthbuffer[x+(width*y)].color.x, assume, 1 );
                  depthbuffer[x+(width*y)].color.x = 1.0f;
                  depthbuffer[x+(width*y)].color.y = 1.0f;
                  depthbuffer[x+(width*y)].color.z = 1.0f;
              } 
              Cx1 -= Dy12;
              Cx2 -= Dy23;
              Cx3 -= Dy31;
          }
          Cy1 += Dx12;
          Cy2 += Dx23;
          Cy3 += Dx31;
      }
  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, ushort width, ushort height){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * width);
  if(x<width && y<height){
     
  }
}

//Writes fragment colors to the framebuffer
__global__ void render(ushort width, ushort height, fragment* depthbuffer, glm::vec3* framebuffer){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * width);

  if(x<width && y<height){
    framebuffer[index] = depthbuffer[index].color;
  }
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, ushort width, ushort height, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize,
                       VertUniform &vsUniform, FragUniform &fsUniform)
{

  // set up crucial magic
  int tileSize = 8;
  dim3 blockSize( tileSize, tileSize );
  dim3 gridSize(( width + blockSize.x -1)/blockSize.x, (height+blockSize.y-1)/blockSize.y );  

  //set up framebuffer
  //framebuffer = NULL;
  //cudaErrorCheck( cudaMalloc((void**)&framebuffer, width*height*sizeof(glm::vec3)) );
  //
  ////set up depthbuffer
  //depthbuffer = NULL;
  //cudaErrorCheck( cudaMalloc((void**)&depthbuffer, width*height*sizeof(fragment)) );

  //kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
  clearImage<<<gridSize, blockSize>>>(width,height, framebuffer, glm::vec3(0,0,0));
  
  fragment frag;
  frag.color = glm::vec3(0,0,0);
  frag.normal = glm::vec3(0,0,0);
  frag.position = glm::vec3(0,0,-10000);
  clearDepthBuffer<<<gridSize, blockSize>>>(width,height, depthbuffer,frag);

  //------------------------------
  //memory stuff
  //------------------------------
  //primitives = NULL;
  //cudaErrorCheck(  cudaMalloc((void**)&primitives, (ibosize/3)*sizeof(triangle)) );

  //device_ibo = NULL;
  //cudaErrorCheck(  cudaMalloc((void**)&device_ibo, ibosize*sizeof(int)) );
  //cudaErrorCheck(  cudaMemcpy( device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice) );

  //device_vbo = NULL;
  //cudaErrorCheck( cudaMalloc((void**)&device_vbo, vbosize*sizeof(float)) );
  //cudaErrorCheck( cudaMemcpy( device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice) );

  //device_cbo = NULL;
  //cudaErrorCheck( cudaMalloc((void**)&device_cbo, cbosize*sizeof(float)) );
  //cudaErrorCheck( cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice) );
 
  tileSize = 32;
  int primitiveBlocks = (vbosize/3 + tileSize-1)/tileSize;
  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, device_vbo_t, width, height, vbosize, vsUniform );

  cudaErrorCheck(  cudaDeviceSynchronize() );
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = (ibosize/3 + tileSize-1 )/tileSize;
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo_t, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives);

  cudaErrorCheck( cudaDeviceSynchronize() );
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, width, height);

  cudaErrorCheck( cudaDeviceSynchronize() );
  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<gridSize, blockSize>>>(depthbuffer, width, height);

  cudaErrorCheck( cudaDeviceSynchronize() );
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  render<<<gridSize, blockSize>>>(width, height, depthbuffer, framebuffer);
  sendImageToPBO<<<gridSize, blockSize>>>(PBOpos, width, height, framebuffer);

  cudaErrorCheck( cudaDeviceSynchronize() );

  //kernelCleanup();

  checkCUDAError("Kernel failed!");
}

void initDeviceBuf( ushort width, ushort height, float* vbo, int vbosize, float* cbo, int cbosize, 
                 int* ibo, int ibosize )
{
    kernelCleanup();

    //create framebuffer
    framebuffer = 0;
    cudaErrorCheck( cudaMalloc((void**)&framebuffer, width*height*sizeof(glm::vec3)) );

    //create depth buffer
    depthbuffer = NULL;
    cudaErrorCheck( cudaMalloc((void**)&depthbuffer, width*height*sizeof(fragment)) );

    //------------------------------
    //memory stuff
    //------------------------------
    primitives = NULL;
    cudaErrorCheck(  cudaMalloc((void**)&primitives, (ibosize/3)*sizeof(triangle)) );

    device_ibo = NULL;
    cudaErrorCheck(  cudaMalloc((void**)&device_ibo, ibosize*sizeof(int)) );
    cudaErrorCheck(  cudaMemcpy( device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice) );

    device_vbo = NULL;
    cudaErrorCheck( cudaMalloc((void**)&device_vbo, vbosize*sizeof(float)) );
    cudaErrorCheck( cudaMemcpy( device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice) );

    device_vbo_t = NULL;
    cudaErrorCheck( cudaMalloc((void**)&device_vbo_t, vbosize*sizeof(float)) );

    device_cbo = NULL;
    cudaErrorCheck( cudaMalloc((void**)&device_cbo, cbosize*sizeof(float)) );
    cudaErrorCheck( cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice) );
}

void kernelCleanup()
{
  if( primitives )
  {
      cudaFree( primitives );
      primitives = 0;
  }
  if( device_vbo )
  {
      cudaFree( device_vbo );
      device_vbo = 0;
  }
  if( device_vbo_t )
  {
      cudaFree( device_vbo_t );
      device_vbo_t = 0;
  }
  if( device_cbo )
  {
      cudaFree( device_cbo );
      device_cbo = 0;
  }
  if( device_ibo )
  {
      cudaFree( device_ibo );
      device_ibo = 0;
  }
  if( framebuffer )
  {
      cudaFree( framebuffer );
      framebuffer = 0;
  }
  if( depthbuffer )
  {
      cudaFree( depthbuffer );
      depthbuffer = 0;
  }

}

