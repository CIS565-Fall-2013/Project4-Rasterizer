// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_cbo;
int* device_ibo;
triangle* primitives;
triangle* host_primitives;

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
__host__ __device__ void writeToDepthbuffer(int x, int y, fragment frag, fragment* depthbuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    depthbuffer[index] = frag;
  }
}

//Reads a fragment from a given location in a fragment buffer
__host__ __device__ fragment getFromDepthbuffer(int x, int y, fragment* depthbuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    return depthbuffer[index];
  }else{
    fragment f;
    return f;
  }
}

//Writes a given pixel to a pixel buffer at a given location
__host__ __device__ void writeToFramebuffer(int x, int y, glm::vec3 value, glm::vec3* framebuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    framebuffer[index] = value;
  }
}

//Reads a pixel from a pixel buffer at a given location
__host__ __device__ glm::vec3 getFromFramebuffer(int x, int y, glm::vec3* framebuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    return framebuffer[index];
  }else{
    return glm::vec3(0,0,0);
  }
}

//Kernel that clears a given pixel buffer with a given color
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image, glm::vec3 color){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = color;
    }
}

//Kernel that clears a given fragment buffer with a given fragment
__global__ void clearDepthBuffer(glm::vec2 resolution, fragment* buffer, fragment frag){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      fragment f = frag;
      f.position.x = x;
      f.position.y = y;
      buffer[index] = f;
    }
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

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
/* The vertex shader takes in vertices and applies the transformations
   that map vertex coordinates to camera coordinates:
     Pclip = (Mmodel-view-projection)(Pmodel)
*/
__global__ void vertexShadeKernel(float* vbo, int vbosize){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  glm::vec4 vertex;
  glm::vec4 vertex_tformd;
  if(index<vbosize/3){
    // Assemble vec4 from vbo ... vertex assembly :)
    vertex.x = vbo[3*index];
    vertex.y = vbo[3*index+1];
    vertex.z = vbo[3*index+2]; 
    vertex.w = 1.0f;

    // Apply model-view-project matrix transform
    // ... at the moment just the identity 
    glm::mat4 model_view_project = glm::mat4( 1.0f );
    // Transform
    vertex_tformd = model_view_project*vertex;

    // Copy back to vbo and apply perspective division ... not sure if this should be here
    vbo[3*index] = vertex_tformd.x/vertex_tformd.w;
    vbo[3*index+1] = vertex_tformd.y/vertex_tformd.w;
    vbo[3*index+2] = vertex_tformd.z/vertex_tformd.w; 
  }
}

//TODO: Implement primative assembly
/* Primative assembly takes vertices from the vbo and triangel indices from the ibo
   and assembles triangle primatives for the rasterizer to work with 
*/
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;

  int i0;
  int i1;
  int i2;
  if(index<primitivesCount){
    // Pull out indices
    i0 = ibo[3*index];
    i1 = ibo[3*index+1];
    i2 = ibo[3*index+2];

    // Copy over vertex points
    primitives[index].p0 = glm::vec3( vbo[3*i0], vbo[3*i0+1], vbo[3*i0+2] );
    primitives[index].p1 = glm::vec3( vbo[3*i1], vbo[3*i1+1], vbo[3*i1+2] );
    primitives[index].p2 = glm::vec3( vbo[3*i2], vbo[3*i2+1], vbo[3*i2+2] );

    // Copy over vertex colors
    primitives[index].c0 = glm::vec3( cbo[3*i0], cbo[3*i0+1], cbo[3*i0+2] );
    primitives[index].c1 = glm::vec3( cbo[3*i1], cbo[3*i1+1], cbo[3*i1+2] );
    primitives[index].c2 = glm::vec3( cbo[3*i2], cbo[3*i2+1], cbo[3*i2+2] );
            
  }
}

//TODO: Implement a rasterization method, such as scanline.
/* 
   Given triangle coordinates, converted to screen coordinates, find fragments inside of triangle
*/
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution) {

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  glm::vec2 p0;
  glm::vec2 p1;
  glm::vec2 p2;
  glm::vec3 min_point;
  glm::vec3 max_point;

  triangle tri;
  glm::vec3 bary_coord;

  float scale_x;
  float scale_y;
  float offs_x;
  float offs_y;
  if ( index<primitivesCount ) {
    // Map primitives from world to window coordinates using the viewport transform
    scale_x = resolution.x/2;
    scale_y = resolution.y/2;
    offs_x = resolution.x/2;
    offs_y = resolution.y/2;
   
    tri.p0.x = scale_x*primitives[index].p0.x + offs_x;
    tri.p1.x = scale_x*primitives[index].p1.x + offs_x;
    tri.p2.x = scale_x*primitives[index].p2.x + offs_x;

    tri.p0.y = offs_y - scale_y*primitives[index].p0.y;
    tri.p1.y = offs_y - scale_y*primitives[index].p1.y;
    tri.p2.y = offs_y - scale_y*primitives[index].p2.y;
  
    /* DEBUG
    primitives[index].p0.x = scale_x*primitives[index].p0.x + offs_x;
    primitives[index].p1.x = scale_x*primitives[index].p1.x + offs_x;
    primitives[index].p2.x = scale_x*primitives[index].p2.x + offs_x;

    primitives[index].p0.y = offs_y - scale_y*primitives[index].p0.y ;
    primitives[index].p1.y = offs_y - scale_y*primitives[index].p1.y ;
    primitives[index].p2.y = offs_y - scale_y*primitives[index].p2.y ;
    */
  
    // Get pixels to look at for a given triangle 
    getAABBForTriangle( primitives[index], min_point, max_point );

    // Convert min and max to window coordinates ... y is flipped so min/max 
    // are flipped for y 
    min_point.x = max( scale_x*min_point.x + offs_x, 0.0f );
    min_point.y = max( offs_y - scale_y*max_point.y, 0.0f );
    max_point.x = min( scale_x*max_point.x + offs_x, resolution.x );
    max_point.y = max( offs_y - scale_y*min_point.y, resolution.y );

    // For each pixel in the bounding box check if its in the triangle 
    for ( int x=glm::floor(min_point.x); x<glm::ceil(max_point.x); ++x ) {
      for ( int y=glm::floor(min_point.y); y<glm::ceil(max_point.x); ++y ) {
	int frag_index = x + (y * resolution.x);
	bary_coord = calculateBarycentricCoordinate( tri, glm::vec2( x,y ) );
	if ( isBarycentricCoordInBounds( bary_coord ) ) {
	  // Color a fragment just for debugging sake 
	  //depthbuffer[frag_index].color = glm::vec3( 1.0, 0.0, 0.0 );  
	  // Interpolate color on triangle ... cause this will look pretty
	  depthbuffer[frag_index].color = bary_coord[0]*primitives[index].c0 \
				        + bary_coord[1]*primitives[index].c1 \
					+ bary_coord[2]*primitives[index].c2;

	}
      }
    }
  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
  }
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y){
    framebuffer[index] = depthbuffer[index].color;
  }
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize){

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));

  //set up framebuffer
  framebuffer = NULL;
  cudaMalloc((void**)&framebuffer, (int)resolution.x*(int)resolution.y*sizeof(glm::vec3));
  
  //set up depthbuffer
  depthbuffer = NULL;
  cudaMalloc((void**)&depthbuffer, (int)resolution.x*(int)resolution.y*sizeof(fragment));

  //kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0));
  
  fragment frag;
  frag.color = glm::vec3(0,0,0);
  frag.normal = glm::vec3(0,0,0);
  frag.position = glm::vec3(0,0,-10000);
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag);

  //------------------------------
  //memory stuff
  //------------------------------
  primitives = NULL;
  cudaMalloc((void**)&primitives, (ibosize/3)*sizeof(triangle));

  host_primitives = NULL;
  host_primitives = (triangle*)malloc((ibosize/3)*sizeof(triangle));


  device_ibo = NULL;
  cudaMalloc((void**)&device_ibo, ibosize*sizeof(int));
  cudaMemcpy( device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice);

  device_vbo = NULL;
  cudaMalloc((void**)&device_vbo, vbosize*sizeof(float));
  cudaMemcpy( device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------
  // DEBUG
  printf( "resolution: [%f, %f] \n", resolution.x, resolution.y );
  printf( "vbosize: %d", vbosize );

  // DEBUG
  printf(" vbo -------- \n ");
  for ( int i=0; i < vbosize/3; i++ ) {
    printf("[%f, %f, %f] \n", vbo[3*i], vbo[3*i+1], vbo[3*i+2]);
  }

  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize);

  // DEBUG
  cudaMemcpy( vbo, device_vbo, vbosize*sizeof(float), cudaMemcpyDeviceToHost );
  printf(" vbo_tf -------- \n ");
  for ( int i=0; i < vbosize/3; i++ ) {
    printf("[%f, %f, %f] \n", vbo[3*i], vbo[3*i+1], vbo[3*i+2]);
  }


  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  // DEBUG
  printf(" ibo -------- \n ");
  for ( int i=0; i < ibosize/3; i++ ) {
    printf("[%d, %d, %d] \n", ibo[3*i], ibo[3*i+1], ibo[3*i+2]);
  }

  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives);

  // Host copy of primitives so that I can't print this out and make sure it works
  cudaMemcpy( host_primitives, primitives, (ibosize/3)*sizeof(triangle), cudaMemcpyDeviceToHost );

  // DEBUG
  printf(" primitives ---------- \n " );
  for ( int i=0; i < ibosize/3; i++ ) {
    printf("p0: [%f, %f, %f] \n", host_primitives[i].p0.x, host_primitives[i].p0.y, host_primitives[i].p0.z );
    printf("p1: [%f, %f, %f] \n", host_primitives[i].p1.x, host_primitives[i].p1.y, host_primitives[i].p1.z );
    printf("p2: [%f, %f, %f] \n", host_primitives[i].p2.x, host_primitives[i].p2.y, host_primitives[i].p2.z );
  }

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);

  // Host copy of primitives so that I can't print this out and make sure it works
  cudaMemcpy( host_primitives, primitives, (ibosize/3)*sizeof(triangle), cudaMemcpyDeviceToHost );

  // DEBUG
  printf(" primitives ---------- \n " );
  for ( int i=0; i < ibosize/3; i++ ) {
    printf("p0: [%f, %f, %f] \n", host_primitives[i].p0.x, host_primitives[i].p0.y, host_primitives[i].p0.z );
    printf("p1: [%f, %f, %f] \n", host_primitives[i].p1.x, host_primitives[i].p1.y, host_primitives[i].p1.z );
    printf("p2: [%f, %f, %f] \n", host_primitives[i].p2.x, host_primitives[i].p2.y, host_primitives[i].p2.z );
  }


  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution);

  cudaDeviceSynchronize();
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

  cudaDeviceSynchronize();

  kernelCleanup();

  checkCUDAError("Kernel failed!");
}

void kernelCleanup(){
  cudaFree( primitives );
  cudaFree( device_vbo );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}

