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

//#define DEBUG 

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_cbo;
float* device_nbo;
int* device_ibo;
vertex* vertices;
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
__global__ void vertexShadeKernel(glm::mat4 view, float* vbo, int vbosize, float* nbo, int nbosize, vertex* vertices ){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  glm::vec4 point;
  glm::vec4 point_tformd;
  glm::vec4 normal;
  glm::vec4 normal_tformd;
  if(index<vbosize/3){
    // Assemble vec4 from vbo ... vertex assembly :)
    point.x = vbo[3*index];
    point.y = vbo[3*index+1];
    point.z = vbo[3*index+2]; 
    point.w = 1.0f;
    
    normal.x = nbo[3*index];
    normal.y = nbo[3*index+1];
    normal.z = nbo[3*index+2];
    normal.w = 0.0f;
     

    // Apply model-view-project matrix transform
    // ... at the moment just the identity 
    glm::mat4 eye = glm::mat4( 1.0f );
    /*
    glm::mat4 trans = glm::translate( eye, glm::vec3(0.5, 0.0, 0.0 ));
    glm::mat4 model = glm::rotate( trans, 30.0f, glm::vec3( 1.0, 0.0, 0.0 ) );
    */
    /*
    glm::mat4 projection = glm::perspective(60.0f, 1.0f, 0.1f, 100.0f);
    glm::quat rot;
    rot = quat::angleAxis( 45.0f, glm::vec3( 1.0, 0.0, 0.0 ) );
    

    glm::mat4 model = quaternion::toMat4( rot );
    */
    
    glm::mat4 projection = glm::perspective(60.0f, 1.0f, 0.1f, 100.0f);

    // Transform vertex and normal
    point_tformd = projection*view*point;
    normal_tformd = projection*view*normal;
    //vertex_tformd = view*vertex;
    //vertex_tformd = (eye*projection)*vertex;

    // Convert to normalized device coordinates
    float div = 1.0f;
    if ( abs(point_tformd.w) > 1e-8 )
      div = 1.0f/point_tformd.w;
    /*
    vbo[3*index] = vertex_tformd.x*div;
    vbo[3*index+1] = vertex_tformd.y*div;
    vbo[3*index+2] = vertex_tformd.z*div; 
    */
    vertices[index].point.x = point_tformd.x*div;
    vertices[index].point.y=  point_tformd.y*div;
    vertices[index].point.z = point_tformd.z*div; 
    vertices[index].normal.x = normal_tformd.x;
    vertices[index].normal.y = normal_tformd.y;
    vertices[index].normal.z = normal_tformd.z;

  }
}

//TODO: Implement primative assembly
/* Primative assembly takes vertices from the vbo and triangel indices from the ibo
   and assembles triangle primatives for the rasterizer to work with 
*/
//__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives){
__global__ void primitiveAssemblyKernel(vertex* vertices, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives){
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
    /*
    primitives[index].p0 = glm::vec3( vbo[3*i0], vbo[3*i0+1], vbo[3*i0+2] );
    primitives[index].p1 = glm::vec3( vbo[3*i1], vbo[3*i1+1], vbo[3*i1+2] );
    primitives[index].p2 = glm::vec3( vbo[3*i2], vbo[3*i2+1], vbo[3*i2+2] );
    */
    primitives[index].p0 = vertices[i0].point;
    primitives[index].p1 = vertices[i1].point;
    primitives[index].p2 = vertices[i2].point;
    
    // Copy over normals
    primitives[index].n0 = vertices[i0].normal;
    primitives[index].n1 = vertices[i1].normal;
    primitives[index].n2 = vertices[i2].normal;

    // Copy over vertex colors
    /*
    primitives[index].c0 = glm::vec3( cbo[3*i0], cbo[3*i0+1], cbo[3*i0+2] );
    primitives[index].c1 = glm::vec3( cbo[3*i1], cbo[3*i1+1], cbo[3*i1+2] );
    primitives[index].c2 = glm::vec3( cbo[3*i2], cbo[3*i2+1], cbo[3*i2+2] );
    */
    primitives[index].c0 = glm::vec3( cbo[0], cbo[1], cbo[2] );
    primitives[index].c1 = glm::vec3( cbo[3], cbo[4], cbo[5] );
    primitives[index].c2 = glm::vec3( cbo[6], cbo[7], cbo[8] );
            
  }
}

//TODO: Implement a rasterization method, such as scanline.
/* 
   Given triangle coordinates, converted to screen coordinates, find fragments inside of triangle using AABB and brute force barycentric coords checks
*/
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer,  glm::vec2 resolution) {

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  glm::vec2 p0;
  glm::vec2 p1;
  glm::vec2 p2;
  glm::vec3 min_point;
  glm::vec3 max_point;

  triangle tri;
  glm::vec3 bary_coord;

  fragment frag; 
  float frag_depth;

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

    // Normal does not point towards the camera
    if ( calculateSignedArea( tri ) <= 0.0f )
      return;
  
    getAABBForTriangle( tri, min_point, max_point );

    // Ensure window bounds are maintained
    min_point.x = max( min_point.x, 0.0f );
    min_point.y = max( min_point.y, 0.0f );
    max_point.x = min( max_point.x, resolution.x );
    max_point.y = min( max_point.y, resolution.y );
    
    // For each pixel in the bounding box check if its in the triangle 
    for ( int x=glm::floor(min_point.x); x<glm::ceil(max_point.x); ++x ) {
      for ( int y=glm::floor(min_point.y); y<glm::ceil(max_point.y); ++y ) {
	//int frag_index = x + (y * resolution.x);
	bary_coord = calculateBarycentricCoordinate( tri, glm::vec2( x,y ) );
	if ( isBarycentricCoordInBounds( bary_coord ) ) {
	  frag_depth = getZAtCoordinate( bary_coord, primitives[index] );
	  
	  // If frag_depth is less than the current depth in the depth buffer then update

	  // Color a fragment just for debugging sake 
	  //frag.color = glm::vec3( 1.0, 0.0, 0.0 );  
	  frag.position = glm::vec3( x, y, frag_depth );
	  frag.normal = bary_coord[0]*primitives[index].n0 \
		      + bary_coord[1]*primitives[index].n1 \
		      + bary_coord[2]*primitives[index].n2;

	  // Solid color for every triangle

	  // Interpolate color on triangle ... cause this will look pretty
	  frag.color = bary_coord[0]*primitives[index].c0 \
		     + bary_coord[1]*primitives[index].c1 \
		     + bary_coord[2]*primitives[index].c2;
	  
	  // Color fragment based interpolated normal
	  //frag.color = frag.normal;
	  
	  // This is BAD and not atomic :/
	  fragment cur_frag = getFromDepthbuffer( x, y, depthbuffer, resolution );
	  // If current value is gt than new value then update
	  if ( frag_depth > cur_frag.position.z ) 
	    writeToDepthbuffer( x, y, frag, depthbuffer, resolution );
	}
      }
    }
  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, int draw_mode ){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  fragment frag;
  if(x<=resolution.x && y<=resolution.y){
    frag = getFromDepthbuffer( x, y, depthbuffer, resolution );

    // Interactive drawing modes
    switch (draw_mode) {
      case( DRAW_SOLID ):
	frag.color = glm::vec3( 1.0, 1.0, 1.0 );
	break;
      case( DRAW_COLOR ):
	// Keep color the same
	break;
      case( DRAW_NORMAL ):
	frag.color = frag.normal;
	break;
      case( SHADE_SOLID ):
	// phong shading solid color
	break;
      case( SHADE_COLOR ):
	// phong shading interpolated color
	break;
    }
    writeToDepthbuffer( x, y, frag, depthbuffer, resolution ); 
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
void cudaRasterizeCore(glm::mat4 view, uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* nbo, int nbosize, float* cbo, int cbosize, int* ibo, int ibosize){

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

  vertices = NULL;
  cudaMalloc((void**)&vertices, (ibosize)*sizeof(vertex));

  host_primitives = NULL;
  host_primitives = (triangle*)malloc((ibosize/3)*sizeof(triangle));


  device_ibo = NULL;
  cudaMalloc((void**)&device_ibo, ibosize*sizeof(int));
  cudaMemcpy( device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice);

  device_vbo = NULL;
  cudaMalloc((void**)&device_vbo, vbosize*sizeof(float));
  cudaMemcpy( device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_nbo = NULL;
  cudaMalloc((void**)&device_nbo, nbosize*sizeof(float));
  cudaMemcpy( device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);



  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------
  // DEBUG
#ifdef DEBUG
  printf( "resolution: [%f, %f] \n", resolution.x, resolution.y );
  printf( "vbosize: %d \n", vbosize );

  glm::mat4 model_view = glm::mat4( 1.0f );
  glm::mat4 projection = glm::perspective(35.0f, 1.0f, 0.1f, 100.0f);
  projection = model_view*projection;
  printf( "projection: [%f, %f, %f, %f, \n %f, %f, %f, %f, \n %f, %f, %f, %f, \n %f, %f, %f, %f] \n", \
		       projection[0][0], projection[0][1], projection[0][2], projection[0][3], \
		       projection[1][0], projection[1][1], projection[1][2], projection[1][3], \
		       projection[2][0], projection[2][1], projection[2][2], projection[2][3], \
		       projection[3][0], projection[3][1], projection[3][2], projection[3][3] );

  // DEBUG
  printf(" vbo -------- \n ");
  for ( int i=0; i < vbosize/3; i++ ) {
    printf("[%f, %f, %f] \n", vbo[3*i], vbo[3*i+1], vbo[3*i+2]);
  }
#endif

  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(view, device_vbo, vbosize, device_nbo, nbosize, vertices);

  // DEBUG
#ifdef DEBUG 
  /*
  cudaMemcpy( vbo, device_vbo, vbosize*sizeof(float), cudaMemcpyDeviceToHost );
  printf(" vbo_tf -------- \n ");
  for ( int i=0; i < vbosize/3; i++ ) {
    printf("[%f, %f, %f] \n", vbo[3*i], vbo[3*i+1], vbo[3*i+2]);
  }
  */
#endif


  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  // DEBUG
#ifdef DEBUG
  printf(" ibo -------- \n ");
  for ( int i=0; i < ibosize/3; i++ ) {
    printf("[%d, %d, %d] \n", ibo[3*i], ibo[3*i+1], ibo[3*i+2]);
  }
#endif

  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  //primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives);
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(vertices, device_cbo, cbosize, device_ibo, ibosize, primitives);

  // Host copy of primitives so that I can't print this out and make sure it works
  cudaMemcpy( host_primitives, primitives, (ibosize/3)*sizeof(triangle), cudaMemcpyDeviceToHost );

  // DEBUG
#ifdef DEBUG
  printf(" primitives ---------- \n " );
  for ( int i=0; i < ibosize/3; i++ ) {
    printf("p0: [%f, %f, %f] \n", host_primitives[i].p0.x, host_primitives[i].p0.y, host_primitives[i].p0.z );
    printf("p1: [%f, %f, %f] \n", host_primitives[i].p1.x, host_primitives[i].p1.y, host_primitives[i].p1.z );
    printf("p2: [%f, %f, %f] \n", host_primitives[i].p2.x, host_primitives[i].p2.y, host_primitives[i].p2.z );
  }
#endif

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);

#ifdef DEBUG
  // Host copy of primitives so that I can't print this out and make sure it works
  cudaMemcpy( host_primitives, primitives, (ibosize/3)*sizeof(triangle), cudaMemcpyDeviceToHost );

  // DEBUG
  printf(" primitives ---------- \n " );
  for ( int i=0; i < ibosize/3; i++ ) {
    printf("p0: [%f, %f, %f] \n", host_primitives[i].p0.x, host_primitives[i].p0.y, host_primitives[i].p0.z );
    printf("p1: [%f, %f, %f] \n", host_primitives[i].p1.x, host_primitives[i].p1.y, host_primitives[i].p1.z );
    printf("p2: [%f, %f, %f] \n", host_primitives[i].p2.x, host_primitives[i].p2.y, host_primitives[i].p2.z );
  }
#endif


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
  cudaFree( vertices );
  cudaFree( device_vbo );
  cudaFree( device_nbo );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}

