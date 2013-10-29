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
float* device_nbo;
int* device_ibo;
triangle* primitives;


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
__host__ __device__ void printvec3(glm::vec3 vec)
{
	printf("b %f,%f,%f \n",vec.x,vec.y,vec.z);
}
__host__ __device__ void printfloat3(float x, float y, float z)
{
	printf("point %f,%f,%f \n",x,y,z);
}

//TODO: Implement a vertex shader
__global__ void vertexShadeKernel(float* vbo, int vbosize
	,float* nbo // transformation
	,glm::mat4 modelM, glm::mat4 viewM, glm::mat4 projectionM
	)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  glm::mat4 mvp = projectionM * viewM * modelM; // 
	  glm::vec4 vertex(vbo[index*3],vbo[index*3+1],vbo[index*3+2],1.0);
	  glm::vec4 newVertex = mvp * vertex; //vertex in clip coordinate

	  vbo[index*3] = newVertex.x; 
	  vbo[index*3+1] = newVertex.y; 
	  vbo[index*3+2] = newVertex.z;



	  glm::vec4 newNormal = modelM * glm::vec4(nbo[index*3],nbo[index*3+1],nbo[index*3+2],1.0);
	  glm::vec3 normal = glm::normalize(glm::vec3(newNormal));	  
	  nbo[index*3] = normal.x; 
	  nbo[index*3+1] = normal.y;
	  nbo[index*3+2] = normal.z;  

  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, 
	float* nbo, int nbosize,
	triangle* primitives)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){	
	  

	  primitives[index].p0 = glm::vec3(vbo[ibo[index*3]*3],vbo[ibo[index*3]*3+1],vbo[ibo[index*3]*3+2]);
	  primitives[index].p1 = glm::vec3(vbo[ibo[index*3+1]*3],vbo[ibo[index*3+1]*3+1],vbo[ibo[index*3+1]*3+2]);
	  primitives[index].p2 = glm::vec3(vbo[ibo[index*3+2]*3],vbo[ibo[index*3+2]*3+1],vbo[ibo[index*3+2]*3+2]);
	  if(cbosize == vbosize)
	  {
			primitives[index].c0 = glm::vec3(cbo[ibo[index*3]*3],cbo[ibo[index*3]*3+1],cbo[ibo[index*3]*3+2]);
			primitives[index].c1 = glm::vec3(cbo[ibo[index*3+1]*3],cbo[ibo[index*3+1]*3+1],cbo[ibo[index*3+1]*3+2]);
			primitives[index].c2 = glm::vec3(cbo[ibo[index*3+2]*3],cbo[ibo[index*3+2]*3+1],cbo[ibo[index*3+2]*3+2]);
	  }
	  else
	  {
			primitives[index].c0 = glm::vec3(cbo[0],cbo[1],cbo[2]);
			primitives[index].c1 = glm::vec3(cbo[3],cbo[4],cbo[5]);
			primitives[index].c2 = glm::vec3(cbo[6],cbo[7],cbo[8]);
	  }
	  
	  

	  //normal
	  primitives[index].n0 = glm::vec3(nbo[ibo[index*3]*3],nbo[ibo[index*3]*3+1],nbo[ibo[index*3]*3+2]);
	  primitives[index].n1 = glm::vec3(nbo[ibo[index*3+1]*3],nbo[ibo[index*3+1]*3+1],nbo[ibo[index*3+1]*3+2]);
	  primitives[index].n2 = glm::vec3(nbo[ibo[index*3+2]*3],nbo[ibo[index*3+2]*3+1],nbo[ibo[index*3+2]*3+2]);


  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){

	  //back face culling
	  
	  //get triangle bounding box
	  glm::vec3 boxMin(0);
	  glm::vec3 boxMax(0);
	  getAABBForTriangle(primitives[index],boxMin,boxMax);
	  /*printvec3(boxMin);
	  printvec3(boxMax);*/

	  //get corresponding pixel for bounding box
	  glm::vec2 pixelMin = convertWorldToPixel(boxMin,resolution);
	  glm::vec2 pixelMax = convertWorldToPixel(boxMax,resolution);


	  pixelMin.x = max((int)pixelMin.x,0);
	  pixelMax.x = min((int)pixelMax.x,(int)resolution.x-1);
	  pixelMax.y = max((int)pixelMax.y,0);
	  pixelMin.y = min((int)pixelMin.y,(int)resolution.y-1);

	  //printf("pixels: %f,%f,%f,%f",pixelMin.x,pixelMin.y,pixelMax.x,pixelMax.y);
	  //loop from ymin to ymax
	
	  fragment tmpFrag;
	  int fragIdx = 0;
	  for(int y = pixelMax.y; y < pixelMin.y; y++)
	  {
		 
		  //loop from xmin to xmax
		  for(int x = pixelMin.x; x < pixelMax.x; x++)
		  {
			  fragIdx = x + y * resolution.x;
			  //get pixel position in Canonical View Volumes
			  glm::vec2 pixelPoint;
			  pixelPoint.x = (2.0 * x / (float)resolution.x) - 1;
			  pixelPoint.y = 1-(2.0 * y / (float)resolution.y);

			  //get barycentricCoordinate
			  glm::vec3 barycCoord = calculateBarycentricCoordinate(primitives[index],pixelPoint);
			  //check if pixel is within the triangle
			  if(!isBarycentricCoordInBounds(barycCoord))
			  {
				  continue;
			  }

			  //get depth value
			  float depth = getZAtCoordinate(barycCoord,primitives[index]);
			 
			  //in normalized device coordinate
			  tmpFrag.position = glm::vec3(pixelPoint.x,pixelPoint.y,depth);
			  //color interpolation
			  tmpFrag.color = barycCoord.x * primitives[index].c0 + barycCoord.y * primitives[index].c1 + barycCoord.z * primitives[index].c2;			  
			 
			  //depth buffer rendering
			 // tmpFrag.color = glm::vec3(1-((abs(depth)-18)/3.5)); 
			  //tmpFrag.color = glm::clamp(tmpFrag.color,glm::vec3(0,0,0),glm::vec3(1,1,1));

			  tmpFrag.normal = (primitives[index].n0 + primitives[index].n1 + primitives[index].n2)/(float)3.0;

			  bool wait = true;
			  while(wait)
			  {
				  if(atomicExch(&(depthbuffer[fragIdx].isLock),1) == 0)
				  {
					  if(tmpFrag.position.z > depthbuffer[fragIdx].position.z)
					  {
						  depthbuffer[fragIdx] = tmpFrag;							 
					  }
					  depthbuffer[fragIdx].isLock = 0;
					  wait = false;
				  }
			  }
		  
		  }//end loop for x
	  }//end loop for y

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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize,
	float* nbo, int nbosize
	,glm::mat4 modelM, glm::mat4 viewM, glm::mat4 projectionM
	,glm::vec3* images
	)
{

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
  frag.isLock = 0;
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag);

  //------------------------------
  //memory stuff
  //------------------------------
  primitives = NULL;
  cudaMalloc((void**)&primitives, (ibosize/3)*sizeof(triangle));

  device_ibo = NULL;
  cudaMalloc((void**)&device_ibo, ibosize*sizeof(int));
  cudaMemcpy( device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice);

  device_vbo = NULL;
  cudaMalloc((void**)&device_vbo, vbosize*sizeof(float));
  cudaMemcpy( device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_nbo = NULL;
  cudaMalloc((void**)&device_nbo, nbosize*sizeof(float));
  cudaMemcpy( device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);


  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize,device_nbo,modelM,viewM,projectionM); 

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize,device_nbo,nbosize,primitives);

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);

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

  //retrieve image from GPU
  cudaMemcpy(images, framebuffer,(int)resolution.x*(int)resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

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

  cudaFree( device_nbo );
}

