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
float* device_nbo;
float* device_cbo;
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

//TODO: Implement a vertex shader
__global__ void vertexShadeKernel(float* vbo, int vbosize,  float* nbo, int nbosize, cudaMat4 mvpMatrix, cudaMat4 mvpInverseTransMatrix,glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  glm::vec4 v(vbo[3*index], vbo[3*index + 1], vbo[3*index+2],1.0f);

	  glm::vec4 transformedVec = multiplyMV(mvpMatrix,v);
	  transformedVec.x = transformedVec.x/transformedVec.w;
	  transformedVec.y = transformedVec.y/transformedVec.w;
	  transformedVec.z = transformedVec.z/transformedVec.w;

	  cudaMat4 viewport;
	  viewport.x = glm::vec4(resolution.x/2.0f,0.0f,0.0f,resolution.x/2.0f);
	  viewport.y = glm::vec4(0.0f,resolution.y/2.0f,0.0f,resolution.y/2.0f);
	  viewport.z = glm::vec4(0.0f,0.0f,0.5f,0.5f);
	  viewport.w = glm::vec4(0.0f,0.0f,0.0f,1.0f);
	  transformedVec.w = 1;
	  transformedVec = multiplyMV(viewport,transformedVec);

	  vbo[3*index] = transformedVec.x;
	  vbo[3*index + 1] = transformedVec.y;
	  vbo[3*index + 2] = transformedVec.z;

	  glm::vec4 n( nbo[3*index], nbo[3*index + 1], nbo[3*index+2],0.0f);
	  glm::vec4 transformedNormal = multiplyMV(mvpInverseTransMatrix,n);
	  nbo[3*index]   = transformedNormal.x;
	  nbo[3*index+1] = transformedNormal.y;
	  nbo[3*index+2] = transformedNormal.z;
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  int firstVertexIdx = ibo[3*index];
	  int secondVertexIdx = ibo[3*index+1];
	  int thirdVertexIdx = ibo[3*index+2];
	  primitives[index].p0 = glm::vec3(vbo[3*firstVertexIdx], vbo[3*firstVertexIdx + 1], vbo[3*firstVertexIdx+2]);
	  primitives[index].p1 = glm::vec3(vbo[3*secondVertexIdx], vbo[3*secondVertexIdx + 1], vbo[3*secondVertexIdx+2]);
	  primitives[index].p2 = glm::vec3(vbo[3*thirdVertexIdx], vbo[3*thirdVertexIdx + 1], vbo[3*thirdVertexIdx+2]);
	  primitives[index].c0 = glm::vec3(cbo[0],cbo[1],cbo[2]);
	  primitives[index].c1 = glm::vec3(cbo[4],cbo[5],cbo[6]);
	  primitives[index].c2 = glm::vec3(cbo[7],cbo[8],cbo[9]);
  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
  triangle t = primitives[index];
  glm::vec3 aabbMin;
  glm::vec3 aabbMax;

  getAABBForTriangle(t,aabbMin,aabbMax);
  int minX = int(floor(aabbMin.x));
  int maxX = int(ceil(aabbMax.x));
  int minY = int(floor(aabbMin.y));
  int maxY = int(ceil(aabbMax.y));


  for(int i=minX; i<maxX ; ++i)
	  for(int j=minY; j<maxY; ++j)
	  {
		  if ( i>=resolution.x || j >= resolution.y)
			  return;
		  //Referred this for point in triangle test
		  //http://www.gamedev.net/topic/295943-is-this-a-better-point-in-triangle-test-2d/

		  //triangle t1;
		  //t1.p0.x = i;
		  //t1.p0.y = j;
		  //t1.p1.x = t.p0.x;
		  //t1.p1.y = t.p0.y;
		  //t1.p2.x = t.p1.x;
		  //t1.p2.y = t.p1.y;
		
		  ////if (calculateSignedArea(t1)<0.0f)
			 //// return;
		  //areaSum+=calculateSignedArea(t1);
		  //t1.p1.x = t.p1.x;
		  //t1.p1.y = t.p1.y;
		  //t1.p2.x = t.p2.x;
		  //t1.p2.y = t.p2.y;
		
		  ////if (calculateSignedArea(t1)<0.0f)
			 //// return;
		  //areaSum+=calculateSignedArea(t1);
		  //t1.p1.x = t.p2.x;
		  //t1.p1.y = t.p2.y;
		  //t1.p2.x = t.p0.x;
		  //t1.p2.y = t.p0.y;
		
		  ////if (calculateSignedArea(t1)<0.0f)
			 //// return;

		  glm::vec3 bCoords =  calculateBarycentricCoordinate(t,glm::vec2(i,j));

		  //if (bCoords.x+bCoords.y+bCoords.z>1.00001f)
			 // return;
		  //if (bCoords.x <0.0f-0.00001f|| bCoords.y<0.0f-0.00001f || bCoords.z<0.0f-0.00001f)
			 // return;
		  fragment f;
		  //f.color = bCoords;
		  f.color = glm::vec3(1.0f,0.0f,0.0f);
		  f.position = glm::vec3(i,j,1.0f);
		  f.normal = glm::vec3(0.0f,0.0f,1.0f);

		  int pixelIndex = i + (j * resolution.x);
		  if ( f.position.z < depthbuffer[pixelIndex].position.z)
		  {
			  depthbuffer[pixelIndex] = f;
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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize,const glm::mat4& modelMatrix,const glm::mat4& viewMatrix,const glm::mat4& projectionMatrix){

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
  frag.position = glm::vec3(0,0,10000);
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

  cudaMat4 mvpMatrix = utilityCore::glmMat4ToCudaMat4(projectionMatrix*viewMatrix*modelMatrix);
  glm::mat4 h_mvpInvTranMatrix = glm::transpose(glm::inverse(viewMatrix*modelMatrix));
  cudaMat4 mvpInvTranMatrix = utilityCore::glmMat4ToCudaMat4(h_mvpInvTranMatrix);
  
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize,device_nbo, nbosize, mvpMatrix, mvpInvTranMatrix,resolution);
  
  ///*TEST VERTEX SHADER*/
  //float* h_vertices = new float[vbosize];
  //cudaMemcpy(h_vertices,device_vbo,vbosize*sizeof(float),cudaMemcpyDeviceToHost);
  //for(int i=0; i<vbosize/3;i++)
  //{
	 // std::cout << vbo[3*i] <<","<<vbo[3*i+1]<<","<<vbo[3*i+2]<<std::endl;
	 // std::cout << h_vertices[3*i] <<","<<h_vertices[3*i+1]<<","<<h_vertices[3*i+2]<<std::endl;
  //}

  //std::cout<<std::endl;
	 // float* h_normals = new float[nbosize];
	 //   cudaMemcpy(h_normals,device_nbo,nbosize*sizeof(float),cudaMemcpyDeviceToHost);

  //for(int i=0; i<nbosize/3;i++)
  //{
	 // std::cout << nbo[3*i] <<","<<nbo[3*i+1]<<","<<nbo[3*i+2]<<std::endl;
	 // std::cout << h_normals[3*i] <<","<<h_normals[3*i+1]<<","<<h_normals[3*i+2]<<std::endl;
  //}
  //std::cout<<std::endl<<std::endl;

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives);

  /*TEST PRIMITIVE ASSEMBLY*/
  //triangle* h_triangles = new triangle[ibosize/3];
  //cudaMemcpy(h_triangles,primitives,ibosize/3*sizeof(triangle),cudaMemcpyDeviceToHost);
  //h_triangles[0];

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

