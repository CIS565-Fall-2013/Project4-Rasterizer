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
float* device_svbo;
float* device_cbo;
int* device_ibo;
 int* lock;
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
__global__ void clearDepthBuffer(glm::vec2 resolution, fragment* buffer, fragment frag, int* lock){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      fragment f = frag;
      f.position.x = x;
      f.position.y = y;
      buffer[index] = f;
	  lock[index] = 0;
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
__global__ void vertexShadeKernel(float* vbo, int vbosize,glm::mat4 mvpp,glm::mat4 mv, glm::vec2 resolution,float* svbo){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  float a = vbo[3*index];
	  float b = vbo[3*index+1];
	  float c = vbo[3*index+2];
	  glm::vec4 v = (mvpp * glm::vec4(a,b,c,1.0f));
	   //To convert to screen co-ordinates
	  //webglfactory.blogspot.com/2011/05/how-to-convert-world-to-screen.html
	  glm::vec4 vv = (mv * glm::vec4(a,b,c,1.0f));
	  svbo[3*index]   = vv.x;
	  svbo[3*index+1] = vv.y;
	  svbo[3*index+2] = vv.z;

	  vbo[3*index]   = (int)floor((v.x/v.w + 1)*resolution.x/2); // v.x/v.w ;//
	  vbo[3*index+1] =  (int)floor((-v.y/v.w + 1)*resolution.y/2); //v.y/v.w ;//
	  vbo[3*index+2] = v.z/v.w ;
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives,float* svbo,glm::vec3 eye){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  primitives[index].p0 = glm::vec3(vbo[3*(ibo[3*index])],vbo[(3*ibo[3*index])+1],vbo[(3*ibo[3*index])+2]);
	  primitives[index].p1 = glm::vec3(vbo[3*(ibo[3*index+1])],vbo[(3*ibo[3*index+1])+1],vbo[(3*ibo[3*index+1])+2]);
	  primitives[index].p2 = glm::vec3(vbo[3*(ibo[3*index+2])],vbo[(3*ibo[3*index+2])+1],vbo[(3*ibo[3*index+2])+2]);

	  primitives[index].pv0 = glm::vec3(svbo[3*(ibo[3*index])],svbo[(3*ibo[3*index])+1],svbo[(3*ibo[3*index])+2]);
	  primitives[index].pv1 = glm::vec3(svbo[3*(ibo[3*index+1])],svbo[(3*ibo[3*index+1])+1],svbo[(3*ibo[3*index+1])+2]);
	  primitives[index].pv2 = glm::vec3(svbo[3*(ibo[3*index+2])],svbo[(3*ibo[3*index+2])+1],svbo[(3*ibo[3*index+2])+2]);

	  glm::vec3 c = glm::cross((primitives[index].pv1 - primitives[index].pv0),(primitives[index].pv2 - primitives[index].pv0));
	  float a = glm::dot(c,glm::vec3(0,0,-1));
	  if( a > 0)
		  primitives[index].bFace = false;
	  else
		  primitives[index].bFace = true;

	  primitives[index].c0 = glm::vec3(1,0,0);// glm::vec3(cbo[0],cbo[1],cbo[2]); //glm::vec3(1,1,1);//
	  primitives[index].c1 = glm::vec3(0,1,0);// glm::vec3(cbo[3],cbo[4],cbo[5]);
	  primitives[index].c2 = glm::vec3(0,0,1);// glm::vec3(cbo[6],cbo[7],cbo[8]);
  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution,int* lock){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
	  if(primitives[index].bFace == true)
		  return;
	  glm::vec3 minpoint,maxpoint,bCoord;
	  getAABBForTriangle(primitives[index], minpoint,maxpoint);
	  int p = 0 ;
	  bool done = false ; 
	   /*   minpoint.x=(int)floor((minpoint.x+ 1)*resolution.x/2);
		  maxpoint.x=(int)floor((maxpoint.x+ 1)*resolution.x/2);
		  maxpoint.y=(int)floor((-minpoint.y + 1)*resolution.y/2);
		  minpoint.y=(int)floor((-maxpoint.y + 1)*resolution.y/2);*/
	     // Implement culling
	 	 if (minpoint.x  >= resolution.x) return;
		 if (minpoint.y >= resolution.y) return;
		 if (maxpoint.x < 0) return;
		 if (maxpoint.y < 0) return;

		 //Implement clipping 
		 if(minpoint.x < 0)  
			 minpoint.x = 0;
		 
		 if(maxpoint.x > resolution.x-1)
			 maxpoint.x = resolution.x-1;

		 if(minpoint.y < 0)
			 minpoint.y = 0 ;

		 if(maxpoint.y > resolution.y-1)
			 maxpoint.y = resolution.y-1;

		 //minpoint.x = max(minpoint.x, 0);
   //      maxpoint.x = min(maxpoint.x, (int)resolution.x - 1);
   //      minpoint.y = max(minpoint.y, 0);
   //      maxpoint.y = min( maxpoint.y , (int)resolution.y - 1);

	  for(int i = (int)minpoint.x ; i < (int)maxpoint.x ; i++)
	  {
		  for(int j = (int)minpoint.y ; j < (int)maxpoint.y ; j++)
		  {
			//float a = ((i * 2 + 1 - resolution.x)/(resolution.x)) ;
			//float b = ((j * 2 + 1 - resolution.y )/(resolution.y)) ;
			bCoord = calculateBarycentricCoordinate(primitives[index], glm::vec2(i,j));
			
			if(isBarycentricCoordInBounds(bCoord))
			{
				p = (i + j*resolution.x);
				
				done = false ;
				while(!done)
				{
					//supercomputingblog.com/cuda/cuda-tutorial-4-atomic-operations/
					int l= atomicExch(&lock[p], 1 );
					if(l == 0)
					{
						float currentZ = getZAtCoordinate(bCoord, primitives[index]);
						if(currentZ > depthbuffer[p].depth)
						{
						depthbuffer[p].color    = primitives[index].c0 * bCoord.x + primitives[index].c1 * bCoord.y + primitives[index].c2 * bCoord.z;
						depthbuffer[p].position = primitives[index].pv0* bCoord.x + primitives[index].pv1 * bCoord.y + primitives[index].pv2 * bCoord.z;
						depthbuffer[p].normal   = glm::abs(glm::cross((primitives[index].pv1 - primitives[index].pv0),(primitives[index].pv2 - primitives[index].pv0))) * 25.0f;
						depthbuffer[p].depth = currentZ ;
						}
						done = true;
						atomicExch(&lock[p], 0);
					}
				}
			}
		  }
	  }

  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, glm::mat4 MV){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  glm::vec4 lig = glm::vec4(0,5,-10,1) * MV ;
	  depthbuffer[index].color = glm::vec3(1,1,1) * glm::dot(depthbuffer[index].normal,glm::normalize(glm::vec3(lig.x,lig.y,lig.z) - depthbuffer[index].position));
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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize,glm::mat4 MVP,glm::mat4 MV,glm::vec3 eye){

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

  
  lock = NULL;
  cudaMalloc((void**)&lock, (resolution.x * resolution.y) * sizeof(int));

  //kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0));
  
  fragment frag;
  frag.color = glm::vec3(0,0,0);
  frag.normal = glm::vec3(0,0,0);
  frag.position = glm::vec3(0,0,-10000);
  frag.depth = -10000;
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag,lock);

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

  device_svbo = NULL;
  cudaMalloc((void**)&device_svbo, vbosize*sizeof(float));
  

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------
  
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, MVP,MV, resolution,device_svbo);
  /*float* h_vbo = new float[vbosize];
  cudaMemcpy(h_vbo,device_vbo , (vbosize)*sizeof(float), cudaMemcpyDeviceToHost);
  h_vbo[1];*/
  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives,device_svbo,eye);

  /*triangle* h_tri = new triangle[ibosize/3];
   cudaMemcpy( h_tri,primitives, (ibosize/3)*sizeof(triangle), cudaMemcpyDeviceToHost);
 h_tri[0];*/
  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution,lock);

  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, MV);

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
  cudaFree( device_svbo );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
  cudaFree( lock );
}

