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
float* device_nbo;
triangle* primitives;

int* stencilbuffer;

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

//clears the stencil buffer
__global__ void clearStencilBuffer(glm::vec2 resolution, int* stencilbuffer, int inital)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
		stencilbuffer[index] = inital;
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
__global__ void vertexShadeKernel(float* vbo, int vbosize, float* nbo, int nbosize, glm::mat4 modelViewProjection, glm::mat4 viewPort){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  glm::vec4 point(vbo[index*3], vbo[index*3+1], vbo[index*3+2], 1);
	
	  point = modelViewProjection * point;	 
	 
	  point = point/point.w;	  
	  point = viewPort * point;

	  vbo[index*3] = point.x;
	  vbo[index*3+1] = point.y;	  
	  vbo[index*3+2] = point.z;	  
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, triangle* primitives){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;

  if(index<primitivesCount){
	  int i1 = ibo[index*3];
	  int i2 = ibo[index*3+1];
	  int i3 = ibo[index*3+2];

	  primitives[index].p0 = glm::vec3(vbo[i1*3], vbo[i1*3+1], vbo[i1*3+2]);
	  primitives[index].p1 = glm::vec3(vbo[i2*3], vbo[i2*3+1], vbo[i2*3+2]);
	  primitives[index].p2 = glm::vec3(vbo[i3*3], vbo[i3*3+1], vbo[i3*3+2]);

	  primitives[index].n0 = glm::vec3(nbo[i1*3], nbo[i1*3+1], nbo[i1*3+2]);
	  primitives[index].n1 = glm::vec3(nbo[i2*3], nbo[i2*3+1], nbo[i2*3+2]);
	  primitives[index].n2 = glm::vec3(nbo[i3*3], nbo[i3*3+1], nbo[i3*3+2]);

	  /*primitives[index].c0 = glm::vec3(cbo[i1*3], cbo[i1*3+1], cbo[i1*3+2]);
	  primitives[index].c1 = glm::vec3(cbo[i2*3], cbo[i2*3+1], cbo[i2*3+2]);
	  primitives[index].c2 = glm::vec3(cbo[i3*3], cbo[i3*3+1], cbo[i3*3+2]);
*/
	  primitives[index].c0 = glm::vec3(cbo[index%3 + 0], cbo[index%3 + 1], cbo[index%3 + 2]);
	  primitives[index].c1 = glm::vec3(cbo[index%3 + 0], cbo[index%3 + 1], cbo[index%3 + 2]);
	  primitives[index].c2 = glm::vec3(cbo[index%3 + 0], cbo[index%3 + 1], cbo[index%3 + 2]);


	  /*primitives[index].c0 = glm::vec3(cbo[index%3 + 0], cbo[index%3 + 1], cbo[index%3 + 2]);
	  primitives[index].c1 = glm::vec3(cbo[index%3 + 3], cbo[index%3 + 4], cbo[index%3 + 5]);
	  primitives[index].c2 = glm::vec3(cbo[index%3 + 6], cbo[index%3 + 7], cbo[index%3 + 8]);*/

	  /*primitives[index].c0 = glm::vec3(cbo[0], cbo[1], cbo[2]);
	  primitives[index].c1 = glm::vec3(cbo[0], cbo[1], cbo[2]);
	  primitives[index].c2 = glm::vec3(cbo[0], cbo[1], cbo[2]);*/

	  primitives[index].isRender = true;
  }
}


__global__ void backFaceCulling(triangle* primitives, int primitivesCount, glm::vec3 viewDir)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<primitivesCount){		
		float result = glm::dot(viewDir, primitives[index].n0);
		if(result > 0)
			primitives[index].isRender = false;		
	}	
}

__device__ glm::vec3 getEdgeValue(glm::vec3 coordinate, float area)
{
	return coordinate * area * 2.0f;
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, glm::vec3 cameraPos, glm::vec3 lookAt){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
	  
	  if(primitives[index].isRender){
		  glm::vec3 minP, maxP;
		  getAABBForTriangle(primitives[index], minP, maxP);	

		  //clipping
		  if(minP.x < 0)
			  minP.x = 0;
		  if(minP.y < 0)
			  minP.y = 0;
		  if(maxP.x > resolution.x)
			  maxP.x = resolution.x;
		  if(maxP.y > resolution.y)
			  maxP.y = resolution.y;


		  float x01 = primitives[index].p1.x - primitives[index].p0.x;	  
		  float x12 = primitives[index].p2.x - primitives[index].p1.x;
		  float x20 = primitives[index].p0.x - primitives[index].p2.x;

		  float y01 = primitives[index].p0.y - primitives[index].p1.y;	  
		  float y12 = primitives[index].p1.y - primitives[index].p2.y;
		  float y20 = primitives[index].p2.y - primitives[index].p0.y;

		  float sign01 = glm::sign(y01);
		  float sign12 = glm::sign(y12);
		  float sign20 = glm::sign(y20);
	 	  
		  float triArea = abs(calculateSignedArea(primitives[index]));
		  glm::vec3 bCoordinate = calculateBarycentricCoordinate(primitives[index], glm::vec2(minP.x,minP.y));	
		  glm::vec3 temp = bCoordinate;

		for(int j = minP.y; j <= maxP.y; j++)
		{		
			//glm::vec3 temp = bCoordinate;
			glm::vec3 Evalue;
			Evalue = getEdgeValue(bCoordinate, triArea);
		
			for(int i = minP.x; i <= maxP.x; i++)
			{			
				if(Evalue.x >= -sign12 * y12 - 0.f && Evalue.y >= -sign20 * y20 - 0.f  && Evalue.z >= -sign01 * y01 - 0.f)
				{				
					int depthIndex = i + j*resolution.x;	
					float interpolateZ = 1.0f + getZAtCoordinate(temp, primitives[index]);			
					//depth testing
					if(interpolateZ >= depthbuffer[depthIndex].position.z)
					{
						depthbuffer[depthIndex].position.z = interpolateZ;
						depthbuffer[depthIndex].color = glm::vec3(1,1,1) * interpolateZ * 10.f;//temp.x * primitives[index].c0 + temp.y * primitives[index].c1 + temp.z * primitives[index].c2;	
						depthbuffer[depthIndex].normal = primitives[index].n0;					
					}				
				}
				Evalue.x += y12; Evalue.y += y20; Evalue.z += y01;
				temp = Evalue * 0.5f / triArea;	
			}	

			glm::vec3 Tvalue;
			Tvalue = getEdgeValue(bCoordinate, triArea);
			Tvalue.x += x12; Tvalue.y += x20; Tvalue.z += x01;
			bCoordinate = Tvalue * 0.5f / triArea;		
		}
	}
  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, glm::mat4 modelViewProjection, glm::mat4 viewPort, glm::vec4 lightPos){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  
  if(x<=resolution.x && y<=resolution.y){
	  float diffuse = max(glm::dot(depthbuffer[index].normal, glm::vec3(lightPos)), 0.0f);
	  //depthbuffer[index].color *= diffuse * glm::vec3(1,1,1) * 0.2f;
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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, 
					   glm::mat4 modelViewProjection, glm::mat4 viewPort, glm::vec4 lightPos, glm::vec3 cameraPos, glm::vec3 lookAt){

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

  //set up stencilbuffer
  stencilbuffer = NULL;
  cudaMalloc((void**)&depthbuffer, (int)resolution.x*(int)resolution.y*sizeof(int));

  //kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0));
  
  fragment frag;
  frag.color = glm::vec3(0,0,0);
  frag.normal = glm::vec3(0,0,0);
  frag.position = glm::vec3(0,0,-10000);
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, frag);

  clearStencilBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, stencilbuffer, 0);
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
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_nbo, nbosize, modelViewProjection, viewPort);

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, device_nbo, nbosize, primitives);

  cudaDeviceSynchronize();

   //------------------------------
  //blackface culling
  //------------------------------
  glm::vec3 viewDir = glm::normalize(lookAt - cameraPos);
  backFaceCulling<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, viewDir);

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution, cameraPos, lookAt);

  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, modelViewProjection, viewPort, lightPos);

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
  cudaFree( device_nbo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}

