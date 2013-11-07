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
bool* lockbuffer;

int* cudastencilbuffer;

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
__global__ void clearDepthBuffer(glm::vec2 resolution, fragment* buffer, fragment frag, bool* lockbuffer){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      fragment f = frag;
      f.position.x = x;
      f.position.y = y;
      buffer[index] = f;
	  lockbuffer[index] = true;
    }
}

//clears the stencil buffer
__global__ void clearStencilBuffer(glm::vec2 resolution, int* cudastencilbuffer, int inital)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
		cudastencilbuffer[index] = inital;
	}
}

//just clear z buffer
__global__ void clearZbuffer(glm::vec2 resolution, fragment* buffer)
{
	 int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
		buffer[index].position.z = 10000;
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
	  /*primitives[index].c0 = glm::vec3(cbo[index%3 + 0], cbo[index%3 + 1], cbo[index%3 + 2]);
	  primitives[index].c1 = glm::vec3(cbo[index%3 + 0], cbo[index%3 + 1], cbo[index%3 + 2]);
	  primitives[index].c2 = glm::vec3(cbo[index%3 + 0], cbo[index%3 + 1], cbo[index%3 + 2]);*/


	 /* primitives[index].c0 = glm::vec3(cbo[index%3 + 0], cbo[index%3 + 1], cbo[index%3 + 2]);
	  primitives[index].c1 = glm::vec3(cbo[index%3 + 3], cbo[index%3 + 4], cbo[index%3 + 5]);
	  primitives[index].c2 = glm::vec3(cbo[index%3 + 6], cbo[index%3 + 7], cbo[index%3 + 8]);*/

	  primitives[index].c0 = glm::vec3(cbo[0], cbo[1], cbo[2]);
	  primitives[index].c1 = glm::vec3(cbo[0], cbo[1], cbo[2]);
	  primitives[index].c2 = glm::vec3(cbo[0], cbo[1], cbo[2]);

	  primitives[index].isRender = true;
  }
}


__global__ void backFaceCulling(triangle* primitives, int primitivesCount, glm::vec3 viewDir)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<primitivesCount){		
		float result = glm::dot(glm::normalize(viewDir), primitives[index].n0);
		if(result > 0.5f)
			primitives[index].isRender = false;		
	}	
}

__device__ glm::vec3 getEdgeValue(glm::vec3 coordinate, float area)
{
	return coordinate * area * 2.0f;
}

__device__ float setupEdge(glm::vec3 a, glm::vec3 b, glm::vec3 c)
{
	return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}


__device__ bool fAtomicMin(float* address, float value)
{
	unsigned int* address_as_ull = (unsigned int*)address;
    unsigned int old = *address_as_ull, assumed;
	if(__int_as_float(old) < value) return false;
    do {
		//make sure the lastest update
		old = *address_as_ull;
		if(__int_as_float(old) < value) return false;		
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,__float_as_int(value));		
	} while (assumed != *address_as_ull);
    return true;
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, 
									glm::vec3 cameraPos, glm::vec3 lookAt, int colorRepresent, bool* lockbuffer){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
	  
	  if(primitives[index].isRender){
		  glm::vec3 minP, maxP;
		  getAABBForTriangle(primitives[index], minP, maxP);	
		  
		  //clipping
		 if(maxP.x >= 0 && maxP.y >= 0 && minP.x <= resolution.x && minP.y <= resolution.y){
			  			  
			  if(minP.x < 0)
				  minP.x = 0;
			  if(minP.y < 0)
				  minP.y = 0;
			  if(maxP.x >= resolution.x)
				  maxP.x = resolution.x-1;
			  if(maxP.y >= resolution.y)
				  maxP.y = resolution.y-1;	  

			  float x01 = primitives[index].p1.x - primitives[index].p0.x;	  
			  float x12 = primitives[index].p2.x - primitives[index].p1.x;
			  float x20 = primitives[index].p0.x - primitives[index].p2.x;

			  float y01 = primitives[index].p0.y - primitives[index].p1.y;	  
			  float y12 = primitives[index].p1.y - primitives[index].p2.y;
			  float y20 = primitives[index].p2.y - primitives[index].p0.y;
			
			  float triArea = (calculateSignedArea(primitives[index]));
			  glm::vec3 bCoordinate = calculateBarycentricCoordinate(primitives[index], glm::vec2((int)minP.x,(int)minP.y));	
			  	 
			  //back face culling
			//if(triArea < 0){
				for(int j = minP.y; j <= maxP.y; j++)
				{		 
					glm::vec3 Evalue;
					Evalue = bCoordinate;	
		
					for(int i = minP.x; i <= maxP.x; i++)
					{			
						if(isBarycentricCoordInBounds(Evalue))
						{				
							int depthIndex = i + j*resolution.x;	
							float interpolateZ = -getZAtCoordinate(Evalue, primitives[index]);							
							
							if(fAtomicMin(&depthbuffer[depthIndex].position.z, interpolateZ))
							{								
								depthbuffer[depthIndex].normal = Evalue.x * primitives[index].n0 + Evalue.y * primitives[index].n1 + Evalue.z * primitives[index].n2;	
								if(colorRepresent == 0)
									depthbuffer[depthIndex].color = glm::vec3(1,1,1) * (1-interpolateZ) * 10.f;
								else if(colorRepresent == 1)
									depthbuffer[depthIndex].color = glm::abs(depthbuffer[depthIndex].normal);
								else 
									depthbuffer[depthIndex].color = Evalue.x * primitives[index].c0 + Evalue.y * primitives[index].c1 + Evalue.z * primitives[index].c2;	
									
							}								
						}
						Evalue.x -= (0.5f * y12) / triArea; Evalue.y -= (0.5f * y20) / triArea; Evalue.z -= (0.5f * y01) / triArea;						
					}

					glm::vec3 Tvalue;				
					Tvalue = bCoordinate;
					Tvalue.x -= (0.5f * x12) / triArea; Tvalue.y -= (0.5f * x20) / triArea; Tvalue.z -= (0.5f * x01) / triArea;				
					bCoordinate = Tvalue;
				}
			// }
		}
	}
  }
}

__global__ void rasterizationKernelStencil(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, glm::vec3 cameraPos, glm::vec3 lookAt, 
										   int* cudastencilbuffer, int stencilValue, int colorRepresent){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
	  
	  if(primitives[index].isRender){
		  glm::vec3 minP, maxP;
		  getAABBForTriangle(primitives[index], minP, maxP);	

		   //clipping
		 if(maxP.x >= 0 && maxP.y >= 0 && minP.x <= resolution.x && minP.y <= resolution.y){
			  			  
			  if(minP.x < 0)
				  minP.x = 0;
			  if(minP.y < 0)
				  minP.y = 0;
			  if(maxP.x >= resolution.x)
				  maxP.x = resolution.x-1;
			  if(maxP.y >= resolution.y)
				  maxP.y = resolution.y-1;	  

			  float x01 = primitives[index].p1.x - primitives[index].p0.x;	  
			  float x12 = primitives[index].p2.x - primitives[index].p1.x;
			  float x20 = primitives[index].p0.x - primitives[index].p2.x;

			  float y01 = primitives[index].p0.y - primitives[index].p1.y;	  
			  float y12 = primitives[index].p1.y - primitives[index].p2.y;
			  float y20 = primitives[index].p2.y - primitives[index].p0.y;
			
			  float triArea = (calculateSignedArea(primitives[index]));
			  glm::vec3 bCoordinate = calculateBarycentricCoordinate(primitives[index], glm::vec2((int)minP.x,(int)minP.y));	
			  	 
			for(int j = minP.y; j <= maxP.y; j++)
			{		
				//glm::vec3 temp = bCoordinate;
				glm::vec3 Evalue;
				Evalue = bCoordinate;	
		
				for(int i = minP.x; i <= maxP.x; i++)
				{			
					if(isBarycentricCoordInBounds(Evalue))
					{			
						int depthIndex = i + j*resolution.x;							
						if(stencilValue == cudastencilbuffer[depthIndex] + 1 || stencilValue == cudastencilbuffer[depthIndex]){
							float interpolateZ = -getZAtCoordinate(Evalue, primitives[index]);							
							if(fAtomicMin(&depthbuffer[depthIndex].position.z, interpolateZ))
							{							
								depthbuffer[depthIndex].normal = Evalue.x * primitives[index].n0 + Evalue.y * primitives[index].n1 + Evalue.z * primitives[index].n2;	
								if(colorRepresent == 0)
									depthbuffer[depthIndex].color = glm::vec3(1,1,1) * (1-interpolateZ) * 10.f;
								else if(colorRepresent == 1)
									depthbuffer[depthIndex].color = glm::abs(depthbuffer[depthIndex].normal);
								else 
									depthbuffer[depthIndex].color = Evalue.x * primitives[index].c0 + Evalue.y * primitives[index].c1 + Evalue.z * primitives[index].c2;
								cudastencilbuffer[depthIndex] = stencilValue;
							}						
						}						
					}
					Evalue.x -= (0.5f * y12) / triArea; Evalue.y -= (0.5f * y20) / triArea; Evalue.z -= (0.5f * y01) / triArea;	
				}
				glm::vec3 Tvalue;				
				Tvalue = bCoordinate;
				Tvalue.x -= (0.5f * x12) / triArea; Tvalue.y -= (0.5f * x20) / triArea; Tvalue.z -= (0.5f * x01) / triArea;				
				bCoordinate = Tvalue;	
			}
		 }
	}
  }
}


//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, glm::mat4 inverseModelViewProjection, glm::mat4 inverseViewPort, glm::vec4 lightPos, int colorRepresent){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y){
	glm::vec4 posWorld = inverseModelViewProjection * inverseViewPort * glm::vec4(depthbuffer[index].position, 1);	  
	float diffuse = max(glm::dot(depthbuffer[index].normal, glm::vec3(lightPos - posWorld)), 0.0f);	
	
	if(colorRepresent != 0){
		depthbuffer[index].color = depthbuffer[index].color * diffuse * glm::vec3(1,1,1) * 0.2f + depthbuffer[index].color * 0.2f;		
	}
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

//A easy anti aliasing
__global__ void antialiasing(fragment* depthbuffer, glm::vec2 resolution)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	
	if(x >= 1 && y >= 1 && x < resolution.x && y < resolution.y)
	{
		glm::vec3 sumColor = glm::vec3(0.f);
		for(int i = -1; i <= 1; i++){
			for(int j = -1; j <= 1; j++)
			{
				index = x + i + ((y+j) * resolution.x);
				sumColor += depthbuffer[index].color;
			}
		}
		index = x + (y * resolution.x);
		depthbuffer[index].color = sumColor / 9.0f;
	}
}


void initalKernel(glm::vec2 resolution, int* stencilBuffer)
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

  //set up lockbuffer
  lockbuffer = NULL;
  cudaMalloc((void**)&lockbuffer, (int)resolution.x*(int)resolution.y*sizeof(bool));

  //set up stencilbuffer
  cudastencilbuffer = NULL;
  cudaMalloc((void**)&cudastencilbuffer, (int)resolution.x*(int)resolution.y*sizeof(int));
  cudaMemcpy( cudastencilbuffer, stencilBuffer, (int)resolution.x*(int)resolution.y*sizeof(int), cudaMemcpyHostToDevice);

  //kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0));
  
  fragment frag;
  frag.color = glm::vec3(0,0,0);
  frag.normal = glm::vec3(0,0,0);
  frag.position = glm::vec3(0,0,10000);
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, frag, lockbuffer);
}


// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, 
					   glm::mat4 modelViewProjection, glm::mat4 viewPort, glm::vec4 lightPos, glm::vec3 cameraPos, glm::vec3 lookAt, bool isStencil, int first, int second, char keyValue, int* stencilBuffer){
 //// set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));

  clearZbuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer); 

  //keyvalue
  int colorRepresent;
  if(keyValue == 'q')
	colorRepresent = 0;
  else if(keyValue == 'w')
	colorRepresent = 1;
  else if(keyValue == 'e')
	colorRepresent = 2;
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
#if BACKCULLING == 1
  backFaceCulling<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, viewDir);
#endif

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  //if(!isStencil)
	//  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution, cameraPos, lookAt, colorRepresent, lockbuffer);
  //else{	
  rasterizationKernelStencil<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution, cameraPos, lookAt, cudastencilbuffer, first, colorRepresent);		
  //}
  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  glm::mat4 inverseModelView = glm::inverse(modelViewProjection);
  glm::mat4 inverseViewPort = glm::inverse(viewPort);
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, inverseModelView, inverseViewPort, lightPos, colorRepresent);

  cudaDeviceSynchronize();


#if ANTIALIASING == 1
  //------------------------------
  //anti aliasing
  //------------------------------
  antialiasing<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution);

  cudaDeviceSynchronize();
#endif 
  
  cudaMemcpy( stencilBuffer, cudastencilbuffer, (int)resolution.x*(int)resolution.y*sizeof(int), cudaMemcpyDeviceToHost);  
}

void renderKernel(uchar4* PBOpos, glm::vec2 resolution)
{
 
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));

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
  cudaFree( cudastencilbuffer );
  cudaFree( lockbuffer );
}

