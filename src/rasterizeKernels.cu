// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"
#include "glm/gtc/matrix_transform.hpp"

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
float* device_nbo;
float* device_vbo_ws;

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
	  f.position.z = -100000;
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
__global__ void vertexShadeKernel(float* vbo, float* vbo_ws, int vbosize, float* nbo, cudaMat4 MV, cudaMat4 P, glm::vec2 resolution, float zNear, float zFar){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  
	  int vInd = index*3;
	  
	  //transform to ws 
	  glm::vec4 point(vbo[vInd],vbo[vInd+1], vbo[vInd+2], 1.0f);
	  point = multiplyMV_4(MV, point);
	  
	  //store in ws vbo
	  vbo_ws[vInd] = point.x;
	  vbo_ws[vInd+1] = point.y;
	  vbo_ws[vInd+2] = point.z;
	  
	  //transform to go to clip space and write to VBO
	  point=multiplyMV_4(P, point);
	  
	  point.x /= point.w;
	  point.y /= point.w;
	  point.z /= point.w;
	  
	  //transfrom to screen coord
	  point.x = (point.x+1)*(resolution.x/2.0f);
	  point.y = (-point.y+1)*(resolution.y/2.0f);		//flip y coordinate
	  point.z = (zFar - zNear)/2.0f*point.z + (zFar + zNear)/2.0f;
	  
	  //write to memory
	  vbo[vInd]=point.x;
	  vbo[vInd+1]=point.y;
	  vbo[vInd+2]=point.z;

	  //transform normals to world space
	  glm::vec3 normal(nbo[vInd], nbo[vInd+1], nbo[vInd+2]);
	  normal=glm::normalize(multiplyMV(MV, glm::vec4(normal, 0.0f)));
	  nbo[vInd] = normal.x;
	  nbo[vInd+1] = normal.y;
	  nbo[vInd+2] = normal.z;

  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, float* vbo_ws, int vbosize, float* nbo, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives, 
										glm::vec3 light, glm::vec3 viewDir){
  
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){

	  triangle tri=primitives[index];
	  
	  //get the vertices and colors of the triangle
	  int iInd = index*3;
	  int vInd = ibo[iInd]*3;

	  tri.p0 = glm::vec3(vbo[vInd], vbo[vInd+1], vbo[vInd+2]);
	  tri.n0 = glm::vec3(nbo[vInd], nbo[vInd+1], nbo[vInd+2]);
	  tri.c0 = glm::vec3(cbo[0], cbo[1], cbo[2]);

	  tri.L0 = glm::vec3(vbo_ws[vInd], vbo_ws[vInd+1], vbo_ws[vInd+2]);

	  //get 2nd vertex
	  iInd++;
	  vInd = ibo[iInd]*3;
	  tri.p1 = glm::vec3(vbo[vInd], vbo[vInd+1], vbo[vInd+2]);
	  tri.n1 = glm::vec3(nbo[vInd], nbo[vInd+1], nbo[vInd+2]);
	  tri.c1 = glm::vec3(cbo[3], cbo[4], cbo[5]);
	  
	  tri.L1 = glm::vec3(vbo_ws[vInd], vbo_ws[vInd+1], vbo_ws[vInd+2]);

	  iInd++;
	  vInd = ibo[iInd]*3;
	  tri.p2 = glm::vec3(vbo[vInd], vbo[vInd+1], vbo[vInd+2]);
	  tri.n2 = glm::vec3(nbo[vInd], nbo[vInd+1], nbo[vInd+2]);
	  tri.c2 = glm::vec3(cbo[6], cbo[7], cbo[8]);
	  
	  tri.L2 = glm::vec3(vbo_ws[vInd], vbo_ws[vInd+1], vbo_ws[vInd+2]);
	  

	  //do backface cull
	  glm::vec3 normal = glm::normalize(glm::cross(tri.L0-tri.L1, tri.L0-tri.L2));
	  if(normal.z <= 0)
		tri.draw = false;
	  else
		  tri.draw = true;

	  //find vector from ws point to light
	  tri.L0 = glm::normalize(light - tri.L0);
	  tri.L1 = glm::normalize(light - tri.L1);
	  tri.L2 = glm::normalize(light - tri.L2);

	  //write to memory
	  primitives[index]=tri;
  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, glm::vec3 viewDir){
  //need atomics to work....
	
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
	  
	  //rasterize with barycentric method
	  triangle tri = primitives[index];

	  if(tri.draw){
		  //find bounding box around triangle
		  glm::vec3 low, high;
		  getAABBForTriangle(tri, low, high);

		  for (int i = clamp(low.x, 0.0f, resolution.x-1); i<=ceil(high.x) && i<resolution.x; ++i){
			  for(int j = clamp(low.y, 0.0f, resolution.y-1); j<=ceil(high.y) && j<resolution.y; ++j){
			  
				  int fragIndex = i*resolution.y + j;
				  fragment frag = depthbuffer[fragIndex];
			  
				  //convert pixel to barycentric coord
				  glm::vec3 baryCoord = calculateBarycentricCoordinate(tri, glm::vec2(i, j));
			  
				  //check if within the triangle
				  if(isBarycentricCoordInBounds(baryCoord)){
					  float z = getZAtCoordinate(baryCoord, tri);
				  
					  //test depth
					  if(z > frag.position.z){
						  frag.position = glm::vec3(i,j,z);
						  frag.normal = glm::normalize(baryCoord.x*tri.n0 + baryCoord.y*tri.n1 + baryCoord.z*tri.n2);
						  frag.lightDir = glm::normalize(baryCoord.x*tri.L0 + baryCoord.y*tri.L1 + baryCoord.z*tri.L2);
						  frag.wsPosition = baryCoord.x*tri.p0 + baryCoord.y*tri.p1 + baryCoord.z*tri.p2;
						  //frag.color = glm::vec3(163/255.0f, 116/255.0f, 235/255.0f);
						  //frag.color = glm::vec3(163.0f/255.0f, 235.0f/255.0f, 116.0f/255.0f);
						  //frag.color = baryCoord.x*tri.c0 + baryCoord.y*tri.c1 + baryCoord.z*tri.c2;
						  frag.color = glm::vec3(abs(frag.normal.x), abs(frag.normal.y), abs(frag.normal.z));
						  depthbuffer[fragIndex] = frag;
					  }
				  }
			  }
		  }
	  }
  }
}

//per fragment rasterization
__global__ void rasterizationKernelFrag(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = x + (y*resolution.x);

	if(x <= resolution.x && y<=resolution.y){
		
		for(int i = 0; i < primitivesCount; ++i){
			
			triangle tri = primitives[i];
			fragment frag = depthbuffer[index];

			//find bounding box around triangle
			glm::vec3 low, high;
			getAABBForTriangle(tri, low, high);
			
			//throw out this fragment if ouside bounding box
			if(x<low.x || x>high.x || y<low.y || y>high.y)
				continue;

			//convert pixel to barycentric coord
			glm::vec3 baryCoord = calculateBarycentricCoordinate(tri, glm::vec2(x, y));

			//check if within the triangle
			if(isBarycentricCoordInBounds(baryCoord)){
				float z = getZAtCoordinate(baryCoord, tri);

				if(z > frag.position.z){
					frag.position = glm::vec3(x,y,z);
					frag.wsPosition = baryCoord.x*tri.p0 + baryCoord.y*tri.p1 + baryCoord.z*tri.p2;
					frag.normal = glm::normalize(baryCoord.x*tri.n0 + baryCoord.y*tri.n1 + baryCoord.z*tri.n2);
					frag.lightDir = glm::normalize(baryCoord.x*tri.L0 + baryCoord.y*tri.L1 + baryCoord.z*tri.L2);
						  
					frag.color = glm::vec3(1,1,1);				
					depthbuffer[index] = frag;
				
				}

			}
		}
	
	}
}


//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 eye){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){

	  fragment frag = depthbuffer[index];

	  if(frag.position.z > -100000){
		  //diffuse shading assuming white light
		  //frag.color = glm::clamp(glm::dot(frag.lightDir, frag.normal), 0.0f, 1.0f)*frag.color;

		  //specular shading assuming white light
		  //glm::vec3 R = glm::normalize(frag.lightDir - 2.0f*glm::dot(frag.lightDir, frag.normal)*frag.normal);
		  //glm::vec3 V = glm::normalize(eye-frag.wsPosition);
		  //frag.color += glm::vec3(1,1,1)*pow(glm::clamp(glm::dot(R, V), 0.0f, 1.0f), 40.0f);

		  //depthbuffer[index] = frag;
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

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, 
					   float* nbo, int nbosize, camera& cam, glm::vec3 lightPos){

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
  frag.position = glm::vec3(0,0,-100000);
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

  device_vbo_ws = NULL;
  cudaMalloc((void**)&device_vbo_ws, vbosize*sizeof(float));
  cudaMemcpy( device_vbo_ws, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_nbo = NULL;
  cudaMalloc((void**)&device_nbo, nbosize*sizeof(float));
  cudaMemcpy( device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);
  
  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //build the MVP matrix
  glm::mat4 model = glm::rotate(glm::mat4(1.0f), 90.0f, glm::vec3(0,0,1));		//temp identity model view matrix for now
  glm::mat4 view = glm::lookAt(cam.eye, cam.center, cam.up);
  glm::mat4 projection = glm::perspective(cam.fov, 1.0f, cam.zNear, cam.zFar);

  glm::vec4 lightTemp = view*model*glm::vec4(lightPos, 0.0);
  lightPos.x = lightTemp.x; lightPos.y = lightTemp.y, lightPos.z = lightTemp.z;

  //performance analysis stuff
  //time cuda call
  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, device_vbo_ws, vbosize, device_nbo, utilityCore::glmMat4ToCudaMat4(view*model), 
												utilityCore::glmMat4ToCudaMat4(projection), resolution, cam.zNear, cam.zFar);
  checkCUDAErrorWithLine("vertex shade failed!");

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, device_vbo_ws, vbosize, device_nbo, device_cbo, cbosize, device_ibo, ibosize, primitives, 
														lightPos, cam.eye-cam.center);
  checkCUDAErrorWithLine("primitive assembly failed!");

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution, cam.eye-cam.center);
  //rasterizationKernelFrag<<<fullBlocksPerGrid, threadsPerBlock>>>(primitives, ibosize/3, depthbuffer, resolution);
  checkCUDAErrorWithLine("rasterize kernel failed!");

  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, cam.eye);
  checkCUDAErrorWithLine("fragment failed!");

  cudaDeviceSynchronize();
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

  cudaDeviceSynchronize();

  //end time record
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  cout<<time<<" has passed"<<endl;


  kernelCleanup();

  checkCUDAError("Kernel failed!");
}

void kernelCleanup(){
  cudaFree( primitives );
  cudaFree( device_vbo );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( device_vbo_ws);
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}

