// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"

using namespace glm;

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_screenvbo;
float* device_cbo;
int* device_ibo;
float* device_nbo;
triangle* primitives;

bool diff1spec0 = 0;


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
__global__ void vertexShadeKernel(float* vbo, float* screenvbo, int vbosize,  float* nbo, int nbosize, mat4 modelViewProjection, vec2 resolution, mat4 model){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
      vec4 worldVertex(vbo[index*3], vbo[index*3+1], vbo[index*3+2], 1.0f);
	  vec4 worldNormal(nbo[index*3], nbo[index*3+1], nbo[index*3+2], 0.0f);
      vec4 projectedVertex = modelViewProjection * worldVertex;
      projectedVertex = (1/projectedVertex.w) * projectedVertex;
      float xNDC = (projectedVertex.x + 1)/2.0f;
      float yNDC = (projectedVertex.y + 1)/2.0f;
      screenvbo[index*3] = xNDC*resolution.x;
      screenvbo[index*3+1] = yNDC* resolution.y;
      screenvbo[index*3+2] = projectedVertex.z;
	
	  vec4 worldSpaceVert = model * worldVertex;
	  vec4 worldSpaceNorm = model * worldNormal;
	  vbo[index*3] = worldSpaceVert.x;
	  vbo[index*3+1] = worldSpaceVert.y;
	  vbo[index*3+2] = worldSpaceVert.z;
	  nbo[index*3] = worldSpaceNorm.x;
	  nbo[index*3+1] = worldSpaceNorm.y;
	  nbo[index*3+2] = worldSpaceNorm.z;
	  
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, float* screenvbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives,
										float* nbo){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
		int vboIndexStart1 = ibo[index*3];
		int vboIndexStart2 = ibo[index*3+1];
		int vboIndexStart3 = ibo[index*3+2];

		//create primitive
		triangle t;

		t.p0 = vec3(screenvbo[vboIndexStart1*3], screenvbo[vboIndexStart1*3+1], screenvbo[vboIndexStart1*3+2]);
		t.p1 = vec3(screenvbo[vboIndexStart2*3], screenvbo[vboIndexStart2*3+1], screenvbo[vboIndexStart2*3+2]);
		t.p2 = vec3(screenvbo[vboIndexStart3*3], screenvbo[vboIndexStart3*3+1], screenvbo[vboIndexStart3*3+2]);
		t.c0 = vec3(1,0,0);//vec3(cbo[vboIndexStart1*3], cbo[vboIndexStart1*3+1], cbo[vboIndexStart1*3+2]);
		t.c1 = vec3(0,1,0);//vec3(cbo[vboIndexStart2*3], cbo[vboIndexStart2*3+1], cbo[vboIndexStart2*3+2]);
		t.c2 = vec3(0,0,1);//vec3(cbo[vboIndexStart3*3], cbo[vboIndexStart3*3+1], cbo[vboIndexStart3*3+2]);

		primitives[index] = t;
  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, int* ibo, float* nbo, float* vbo, vec3 eye){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){

	int vboIndexStart1 = ibo[index*3];
	int vboIndexStart2 = ibo[index*3+1];
	int vboIndexStart3 = ibo[index*3+2];
				
	vec3 normalP0 = vec3(nbo[vboIndexStart1*3], nbo[vboIndexStart1*3+1], nbo[vboIndexStart1*3+2]);
	vec3 normalP1 = vec3(nbo[vboIndexStart2*3], nbo[vboIndexStart2*3+1], nbo[vboIndexStart2*3+2]);
	vec3 normalP2 = vec3(nbo[vboIndexStart3*3], nbo[vboIndexStart3*3+1], nbo[vboIndexStart3*3+2]);

	vec3 worldP0 = vec3(vbo[vboIndexStart1*3], vbo[vboIndexStart1*3+1], vbo[vboIndexStart1*3+2]);
	vec3 worldP1 = vec3(vbo[vboIndexStart2*3], vbo[vboIndexStart2*3+1], vbo[vboIndexStart2*3+2]);
	vec3 worldP2 = vec3(vbo[vboIndexStart3*3], vbo[vboIndexStart3*3+1], vbo[vboIndexStart3*3+2]);

	//check if any normals are facing camera
	bool facingCamera = false;
	vec3 eyeToP0 = eye-worldP0;
	vec3 eyeToP1 = eye-worldP1;
	vec3 eyeToP2 = eye-worldP2;
	if (dot(normalize(eyeToP0),normalize(normalP0)) >= 0){
		facingCamera = true;
	}else if (dot(normalize(eyeToP1),normalize(normalP1)) >= 0){
		facingCamera = true;
	}else if (dot(normalize(eyeToP2),normalize(normalP2)) >= 0){
		facingCamera = true;
	}

	if (facingCamera){

		//bounding box
		vec3 minP;
		vec3 maxP;

		getAABBForTriangle(primitives[index], minP, maxP);

		vec3 colorP0 = primitives[index].c0;
		vec3 colorP1 = primitives[index].c1;
		vec3 colorP2 = primitives[index].c2;

		for (int x = minP.x; x < maxP.x; x++){
			for (int y = minP.y; y < maxP.y; y++){
				vec3 barycentricCoord = calculateBarycentricCoordinate(primitives[index], vec2((float)x,(float)y));
				if (isBarycentricCoordInBounds(barycentricCoord)){
					vec3 bc = vec3(barycentricCoord.x, barycentricCoord.y, getZAtCoordinate(barycentricCoord, primitives[index]));
					//interpolate color
					vec3 interpColor = barycentricCoord.x*colorP0+barycentricCoord.y*colorP1+barycentricCoord.z*colorP2;

					//interpolate normal
					vec3 interpNormal = barycentricCoord.x* normalP0+barycentricCoord.y* normalP1+barycentricCoord.z* normalP2;

					//world position
					vec3 interpWorldPos = barycentricCoord.x* worldP0+barycentricCoord.y* worldP1+barycentricCoord.z* worldP2;

					fragment f;
					f.color = interpColor;//vec3(.8f, .8f, .8f);
					f.normal = normalize(interpNormal);
					f.position = bc;
					f.worldPosition = interpWorldPos;
					int dbindex = y*resolution.x+x;

					if (f.position.z >= depthbuffer[dbindex].position.z)
						if (dbindex >= 0 && dbindex < resolution.x*resolution.y)
							depthbuffer[dbindex] = f;
					
				}
			}
		}
	}
  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, vec3 lightPos, vec3 eye, bool diff1spec0){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  if (depthbuffer[index].worldPosition.x > -999999){
		vec3 surfaceColor = depthbuffer[index].color;
		vec3 lightToPos = lightPos-depthbuffer[index].worldPosition;
		lightToPos = normalize(lightToPos);
		vec3 color;
		if (diff1spec0){//diffuse
			//depthbuffer[index].color = abs(lightPos-depthbuffer[index].worldPosition);
			color = dot(depthbuffer[index].normal, lightToPos)*surfaceColor;
		}else{//specular
			vec3 eyeToPos = normalize(depthbuffer[index].worldPosition-eye);
			vec3 posNormal = normalize(depthbuffer[index].normal);
			vec3 reflectedDir = eyeToPos - 2.0f*posNormal*(dot(eyeToPos,posNormal));
			reflectedDir = normalize(reflectedDir);
			float specExponent = 0.3;
			float D = dot(reflectedDir, lightToPos);
			if (D < 0) D = 0;
			color = surfaceColor*pow(D, specExponent);
		}
		depthbuffer[index].color = color;
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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, 
						int ibosize, float* nbo, int nbosize, mat4 modelViewProjection, mat4 model, vec3 lightPos, vec3 eye){

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
  frag.color = glm::vec3(.2,.2,.2);
  frag.normal = glm::vec3(0,0,0);
  frag.position = glm::vec3(0,0,-10000);
  frag.worldPosition = glm::vec3(-999999,-999999,-999999);
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

  device_screenvbo = NULL;
  cudaMalloc((void**)&device_screenvbo, vbosize*sizeof(float));

  tileSize = 512;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, device_screenvbo, vbosize, device_nbo, nbosize, modelViewProjection, resolution, model);
  checkCUDAError("Kernel failed at vertexShadeKernel!");

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, device_screenvbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, 
															primitives, device_nbo);
  checkCUDAError("Kernel failed at primitiveAssemblyKernel!");

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution, device_ibo, device_nbo, device_vbo, eye);
  checkCUDAError("Kernel failed at rasterizationKernel!");

  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, lightPos, eye, diff1spec0);
  checkCUDAError("Kernel failed at fragmentShadeKernel!");

  cudaDeviceSynchronize();
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
  checkCUDAError("Kernel failed at render!");
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);
  checkCUDAError("Kernel failed at sendImageToPBO!");
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
  cudaFree( device_screenvbo );
  cudaFree( device_nbo );
}

