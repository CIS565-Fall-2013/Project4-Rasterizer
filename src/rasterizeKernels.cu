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

#define DEBUG 0

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_cbo;
float* device_nbo;
int* device_ibo;
triangle* primitives;
light* device_lights;

void checkCUDAError(const char *msg, int line = -1)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        if( line >= 0 )
        {
            fprintf(stderr, "Line %d: ", line);
        }
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
	  f.position.z = -1e6; // Look: Depth is initialized to some small number
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
//Transform incoming vertex position from model to clip coordinates.
//Use model matrix to transform into model space. Use view matrix to transform into camera space. Then,
//use projection matrix to transform into clip space. Next, convert to NDC. Lastly, convert to window coordinates
__global__ void vertexShadeKernel(float* vbo, int vbosize, float* nbo, int nbosize, mat4 mvp, mat4 mvInvT, vec2 reso, float zNear, float zFar)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(index<vbosize/3)
  {
	  const int id1 = index * 3;
	  const int id2 = id1 + 1;
	  const int id3 = id1 + 2;

	  // set up vbo
	  vec4 hPoint(vbo[id1], vbo[id2], vbo[id3], 1.0f);
	  
	  // to clip
	  hPoint = mvp * hPoint;
	  float wClip = hPoint.w;

	  // to ndc
	  vec4 ndcPoint(hPoint.x / wClip, hPoint.y / wClip, hPoint.z / wClip, hPoint.w / wClip);
	  
	  // to window
	  vec3 windowPoint;

	  // add 1 to get NDC to range [0, 2]. Then multiply by reso.x or y * 0.5 so that window coordinates go from [0, reso.x or y]
	  windowPoint.x = reso.x * (ndcPoint.x + 1.f) * 0.5f; // range: [0, reso.x]
	  windowPoint.y = reso.y * (ndcPoint.y + 1.f) * 0.5f; // range: [0, reso.y]
	  //windowPoint.z = ndcPoint.z; // range: [-1, 1]
	  //windowPoint.z = 0.5f * (zFar - zNear) * ndcPoint.z + 0.5f * (zFar + zNear); // range: [zNear, zFar]
	  windowPoint.z = (ndcPoint.z + 1.f) * 0.5f; // range: [0, 1]

	  vbo[id1] = windowPoint.x;
	  vbo[id2] = windowPoint.y;
	  vbo[id3] = windowPoint.z;

	  // set up nbo
	  vec4 normal(nbo[id1], nbo[id2], nbo[id3], 0.0f);
	  vec4 normalEye = mvInvT * normal;

	  nbo[id1] = normalEye.x;
	  nbo[id2] = normalEye.y;
	  nbo[id3] = normalEye.z;
  }
}

//TODO: Implement primative assembly
//Given the vbo, cbo, ibo, group them into triangles and output the result
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, triangle* primitives)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;

  if(index<primitivesCount)
  {
	  // ibo indices
	  const int iboId1 = index * 3;
	  const int iboId2 = iboId1 + 1;
	  const int iboId3 = iboId1 + 2;

	  // vbo indices for each ibo index
	  const int nvboId11 = ibo[iboId1] * 3;
	  const int nvboId12 = nvboId11 + 1;
	  const int nvboId13 = nvboId11 + 2;

	  const int nvboId21 = ibo[iboId2] * 3;
	  const int nvboId22 = nvboId21 + 1;
	  const int nvboId23 = nvboId21 + 2;

	  const int nvboId31 = ibo[iboId3] * 3;
	  const int nvboId32 = nvboId31 + 1;
	  const int nvboId33 = nvboId31 + 2;
	  
	  // cbo indices
	  const int cboId1 = index % 3;
	  const int cboId2 = cboId1 + 1;
	  const int cboId3 = cboId2 + 2;

	  // retrieve vertices
	  vec3 vert1 = vec3(vbo[nvboId11], vbo[nvboId12], vbo[nvboId13]);
	  vec3 vert2 = vec3(vbo[nvboId21], vbo[nvboId22], vbo[nvboId23]);
	  vec3 vert3 = vec3(vbo[nvboId31], vbo[nvboId32], vbo[nvboId33]);

	  // retrieve normals
	  vec3 normal1 = vec3(nbo[nvboId11], nbo[nvboId12], nbo[nvboId13]);
	  vec3 normal2 = vec3(nbo[nvboId21], nbo[nvboId22], nbo[nvboId23]);
	  vec3 normal3 = vec3(nbo[nvboId31], nbo[nvboId32], nbo[nvboId33]);

	  // retrieve colors
	  //vec3 vert1Color = vec3(cbo[cboId1], cbo[cboId2], cbo[cboId3]);
	  //vec3 vert2Color = vec3(cbo[cboId1], cbo[cboId2], cbo[cboId3]);
	  //vec3 vert3Color = vec3(cbo[cboId1], cbo[cboId2], cbo[cboId3]);

	  vec3 vert1Color = vec3(cbo[0], cbo[1], cbo[2]);
	  vec3 vert2Color = vec3(cbo[3], cbo[4], cbo[5]);
	  vec3 vert3Color = vec3(cbo[6], cbo[7], cbo[8]);

	  // build triangle
	  triangle tri;

	  tri.p0 = vert1;
	  tri.p1 = vert2;
	  tri.p2 = vert3;
	  tri.c0 = vert1Color;
	  tri.c1 = vert2Color;
	  tri.c2 = vert3Color;
	  tri.n0 = normal1;
	  tri.n1 = normal2;
	  tri.n2 = normal3;

	  primitives[index] = tri;
  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<primitivesCount)
	{
		triangle tri = primitives[index];

		//float signedarea = calculateSignedArea(tri);

		//if (signedarea < 1e-10)
		//	return;

		// TODO: Backface culling: Use normal to figure out if the triangle is facing the camera or not. Skip the ones that are facing away from the camera.

		vec3 triMinPoint;
		vec3 triMaxPoint;
		
		getAABBForTriangle(tri, triMinPoint, triMaxPoint);
	  
		triMinPoint.x = triMinPoint.x > 0 ? triMinPoint.x : 0;
		triMinPoint.y = triMinPoint.y > 0 ? triMinPoint.y : 0;
		triMaxPoint.x = triMaxPoint.x < resolution.x ? triMaxPoint.x : resolution.x;
		triMaxPoint.y = triMaxPoint.y < resolution.y ? triMaxPoint.y : resolution.y;

		// go through each pixel within the AABB for the triangle and fill the depthbuffer (fragments) appropriately
		for (int x = triMinPoint.x ; x < triMaxPoint.x ; ++x)
		{
			for (int y = triMinPoint.y ; y < triMaxPoint.y ; ++y)
			{
				vec2 pointInTri((float)x,(float)y);
				vec3 bc = calculateBarycentricCoordinate(tri, pointInTri);
				if (isBarycentricCoordInBounds(bc))
				{
					// TODO: depth check
					int depthBufferId = y * resolution.x + x;
					float z = getZAtCoordinate(bc, tri);
					

					if (z > depthbuffer[depthBufferId].position.z)
					{
						depthbuffer[depthBufferId].position = tri.p0 * bc.x + tri.p1 * bc.y + tri.p2 * bc.z;
						depthbuffer[depthBufferId].color = tri.c0 * bc.x + tri.c1 * bc.y + tri.c2 * bc.z;
						depthbuffer[depthBufferId].normal = tri.n0 * bc.x + tri.n1 * bc.y + tri.n2 * bc.z;

						// depth test
						//depthbuffer[depthBufferId].color = glm::normalize(vec3(-z,-z,-z));
					}

					// normal test 
					//if (depthbuffer[depthBufferId].normal.z > 0)
					//	depthbuffer[depthBufferId].color = depthbuffer[depthBufferId].normal;
					//else
					//	depthbuffer[depthBufferId].color = depthbuffer[depthBufferId].normal;

					// test
					//fragment frag;
					//frag.color = vec3(1,1,1);
					//depthbuffer[depthBufferId] = frag;
				}
			}
		}
	}
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, light* lights, int numlights)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y)
  {

  }
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer)
{

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y)
  {
    framebuffer[index] = depthbuffer[index].color;
  }
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(camera* cam, uchar4* PBOpos, glm::vec2 resolution, float frame, 
					   float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, 
					   float* nbo, int nbosize, light* lights, int lightsize)
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
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag);

  //------------------------------
  //memory stuff
  //------------------------------
  allocateDeviceMemory(vbo, vbosize, cbo, cbosize, ibo, ibosize, nbo, nbosize, lights, lightsize);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------
  // turn table
  mat4 modelMatrix(1);
  float d = (int)frame % 361;
  modelMatrix = glm::rotate(modelMatrix, -d, vec3(0,1,0));

  // retrieve camera information
  mat4 viewMatrix = cam->view;
  mat4 projectionMatrix = cam->projection;
  mat4 mvp = projectionMatrix * viewMatrix * modelMatrix;
  mat4 mvInvT = transpose(inverse(viewMatrix * modelMatrix));
  float zFar = cam->zFar;
  float zNear = cam->zNear;
  vec2 reso = cam->resolution;
  
  // launch vertex shader kernel
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_nbo, nbosize, mvp, mvInvT, reso, zNear, zFar);
  cudaDeviceSynchronize();

#if DEBUG == 1
  printVAO(device_vbo, vbosize);
#endif

  //checkCUDAErrorWithLine("vertex shader kernel failed");
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, device_nbo, nbosize, primitives);
  cudaDeviceSynchronize();

#if DEBUG == 1
  printVAO(device_nbo, nbosize);
#endif

  //checkCUDAErrorWithLine("primitive assembly kernel failed");
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);
  cudaDeviceSynchronize();
  //checkCUDAErrorWithLine("rasterization kernel failed");
  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, device_lights, lightsize);
  cudaDeviceSynchronize();
  //checkCUDAErrorWithLine("fragment shader kernel failed");
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
  //checkCUDAErrorWithLine("render kernel failed");
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);
  //checkCUDAErrorWithLine("send image to pbo kernel failed");
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

// debug
void printVAO(float* device_vao, int size)
{
	float* vao = NULL;
	vao = new float[size];
	cudaMemcpy(vao, device_vao, size*sizeof(float), cudaMemcpyDeviceToHost);

	printf ("PrintVAO invoked\n");
	for (int i = 0 ; i < size ; ++i)
	{
		float v1 = vao[i]; i++;
		float v2 = vao[i]; i++;
		float v3 = vao[i];

		printf ("%f,%f,%f\n", v1, v2, v3);
	}


	delete[] vao;
}

void allocateDeviceMemory( float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, 
						   float* nbo, int nbosize, light* lights, int lightsize)
{
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

  device_lights = NULL;

}
