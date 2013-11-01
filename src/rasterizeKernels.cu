// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

#if RASTERIZE_MODE == 0
	#define NUM_BLOCKS(ibosize, resolution, tileSize) ceil(((float)ibosize/3)/((float)tileSize))
#if SCANLINE == 0
	#define RASTERIZE(x,y,z,w) rasterizeByPrimitive(x,y,z,w)
#else
	#define RASTERIZE(x,y,z,w) rasterizeByPrimitiveByScanLine(x,y,z,w)
#endif
#elif RASTERIZE_MODE == 1	
	#define NUM_BLOCKS(iboSize, resolution, tileSize) ceil(((float)resolution.x * resolution.y) / ((float)tileSize))
	#define RASTERIZE(x,y,z,w) rasterizeByFragment(x,y,z,w)
#endif

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_cbo;
int* device_ibo;
triangle* primitives;
triangle* primitivesBuffer;
int* backfaceculling;

#if BFCULL == 1
struct is_render{
	__host__ __device__ bool operator()(triangle t){
		return t.render;
	}
};
#endif

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
__global__ void vertexShadeKernel(float* vbo, int vbosize, glm::mat4 MV, glm::vec2 resolution, glm::mat4 proj){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
#if THREE == 0 
  int size = vbosize / 4;
#else 
  int size = vbosize / 3;
#endif

  if(index < size){

#if THREE == 0
	  glm::vec4 in_vertex = glm::vec4(vbo[4 * index], vbo[4 * index + 1], vbo[4 * index + 2], vbo[4 * index + 3]);
#else 
	  glm::vec4 in_vertex = glm::vec4(vbo[3 * index], vbo[3 * index + 1], vbo[3 * index + 2], 1.0f);
#endif

	  glm::vec4 P;

#if POINT_MODE == 1
	  P = proj * MV * in_vertex;
	  glm::vec3 P_ndc = glm::vec3(P.x / P.w, P.y / P.w, P.z / P.w);

	  P.x = resolution.x / 2 * P_ndc.x + resolution.x / 2;
	  P.y = resolution.y / 2 * P_ndc.y + resolution.y / 2;
	  P.z = P_ndc.z;
#else 
	  P = MV * in_vertex;
#endif
	  
#if THREE == 0
	  vbo[4 * index] = P.x;
	  vbo[4 * index + 1] = P.y;
	  vbo[4 * index + 2] = P.z;
	  vbo[4 * index + 3] = P.w;
#else
	  vbo[3 * index] = P.x;
	  vbo[3 * index + 1] = P.y
	  vbo[3 * index + 2] = P.z;
#endif

  }
}

// Primative assembly
#if POINT_MODE == 0
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitivesBuffer, int* backfaceculling){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index < primitivesCount){
	  int a = ibo[3 * index], b = ibo[3 * index + 1], c = ibo[3 * index + 2];
	
	  
	  // Create triangle
	  triangle t;
#if THREE == 0
	  t.p0 = glm::vec4(vbo[4 * a], vbo[4 * a + 1], vbo[4 * a + 2], vbo[4 * a + 3]);
	  t.p1 = glm::vec4(vbo[4 * b], vbo[4 * b + 1], vbo[4 * b + 2], vbo[4 * b + 3]);
	  t.p2 = glm::vec4(vbo[4 * c], vbo[4 * c + 1], vbo[4 * c + 2], vbo[4 * c + 3]);

	  t.normal = glm::normalize(glm::cross(glm::vec3(t.p1 - t.p0) , glm::vec3(t.p2 - t.p1)));
#else 
	  t.p0 = glm::vec3(vbo[3 * a], vbo[3 * a + 1], vbo[3 * a + 2]);
	  t.p1 = glm::vec3(vbo[3 * b], vbo[3 * b + 1], vbo[3 * b + 2]);
	  t.p2 = glm::vec3(vbo[3 * c], vbo[3 * c + 1], vbo[3 * c + 2]);

	  t.normal = glm::normalize(glm::cross(t.p1 - t.p0, t.p2 - t.p1));
#endif

#if COLOR == 0
	  t.c0 = t.c1 = t.c2 = glm::vec3(1.0f);
#elif COLOR == 1
	  t.c0 = glm::vec3(cbo[0], cbo[1], cbo[2]);
	  t.c1 = glm::vec3(cbo[3], cbo[4], cbo[5]);
	  t.c2 = glm::vec3(cbo[6], cbo[7], cbo[8]);
#elif COLOR == 2
	  t.c0 = t.normal;
	  t.c1 = t.normal;
	  t.c2 = t.normal;
#endif

	  // Cull Back-Face Triangle
	  bool cull = glm::dot(t.normal, -glm::vec3((t.p1 + t.p0 + t.p2) / 3)) < 0;

	  if(cull) t.render = false;
	  else t.render = true;
#if BFCULL == 1
	  backfaceculling[index] = cull ? 1 : 0; 
#endif

	  // Put triangle into primitives
	  primitivesBuffer[index] = t;
  }
}

void __global__ scatter(triangle* src, triangle* dest, int* scanArr, int numTri){
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if(k < numTri){
		if(src[k].render){
			int idx = scanArr[k];
			dest[idx] = src[k];
		}
	}
}

void cullBackFaceKernel(triangle* src, triangle* dest, int* backfaceculling, int primitiveCount, int& fullBlocksPerGrid, int threadsPerBlock, int& primCount){
	thrust::device_ptr<int> t_bfcull = thrust::device_pointer_cast(backfaceculling);
	thrust::device_ptr<triangle> t_primBuffer = thrust::device_pointer_cast(src);
	thrust::device_ptr<triangle> t_prim = thrust::device_pointer_cast(dest);

	// Compact
	thrust::copy_if(t_primBuffer, t_primBuffer + primitiveCount, t_prim, is_render());

	// Reduce
	primCount = thrust::reduce(t_bfcull, t_bfcull + primitiveCount, 0);

	// Recalculate kernel dimensions
	fullBlocksPerGrid = (int) ceil((float)primCount/threadsPerBlock);
}

__global__ void viewportTransformKernel(triangle* primitives, int primitivesCount, glm::vec2 resolution, glm::mat4 proj){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < primitivesCount){
		triangle t = primitives[index];

		// Transform to NDC
		glm::vec4 p0_ndc = proj * t.p0;
		glm::vec4 p1_ndc = proj * t.p1;
		glm::vec4 p2_ndc = proj * t.p2;

		p0_ndc = p0_ndc / p0_ndc.w;
		p1_ndc = p1_ndc / p1_ndc.w;
		p2_ndc = p2_ndc / p2_ndc.w;

		// Transform to Screen Coords
		t.p0.x = 1.0f / 2 * resolution.x * p0_ndc.x + (1.0f / 2 * resolution.x);
		t.p0.y = 1.0f / 2 * resolution.y * p0_ndc.y + (1.0f / 2 * resolution.y);
		t.p0.z = p0_ndc.z;

		t.p1.x = 1.0f / 2 * resolution.x * p1_ndc.x + (1.0f / 2 * resolution.x);
		t.p1.y = 1.0f / 2 * resolution.y * p1_ndc.y + (1.0f / 2 * resolution.y);
		t.p1.z = p1_ndc.z;

		t.p2.x = 1.0f / 2 * resolution.x * p2_ndc.x + (1.0f / 2 * resolution.x);
		t.p2.y = 1.0f / 2 * resolution.y * p2_ndc.y + (1.0f / 2 * resolution.y);
		t.p2.z = p2_ndc.z;

#if OOVIGNORE == 1
		glm::vec3 minpt, maxpt;
		getAABBForTriangle(t, minpt, maxpt);
		if(maxpt.x < 0 && maxpt.y < 0) t.render = false;
		if(minpt.x > resolution.x && minpt.y > resolution.y) t.render = false;
#endif
		
		primitives[index] = t;
	}
}
#endif

__global__ void rasterizePoints(float* vbo, float vbosize, fragment* depthbuffer, glm::vec2 resolution)
{
  int index = threadIdx.x + (blockDim.x * blockIdx.x);

#if THREE == 0
  int size = vbosize / 4;
#else 
  int size = vbosize / 3;
#endif

  if(index < size){

#if THREE == 0
	  glm::vec2 point = glm::vec2(vbo[4 * index], vbo[4 * index + 1]);
#else
	  glm::vec2 point = glm::vec2(vbo[3 * index], vbo[3 * index + 1]);
#endif

	  if(point.x >=0 && point.x < resolution.x && point.y >=0 && point.y < resolution.y)
	  {
		  fragment f;
		  f.position.x = point.x;
		  f.position.y = point.y;

#if THREE == 0
		  f.position.z = vbo[4 * index + 2];
#else 
		  f.position.z = vbo[3 * index + 2];
#endif

		  int bufferindex = int(point.x) + int(point.y) * resolution.x;
		  f.color = glm::vec3(1,1,1);
		  depthbuffer[bufferindex] = f;
	  }
  }
}

#if POINT_MODE == 0

#if RASTERIZE_MODE == 0

#if SCANLINE == 0

// Rasterize by Primitive
// Barycentric with non-geometric clipping
__device__ void rasterizeByPrimitive(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < primitivesCount){
		triangle t  = primitives[index];
		if(!t.render) return;
		glm::vec2 minpoint, maxpoint;
		getTightBoxForTriangle(t, minpoint, maxpoint, resolution);
		glm::vec3 bary_center;
		float z;
		fragment f;
		int count, idx;
		for(int i = minpoint.y; i < maxpoint.y; i++){
			count = 0; // reset line hit counter
			for(int j = minpoint.x; j < maxpoint.x && count < 2; j++){
				bary_center = calculateBarycentricCoordinate(t, glm::vec2(j,i));
				if(isBarycentricCoordInBounds(bary_center)){
					if(count == 0) ++count;
					z = getZAtCoordinate(bary_center, t);		
					idx = i * (int)resolution.x + j;

					// Critical Section
					//lock(&depthbuffer[idx].lock);
					if(z > depthbuffer[idx].position.z){
						depthbuffer[idx].color.x = t.c0.x * bary_center.x + t.c1.x * bary_center.y + t.c2.x * bary_center.z;
						depthbuffer[idx].color.y = t.c0.y * bary_center.x + t.c1.y * bary_center.y + t.c2.y * bary_center.z;
						depthbuffer[idx].color.z = t.c0.z * bary_center.x + t.c1.z * bary_center.y + t.c2.z * bary_center.z;
#if READ_NORMALS == 0
						depthbuffer[idx].normal = t.normal;
#endif
						depthbuffer[idx].position = glm::vec3(j,i,z);
					}
					//unlock(&depthbuffer[idx].lock);
				}else{
					if(count == 1) ++count;
				}
			}
		}
	}
}

#else

// Rasterize by Primitive
// Bresenham modified Scanline method with clipping

__device__ void rasterizeByPrimitiveByScanLine(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < primitivesCount){
		triangle t = primitives[index];
		glm::vec2 minpoint, maxpoint;
		getTightBoxForTriangle(t, minpoint, maxpoint, resolution);

		float m0 = epsilonCheck((t.p1 - t.p0).x, 0.0f) ? 100000 : (t.p1 - t.p0).y / (t.p1 - t.p0).x;
		float m1 = epsilonCheck((t.p2 - t.p1).x, 0.0f) ? 100000 : (t.p2 - t.p1).y / (t.p2 - t.p1).x;
		float m2 = epsilonCheck((t.p0 - t.p2).x, 0.0f) ? 100000 : (t.p0 - t.p2).y / (t.p0 - t.p2).x;

		glm::vec2 i0_min, i0_max, i1_min, i1_max, i2_min, i2_max;

		glm::vec3 bary_center;
		float z, m;
		fragment f;
		int count, idx;

		for(int i = minpoint.y; i <= maxpoint.y; i++){

		}
	}
}

#endif
#else
// Rasterization by pixel / fragment (one pixel per thread)
__device__ void rasterizeByFragment(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < resolution.x * resolution.y){
		int y = (index / resolution.x);
		int x = (index - resolution.x * y);
		glm::vec2 px_center = glm::vec2(x,y);
		triangle t;
		glm::vec3 bary_center;
		float z;

		for(int i = 0; i < primitivesCount; i++){
			t = primitives[i];
			bary_center = calculateBarycentricCoordinate(t, px_center);
			if(isBarycentricCoordInBounds(bary_center)){
				z = getZAtCoordinate(bary_center, t);
				if(z > depthbuffer[index].position.z){
					depthbuffer[index].color.x = t.c0.x * bary_center.x + t.c1.x * bary_center.y + t.c2.x * bary_center.z;
					depthbuffer[index].color.y = t.c0.y * bary_center.x + t.c1.y * bary_center.y + t.c2.y * bary_center.z;
					depthbuffer[index].color.z = t.c0.z * bary_center.x + t.c1.z * bary_center.y + t.c2.z * bary_center.z;
#if READ_NORMALS == 0
					depthbuffer[index].normal = t.normal;
#endif
					depthbuffer[index].position = glm::vec3(px_center, z);
				}
			}
		}
	}
}

#endif

// Rasterizer
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
  RASTERIZE(primitives, primitivesCount, depthbuffer, resolution);
}

// Fragment Shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, glm::vec4 light){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  fragment f = depthbuffer[index];
#if SHADING == 0
	  // FLAT SHADING
	  // no change to color
#elif SHADING == 1
	  // LAMBERT SHADING
	  f.color = f.color * glm::dot(f.normal, glm::vec3(light));
#endif
	  depthbuffer[index] = f;
  }
}
#endif

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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, glm::mat4 MV, glm::mat4 proj, glm::vec4 light){

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

#if BFCULL == 1
  primitivesBuffer = NULL;
  cudaMalloc((void**)&primitivesBuffer, (ibosize/3) * sizeof(triangle));

  backfaceculling = NULL;
  cudaMalloc((void**)&backfaceculling, (ibosize / 3) * sizeof(triangle));
#endif

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
  
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, MV, resolution, proj);
  cudaDeviceSynchronize();

  //------------------------------
  //primitive assembly
  //------------------------------

  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  
  int primitiveCount = ibosize/3;

#if POINT_MODE == 0
#if BFCULL == 1
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitivesBuffer, backfaceculling);
  cullBackFaceKernel(primitivesBuffer, primitives, backfaceculling, ibosize/3, primitiveBlocks, tileSize, primitiveCount);
#else 
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_vbo, cbosize, device_ibo, ibosize, primitives, backfaceculling);
  primitiveCount = ibosize/3;
#endif
  cudaDeviceSynchronize();

  //------------------------------
  //viewport transformation
  //------------------------------
  viewportTransformKernel<<<primitiveBlocks, tileSize>>>(primitives, primitiveCount, resolution, proj);
  cudaDeviceSynchronize();

  //------------------------------
  //rasterization
  //------------------------------

  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, primitiveCount, depthbuffer, resolution);
#else
  primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));
  rasterizePoints<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, depthbuffer, resolution);
#endif
  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, light);

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
  cudaFree( primitivesBuffer );
  cudaFree( backfaceculling );
}

