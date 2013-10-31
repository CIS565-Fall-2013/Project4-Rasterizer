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
fragment*  depthbuffer;
int*       lock;
cudaMat4*  transform;
float*     device_vbo;
float*     device_cbo;
int*       device_ibo;
triangle*  primitives;

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
__global__ void vertexShadeKernel(float* vbo, int vbosize, const cudaMat4 transform){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index < vbosize/3){
    glm::vec3 newVertex = multiplyMV(transform, glm::vec4(vbo[3 * index], vbo[3 * index + 1], vbo[3 * index + 2], 1.0f));
    vbo[3 * index]     = newVertex.x;
    vbo[3 * index + 1] = newVertex.y;
    vbo[3 * index + 2] = newVertex.z;
  }
}

//TODO: Implement primitive assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index < primitivesCount){
    // The actual indices of vertices are stored in the index buffer object.
    const int* vertexIndex = &ibo[3 * index];
    primitives[index].p0 = glm::vec3(vbo[3 * vertexIndex[0]], vbo[3 * vertexIndex[0] +1], vbo[3 * vertexIndex[0] + 2]);
    primitives[index].p1 = glm::vec3(vbo[3 * vertexIndex[1]], vbo[3 * vertexIndex[1] +1], vbo[3 * vertexIndex[1] + 2]);
    primitives[index].p2 = glm::vec3(vbo[3 * vertexIndex[2]], vbo[3 * vertexIndex[2] +1], vbo[3 * vertexIndex[2] + 2]);

    // The size of cbo is nine, only needs to give the nine RGB values to the color in the triangle variable's color vector.
    primitives[index].c0 = glm::vec3(cbo[0], cbo[1], cbo[2]);
    primitives[index].c1 = glm::vec3(cbo[3], cbo[4], cbo[5]);
    primitives[index].c2 = glm::vec3(cbo[6], cbo[7], cbo[8]);
  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, glm::vec3 view, int* lock){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index < primitivesCount){
    // Initialize triangle and back face culling if the normal of z is point to the back
    triangle currentTriangle = primitives[index];
    glm::vec3 normal = glm::normalize(glm::cross(currentTriangle.p1 - currentTriangle.p0, currentTriangle.p2 - currentTriangle.p0));
    if (glm::dot(normal, view) < 0.0f )
      return;
	
    // Add min max vectors and integers for the bounds and project the min back to the screen coordinate
    glm::vec3 minPoint, maxPoint;
    int minX, minY, maxX, maxY;
    getAABBForTriangle(currentTriangle, minPoint, maxPoint);
    scale2screen(minPoint, minX, maxY, resolution);
    scale2screen(maxPoint, maxX, minY, resolution);

    if (minX > resolution.x - 1 || minY > resolution.y - 1 || minX < 0 || minY < 0 ) 
      return;

    // Clipping the points outside the  image inside.
    minX = minX < 0 ? 0 : minX;
    minY = minY < 0 ? 0 : minY;
    maxX = (maxX > resolution.x - 1) ? resolution.x - 1 : maxX;
    maxY = (maxY > resolution.y - 1) ? resolution.y - 1 : maxY;

    // Loop and rasterize the interpolated area across the current primitive.
    int idx;
	float depth = 0;
    for (int y = minY; y < maxY; ++ y) {
      for (int x = minX; x < maxX; ++ x) {
        idx = y * resolution.x + x; 
        glm::vec3 barycentricCoordinates = calculateBarycentricCoordinate(currentTriangle, screen2scale(x, y, resolution));

        // Determine whether the current pixel is within the bounds of the current primitive
        if (!isBarycentricCoordInBounds(barycentricCoordinates))
          continue;
        depth = getZAtCoordinate(barycentricCoordinates, currentTriangle);
        if (depth < - 1.0f || depth > 1.0f) 
          return;
        bool loopFlag = true;
        do {
          if (atomicCAS(&lock[idx], 0, 1) == 0) {
            //Depth Test
            if(depth > depthbuffer[idx].position.z) {
                depthbuffer[idx].position.x = screen2scale(x, y, resolution).x;
                depthbuffer[idx].position.y = screen2scale(x, y, resolution).y;
                depthbuffer[idx].position.z = depth;
                depthbuffer[idx].normal = normal;
                depthbuffer[idx].color = barycentricCoordinates.x * currentTriangle.c0 + barycentricCoordinates.y * currentTriangle.c1 + barycentricCoordinates.z * currentTriangle.c2;
		    }
		  loopFlag = false;
		  __threadfence();
          atomicExch(&(lock[idx]), 0);
		  }
        } while (loopFlag);
	  } // for x
	} // for y
  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, light light, bool depthFlag){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
    // Compute the vectors of light, view and H in Blinn-Phong lighting model, referring to http://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_shading_model
    glm::vec3 position = depthbuffer[index].position;
    glm::vec3 normal   = depthbuffer[index].normal;
	glm::vec3 L        = glm::normalize(light.position - position);
    glm::vec3 V        = glm::normalize(- position);
    glm::vec3 H        = glm::normalize(L + V);

    // Compute the diffusion and Blinn-Phong lighting
    float diffuse  = glm::max(glm::dot(L, normal), 0.0f);
	float specular = glm::max(glm::pow(glm::dot(H, normal), 10.0f), 0.0f);

    // Compute final color
    if (depthFlag)
      depthbuffer[index].color = depthbuffer[index].position.z * light.color;
	else
    depthbuffer[index].color *= 2.0f * (0.5f * diffuse + 0.5f * specular) * light.color;
  }
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer, bool antialiasing){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y){
    if (ANTIALIASING || antialiasing) {
      // Using super sampling and initialize with the central pixel, the corner flag indicates which corner the pixel is
      int sampleNum = 1;
	  glm::vec3 sampling = depthbuffer[index].color;

	  // Left side
	  sampleNum += (x > 0) ? 1 : 0;
      sampling += (x > 0) ? depthbuffer[index - 1].color : glm::vec3(0.0f);
	  // Upper side
	  sampleNum += (y > 0) ? 1 : 0;
	  sampling += (y > 0) ? depthbuffer[index - (int)resolution.x].color : glm::vec3(0.0f);
	  // Right side
	  sampleNum += (x < resolution.x - 1) ? 1 : 0;
	  sampling += (x < resolution.x - 1) ? depthbuffer[index + 1].color : glm::vec3(0.0f);
	  // Bottom side
	  sampleNum += (x < resolution.y - 1) ? 1 : 0;
	  sampling += (x < resolution.y - 1) ? depthbuffer[index + (int)resolution.x].color : glm::vec3(0.0f);

	  // Four corners
	  sampleNum += (sampleNum == 5) ? 4 : 0;
      sampling += (sampleNum == 9) ? (depthbuffer[index - (int)resolution.x - 1].color + depthbuffer[index - (int)resolution.x + 1].color + depthbuffer[index + (int)resolution.x - 1].color + depthbuffer[index + (int)resolution.x + 1].color) : glm::vec3(0.0f);
	  if (sampleNum == 9) {
        framebuffer[index] =  sampling / 9.0f; 
        return;
      }

	  // Two corners
	  sampleNum += (sampleNum == 4) ? 2 : 0;
      if (sampleNum == 6) {
		sampling += x == 0 ? depthbuffer[index - (int)resolution.x + 1].color + depthbuffer[index + (int)resolution.x + 1].color: glm::vec3(0.0f);
        sampling += y == 0 ? depthbuffer[index + (int)resolution.x - 1].color + depthbuffer[index + (int)resolution.x + 1].color: glm::vec3(0.0f);
        sampling += x == resolution.x ? depthbuffer[index - (int)resolution.x - 1].color + depthbuffer[index + (int)resolution.x - 1].color: glm::vec3(0.0f);
        sampling += y == resolution.y ? depthbuffer[index - (int)resolution.x - 1].color + depthbuffer[index - (int)resolution.x + 1].color: glm::vec3(0.0f);
		framebuffer[index] =  sampling / 6.0f;
        return;
	  }

	  // One corner
	  sampleNum += (sampleNum == 3) ? 1 : 0;
	  if((x == 0) && (y == 0)) {
        sampling += depthbuffer[index + (int)resolution.x + 1].color;
	  } else if ((x == resolution.x -1) && (y == 0)) {
        sampling += depthbuffer[index + (int)resolution.x - 1].color;
	  } else if ((y == resolution.y -1) && (x == 0)) {
        sampling += depthbuffer[index - (int)resolution.x + 1].color;
	  } else {
        sampling += depthbuffer[index - (int)resolution.x - 1].color;
	  }
      framebuffer[index] =  sampling / 4.0f;

	} else
    framebuffer[index] = depthbuffer[index].color;
  }
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, const cudaMat4* transform, glm::vec3 viewPort, bool antialiasing, bool depthFlag){
  
  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));

  // set up framebuffer
  framebuffer = NULL;
  cudaMalloc((void**)&framebuffer, (int)resolution.x*(int)resolution.y*sizeof(glm::vec3));
  
  // set up depthbuffer
  depthbuffer = NULL;
  cudaMalloc((void**)&depthbuffer, (int)resolution.x*(int)resolution.y*sizeof(fragment));

  // set up lock
  lock = NULL;
  cudaMalloc((void**)&lock, (int)resolution.x * (int)resolution.y * sizeof(int));

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

  // Index Buffer Object
  device_ibo = NULL;
  cudaMalloc((void**)&device_ibo, ibosize*sizeof(int));
  cudaMemcpy( device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice);

  // Vertex Buffer Object
  device_vbo = NULL;
  cudaMalloc((void**)&device_vbo, vbosize*sizeof(float));
  cudaMemcpy( device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  // Color Buffer Object
  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, *transform);

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives);

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution, viewPort, lock);

  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  light light;
  light.position = glm::vec3(0.0f);
  light.color = glm::vec3(1.0f);
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, light, depthFlag);

  cudaDeviceSynchronize();
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer, antialiasing);
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
  cudaFree( lock );
  cudaFree( transform );
}

