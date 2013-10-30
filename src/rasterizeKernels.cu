// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include "rasterizeKernels.h"
#include "rasterizeTools.h"


glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
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


__host__ __device__ float getDepthFromDepthbuffer(int x, int y, fragment* depthbuffer, glm::vec2 resolution){
	if(x<resolution.x && y<resolution.y){
		int index = (y*resolution.x) + x;
		return depthbuffer[index].position.z;
	}else{
		return 0;
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

__global__ void vertexShadeKernel(float* vbo, int vbosize, uniforms u_variables, pipelineOpts opts){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<vbosize/3){
		glm::vec4 vertexPos = glm::vec4(vbo[index*3+0],vbo[index*3+1],vbo[index*3+2],1.0);
		vertexPos = u_variables.perspectiveTransform*u_variables.viewTransform*u_variables.modelTransform*vertexPos;

		//Perspective division
		vertexPos.x /= vertexPos.w;
		vertexPos.y /= vertexPos.w;
		vertexPos.z /= vertexPos.w;

		vbo[index*3+0] = vertexPos.x;
		vbo[index*3+1] = vertexPos.y;
		vbo[index*3+2] = vertexPos.z;
	}
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, 
										int* ibo, int ibosize, triangle* primitives, 
										uniforms u_variables, pipelineOpts opts)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int primitivesCount = ibosize/3;
	if(index<primitivesCount){
		//3 floats per vert, 3 verts per triangle
		triangle primative;
		//Load verticies and do perspective division while we're at it.

		//Vertex 0
		//primitive index*3 + vert number
		int vertIndex = ibo[index*3+0];
		//For now just mod 3, TODO: Improve cbo functionality
		int colorIndex = 3*(vertIndex % 3);//3 floats per color
		vertIndex *= 3;//3 floats per vert

		primative.p0 = glm::vec3(vbo[vertIndex+0],vbo[vertIndex+1],vbo[vertIndex+2]);
		primative.c0 = glm::vec3(cbo[colorIndex+0],cbo[colorIndex+1],cbo[colorIndex+2]);

		//Vertex 1
		vertIndex = ibo[index*3+1];
		colorIndex = 3*(vertIndex % 3);//3 floats per color
		vertIndex *= 3;//3 floats per vert

		primative.p1 = glm::vec3(vbo[vertIndex+0],vbo[vertIndex+1],vbo[vertIndex+2]);
		primative.c1 = glm::vec3(cbo[colorIndex+0],cbo[colorIndex+1],cbo[colorIndex+2]);

		//Vertex 2
		vertIndex = ibo[index*3+2];
		colorIndex = 3*(vertIndex % 3);//3 floats per color
		vertIndex *= 3;//3 floats per vert

		primative.p2 = glm::vec3(vbo[vertIndex+0],vbo[vertIndex+1],vbo[vertIndex+2]);
		primative.c2 = glm::vec3(cbo[colorIndex+0],cbo[colorIndex+1],cbo[colorIndex+2]);

		//Write back primative
		primitives[index] = primative;
	}
}

//TODO: Do this a lot more efficiently and in parallel
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, 
									glm::vec2 resolution, uniforms u_variables, pipelineOpts opts)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<primitivesCount){
		//For each primative
		//Load triangle localy
		triangle tri = primitives[index];

		//Compute surface normal.
		glm::vec3 normal = glm::normalize(glm::cross(tri.p1-tri.p0, tri.p2-tri.p0));

		transformTriToScreenSpace(tri, resolution);

		//AABB for triangle
		glm::vec3 minPoint;
		glm::vec3 maxPoint;
		getAABBForTriangle(tri, minPoint, maxPoint);


		//Compute pixel range
		int minX = glm::floor(minPoint.x);
		int maxX = glm::ceil(maxPoint.x);
		int minY = glm::floor(minPoint.y);
		int maxY = glm::ceil(maxPoint.y);


		fragment frag;
		//Flat shading for now
		frag.normal = normal;

		//TODO: Do something more efficient than this
		for(int x = minX; x <= maxX; ++x)
		{
			for(int y = minY; y <= maxY; ++y)
			{
				glm::vec3 bCoords = calculateBarycentricCoordinate(tri, glm::vec2(x,y));
				if(isBarycentricCoordInBounds(bCoords))
				{
					frag.color = tri.c0*bCoords.x+tri.c1*bCoords.y+tri.c2*bCoords.z;
					frag.position = tri.p0*bCoords.x+tri.p1*bCoords.y+tri.p2*bCoords.z;

					//Handle race conditions in a lousy way
					while(frag.position.z < getDepthFromDepthbuffer(x,y,depthbuffer,resolution))
					{
						writeToDepthbuffer(x,y,frag, depthbuffer,resolution);
					}
				}
			}
		}

	}
}

__host__ __device__ void depthFSImpl(fragment* depthbuffer, int index,  uniforms u_variables, pipelineOpts opts)
{
	float depth = depthbuffer[index].position.z;
	if(depth < MAX_DEPTH)
		depthbuffer[index].color = glm::vec3(1.0f-depth); 
}


__host__ __device__ void ambientFSImpl(fragment* depthbuffer, int index,  uniforms u_variables, pipelineOpts opts)
{
	//Do nothing. Interpolated color is assumed to be right
}

__host__ __device__ void normalFSImpl(fragment* depthbuffer, int index,  uniforms u_variables, pipelineOpts opts)
{	
	glm::vec3 color = depthbuffer[index].normal;
	color.x = abs(color.x);
	color.y = abs(color.y);
	color.z = abs(color.z);
	depthbuffer[index].color = color; 

}


__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, 
									uniforms u_variables, pipelineOpts opts)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if(x<=resolution.x && y<=resolution.y){
		switch(opts.fShaderProgram)
		{
		case DEPTH_BUFFER:
			depthFSImpl(depthbuffer, index, u_variables, opts);
			break;
		case AMBIENT_LIGHTING:
			ambientFSImpl(depthbuffer, index, u_variables, opts);
			break;
		case NORMAL_SHADING:
			normalFSImpl(depthbuffer, index, u_variables, opts);
			break;
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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, 
					   float* cbo, int cbosize, int* ibo, int ibosize, uniforms u_variables, pipelineOpts opts)
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
	frag.color = glm::vec3(0.0f);
	frag.normal = glm::vec3(0.0f);
	frag.position = glm::vec3(0.0f,0.0f,MAX_DEPTH);
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

	tileSize = 32;
	int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

	//------------------------------
	//vertex shader
	//------------------------------
	vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, u_variables, opts);

	cudaDeviceSynchronize();
	//------------------------------
	//primitive assembly
	//------------------------------
	primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
	primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives, u_variables, opts);

	cudaDeviceSynchronize();
	//------------------------------
	//rasterization
	//------------------------------

	rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution, u_variables, opts);



	cudaDeviceSynchronize();
	//------------------------------
	//fragment shader
	//------------------------------
	fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, u_variables, opts);

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

