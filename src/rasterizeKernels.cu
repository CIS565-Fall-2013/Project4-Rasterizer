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
float* device_original_vbo;
float* device_cbo;
int* device_ibo;
triangle* primitives;

struct terminated
{
	__host__ __device__
		bool operator()(const triangle t)
	{
		return !t.isVisible;
	}
};

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
	  f.position.z=100000000.0f;
	  f.color=glm::vec3(1,1,1);
      buffer[index] = f;
    }
}

__global__ void converge(glm::vec2 resolution, fragment* buffer, fragment* antialiasBuffer){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if(x<=resolution.x && y<=resolution.y){
		glm::vec3 c(0,0,0);
		for(int i=x*2;i<x*2+2;i++)
		for(int j=y*2;j<y*2+2;j++)
			c+=antialiasBuffer[i+j*(int)resolution.x*2].color;
		buffer[index].color=c*0.25f;

		//buffer[index].color=antialiasBuffer[x+y*(int)resolution.x].color;
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
__global__ void vertexShadeKernel(float* vbo, int vbosize,glm::mat4 projection, cameraInfo cam,glm::mat4 modelTransform){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  glm::vec4 thePos(vbo[index*3],vbo[index*3+1],vbo[index*3+2],1.0f);
	  thePos=(projection)*modelTransform*thePos;
	  thePos.x/=thePos.z;
	  thePos.y/=thePos.z;
	  thePos/=glm::tan(cam.fovy*3.1415926f/180.0f);
	  vbo[index*3]=thePos.x;
	  vbo[index*3+1]=thePos.y;
	  vbo[index*3+2]=-thePos.z;

  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, float* nbo, int nbosize, int* ibo, int ibosize, triangle* primitives, float* origin_vbo, glm::mat4 projection){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  int iboidx0=ibo[index*3];
	  int iboidx1=ibo[index*3+1];
	  int iboidx2=ibo[index*3+2];
	  primitives[index].p0=glm::vec3(vbo[3*iboidx0],vbo[3*iboidx0+1],vbo[3*iboidx0+2]);
	  primitives[index].p1=glm::vec3(vbo[3*iboidx1],vbo[3*iboidx1+1],vbo[3*iboidx1+2]);
	  primitives[index].p2=glm::vec3(vbo[3*iboidx2],vbo[3*iboidx2+1],vbo[3*iboidx2+2]);
	  primitives[index].c0=glm::vec3(cbo[0],cbo[1],cbo[2]);
	  primitives[index].c1=glm::vec3(cbo[3],cbo[4],cbo[5]);
	  primitives[index].c2=glm::vec3(cbo[6],cbo[7],cbo[8]);
	  glm::vec3 p0=glm::vec3(vbo[3*iboidx0],vbo[3*iboidx0+1],vbo[3*iboidx0+2]);
	  glm::vec3 p1=glm::vec3(vbo[3*iboidx1],vbo[3*iboidx1+1],vbo[3*iboidx1+2]);
	  glm::vec3 p2=glm::vec3(vbo[3*iboidx2],vbo[3*iboidx2+1],vbo[3*iboidx2+2]);

	  glm::vec3 N=glm::normalize(glm::cross(p1-p0,p2-p1));
	  primitives[index].n0=glm::vec3(nbo[3*iboidx0],nbo[3*iboidx0+1],nbo[3*iboidx0+2]);
	  primitives[index].n1=glm::vec3(nbo[3*iboidx1],nbo[3*iboidx1+1],nbo[3*iboidx1+2]);
	  primitives[index].n2=glm::vec3(nbo[3*iboidx2],nbo[3*iboidx2+1],nbo[3*iboidx2+2]);;

	  glm::vec3 camDir(projection[0][2],projection[1][2],projection[2][2]);
	  float dt=glm::dot(camDir,N);
	  primitives[index].isVisible=(dt>0);
  }
}

//TODO: Implement a rasterization method, such as scanline.

__device__ int getMin(int a, int b)
{
	return (a<=b)?a:b;
}
__device__ int getMax(int a, int b)
{
	return (a>=b)?a:b;
}
__device__ int getMin(int a, int b, int c)
{
	if(a<=b && a<=c) return a;
	if(b<=a && b<=c) return b;
	return c;
}

__device__ int getMax(int a, int b, int c)
{
	if(a>=b && a>=c) return a;
	if(b>=a && b>=c) return b;
	return c;
}
__device__ bool isValid(float x, float y)
{
	return x>=-1.0f && x<1.0f && y>=-1.0f && y<1.0f;
}

__device__ int inBoundary(int x, int y , glm::vec2 resolution)
{
	return x>=0 && y>=0 && x<resolution.x && y<resolution.y;
}


///CALCULATE the det of the 2*2 matrix
/// | a c |
/// | b d |
///=ad-bc
///////////////////////
__host__ __device__ float detMatrix22(float a, float b, float c ,float d)
{
	return a*d-b*c;
}

__device__ glm::vec2 getInterpolateCoeff(int x, int y, float x1,float y1,float x2, float y2, float x3, float y3)
{
	glm::vec3 v21(x2-x1,y2-y1,0);
	glm::vec3 v31(x3-x1,y3-y1,0);
	glm::vec3 tbd(x-x1,y-y1,0);
	float theBase=detMatrix22(v21.y,v21.x,v31.y,v31.x);
	if(abs(theBase)<0.0001f)
	{
		if(abs(v21.x)>0.0001f) return glm::vec2(tbd.x/v21.x,0);
		if(abs(v21.y)>0.0001f) return glm::vec2(tbd.y/v21.y,0);
		if(abs(v31.x)>0.0001f) return glm::vec2(0,tbd.x/v31.x);
		if(abs(v31.y)>0.0001f) return glm::vec2(0,tbd.y/v31.y);
		return glm::vec2(-1,-1);
	}
	float up1=detMatrix22(v21.y,v21.x,tbd.y,tbd.x);
	float up2=-detMatrix22(v31.y,v31.x,tbd.y,tbd.x);
	float t1=up2/theBase;
	float t2=up1/theBase;
	return glm::vec2(t1,t2);
}
__device__ float getZ(glm::vec2 interpCoeff, float z1, float z2, float z3)
{
	
	return z1+interpCoeff.x*(z2-z1)+interpCoeff.y*(z3-z1);
}
__device__ glm::vec3 getC(glm::vec2 interpCoeff, glm::vec3 c1, glm::vec3 c2, glm::vec3 c3)
{
	return c1+interpCoeff.x*(c2-c1)+interpCoeff.y*(c3-c1);
}
__device__ glm::vec3 getN(glm::vec2 interpCoeff, glm::vec3 n1, glm::vec3 n2, glm::vec3 n3)
{
	return n1+interpCoeff.x*(n2-n1)+interpCoeff.y*(n3-n1);
}
__device__ float clampit(float x, float min, float max)
{
	if(x<min) return min;
	if(x>max) return max;
	return x;
}
__device__ glm::vec2 getStartEndPoint(int x, int x1, int y1, int x2, int y2, int x3, int y3,glm::vec2 resolution)
{
	bool intersect1=((x-x1)*(x-x2)<=0);
	bool intersect2=((x-x2)*(x-x3)<=0);
	bool intersect3=((x-x3)*(x-x1)<=0);
	float ip1, ip2, ip3;
	ip1=(x1==x2)?-99999999:(y1+(y2-y1)*(float)((float)(x-x1)/(float)(x2-x1)));
	ip2=(x2==x3)?-99999999:(y2+(y3-y2)*(float)((float)(x-x2)/(float)(x3-x2)));
	ip3=(x1==x3)?-99999999:(y3+(y1-y3)*(float)((float)(x-x3)/(float)(x1-x3)));
	float tmp;
	float start, end;
	if(ip1<-9999999 && ip2<-9999999 && ip3<-9999999 )
	{		
		start=clampit((float)getMin(y1,y2,y3),0,resolution.y);
		end=clampit((float)getMax(y1,y2,y3),0,resolution.y);
	}
	else if((!intersect3) || ip3<-9999999)
	{
		start=clampit(ip1,0,resolution.y);
		end=clampit(ip2,0,resolution.y);
		if(start>end) {tmp=start;start=end;end=tmp;}
	}
	else if((!intersect1) || ip1<-9999999)
	{
		start=clampit(ip2,0,resolution.y);
		end=clampit(ip3,0,resolution.y);
		if(start>end) {tmp=start;start=end;end=tmp;}
	}
	else if((!intersect2) || ip2<-9999999)
	{
		start=clampit(ip3,0,resolution.y);
		end=clampit(ip1,0,resolution.y);
		if(start>end) {tmp=start;start=end;end=tmp;}
	}
	else if(intersect1 && intersect2 && intersect3)
	{
		start=getMin((float)ip1,(float)ip2,(float)ip3);
		end=getMax((float)ip1,(float)ip2,(float)ip3);
	}	
	else
	{
		start=0;
		end=resolution.y;
	}
	return glm::vec2(start,end);
}

__global__ void pixel_level_rasterization(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if(!inBoundary(x,y,resolution)) return;

	float posx1f,posy1f, posx2f, posy2f, posx3f, posy3f, posz1f,posz2f,posz3f;
	float mindist=10000;
	for(int i=0;i<primitivesCount;i++)
	{
		triangle current=primitives[i];

		posx1f=resolution.x*(current.p0.x+1.0f)/2.0f;
		posy1f=resolution.y*(current.p0.y+1.0f)/2.0f;
		posz1f=current.p0.z;
		posx2f=resolution.x*(current.p1.x+1.0f)/2.0f;
		posy2f=resolution.y*(current.p1.y+1.0f)/2.0f;
		posz2f=current.p1.z;
		posx3f=resolution.x*(current.p2.x+1.0f)/2.0f;
		posy3f=resolution.y*(current.p2.y+1.0f)/2.0f;
		posz3f=current.p2.z;

		glm::vec3 c1=current.c0;
		glm::vec3 c2=current.c1;
		glm::vec3 c3=current.c2;

		glm::vec3 n1=current.n0;
		glm::vec3 n2=current.n1;
		glm::vec3 n3=current.n2;


		//int targidx=i+j*(int)resolution.x;
		glm::vec2 interpcoeff=getInterpolateCoeff(x,y,posx1f,posy1f,posx2f,posy2f,posx3f,posy3f);
		if (interpcoeff.x<-0.00f || interpcoeff.y<-0.00f || interpcoeff.x+interpcoeff.y>1.00f) continue;
		float d=getZ(interpcoeff,posz1f,posz2f,posz3f);
		glm::vec3 c=getC(interpcoeff,c1,c2,c3);
		glm::vec3 n=getN(interpcoeff,n1,n2,n3);
		//float d=(posz1f,posz2f,posz3f)/3.0f;
		if(d<depthbuffer[index].position.z)
		{
			depthbuffer[index].position.z=d;
			depthbuffer[index].color=c;
			depthbuffer[index].normal=n;
		}

		
	}
}


__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int posx1,posy1, posx2, posy2, posx3, posy3;
  float posx1f,posy1f, posx2f, posy2f, posx3f, posy3f, posz1f,posz2f,posz3f;
  if(index<primitivesCount){

	//glm::vec3 com=(primitives[index].p0+primitives[index].p1+primitives[index].p2)/3.0f;
	triangle current=primitives[index];
	posx1=(int)(resolution.x*(current.p0.x+1.0f)/2.0f);
	posy1=(int)(resolution.y*(current.p0.y+1.0f)/2.0f);
	posx2=(int)(resolution.x*(current.p1.x+1.0f)/2.0f);
	posy2=(int)(resolution.y*(current.p1.y+1.0f)/2.0f);
	posx3=(int)(resolution.x*(current.p2.x+1.0f)/2.0f);
	posy3=(int)(resolution.y*(current.p2.y+1.0f)/2.0f);

	posx1f=resolution.x*(current.p0.x+1.0f)/2.0f;
	posy1f=resolution.y*(current.p0.y+1.0f)/2.0f;
	posz1f=current.p0.z;
	posx2f=resolution.x*(current.p1.x+1.0f)/2.0f;
	posy2f=resolution.y*(current.p1.y+1.0f)/2.0f;
	posz2f=current.p1.z;
	posx3f=resolution.x*(current.p2.x+1.0f)/2.0f;
	posy3f=resolution.y*(current.p2.y+1.0f)/2.0f;
	posz3f=current.p2.z;

	glm::vec3 c1=current.c0;
	glm::vec3 c2=current.c1;
	glm::vec3 c3=current.c2;

	glm::vec3 n1=current.n0;
	glm::vec3 n2=current.n1;
	glm::vec3 n3=current.n2;

	int xmin=getMin(posx1,posx2,posx3);
	int xmax=getMax(posx1, posx2,posx3);
	int ymin=getMin(posy1,posy2,posy3);
	int ymax=getMax(posy1,posy2,posy3);
	if(!(inBoundary(xmin,0,resolution)||inBoundary(xmax,0,resolution))) return;
	for(int i=xmin;i<=xmax;i++)
	{
		glm::vec2 termi;//=getStartEndPoint(i,posx1,posy1,posx2,posy2,posx3,posy3,resolution);
		termi.x=ymin;termi.y=ymax;
		for(int j=(int)termi.x;j<=termi.y+0.001f;j++)
		{
			if(!inBoundary(i,j,resolution)) continue;
			int targidx=i+j*(int)resolution.x;
			glm::vec2 interpcoeff=getInterpolateCoeff(i,j,posx1f,posy1f,posx2f,posy2f,posx3f,posy3f);
			if (interpcoeff.x<-0.0f || interpcoeff.y<-0.0f || interpcoeff.x+interpcoeff.y>1.0f) continue;
			float d=getZ(interpcoeff,posz1f,posz2f,posz3f);
			glm::vec3 c=getC(interpcoeff,c1,c2,c3);
			glm::vec3 n=getN(interpcoeff,n1,n2,n3);
			//float d=com.z;
			if(d<depthbuffer[targidx].position.z)
			{
				depthbuffer[targidx].position.z=d;
				depthbuffer[targidx].color=c;
				depthbuffer[targidx].normal=n;
			}
			
		}
		
	}

  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, float minz, float maxz){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  
	  ///////NORMAL TEST
	  //depthbuffer[index].color=(depthbuffer[index].position.z>1000.0f)?glm::vec3(1,1,1):((depthbuffer[index].normal+glm::vec3(1.0f))*0.5f); 

	  ///////DEPTH TEST
	  //depthbuffer[index].color=(depthbuffer[index].position.z>998.0f)?glm::vec3(0,0,0):glm::vec3(1.0f)*(maxz-depthbuffer[index].position.z)/(maxz-minz);

	  //////ILLUMINATION
	  float yratio=y/resolution.y;
	  depthbuffer[index].color=(depthbuffer[index].position.z>998.0f)?glm::vec3(0.5f*yratio,0.3f*(1-yratio),0.8f):glm::vec3(238,201,25)*(1/255.0f)*(0.1f+//Ambient Light
		  glm::clamp(glm::dot(glm::vec3(0,-1,0),depthbuffer[index].normal),0.0f,1.0f)+
		  glm::clamp(glm::dot(glm::vec3(0,1,0),depthbuffer[index].normal),0.0f,1.0f));
	  //depthbuffer[index].color=(depthbuffer[index].position.z>998.0f)?glm::vec3(0,0,0):depthbuffer[index].color*(0.1f+//Ambient Light
		 // glm::clamp(glm::dot(glm::vec3(0,-1,0),depthbuffer[index].normal),0.0f,1.0f)+
		 // glm::clamp(glm::dot(glm::vec3(0,1,0),depthbuffer[index].normal),0.0f,1.0f));
	  //depthbuffer[index].color=(depthbuffer[index].position.z>998.0f)?glm::vec3(0,0,0):glm::vec3(238,201,25)*(1/255.0f)*(
		 // glm::clamp(glm::dot(glm::vec3(0,-1,0),depthbuffer[index].normal),0.0f,1.0f));


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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, float* nbo, int nbosize,int* ibo, int ibosize,glm::mat4 projection, cameraInfo camInfo,glm::mat4 modelTransform){

  // set up crucial magic
  int tileSize = 16;
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

  device_ibo = NULL;
  cudaMalloc((void**)&device_ibo, ibosize*sizeof(int));
  cudaMemcpy( device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice);

  device_vbo = NULL;
  cudaMalloc((void**)&device_vbo, vbosize*sizeof(float));
  cudaMemcpy( device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_nbo = NULL;
  cudaMalloc((void**)&device_nbo, nbosize*sizeof(float));
  cudaMemcpy( device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_original_vbo = NULL;
  cudaMalloc((void**)&device_original_vbo, vbosize*sizeof(float));
  cudaMemcpy( device_original_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------

  //cudaMat4 cudaprojection=utilityCore::glmMat4ToCudaMat4(projection);
  //utilityCore::printCudaMat4(cudaprojection);
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize,projection,camInfo, modelTransform);

  //float* localvbo=new float[vbosize];
  //cudaMemcpy(localvbo,device_vbo,vbosize*sizeof(float),cudaMemcpyDeviceToHost);

  //if(frame<=0.5)
  //{
	 // for(int i=0;i<vbosize/3;i++)
	 // {
		//  printf("%f , %f , %f\n",localvbo[i*3],localvbo[i*3+1],localvbo[i*3+2]);
	 // }
	 // printf("------------------------------\n");
  //}
  //delete localvbo;

  cudaDeviceSynchronize();
  checkCUDAError("Kernel failed 1!");
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_nbo,nbosize, device_ibo, ibosize, primitives, device_original_vbo,projection);

  cudaDeviceSynchronize();
  checkCUDAError("Kernel failed 2!");
  //------------------------------
  //rasterization
  //------------------------------

  //if(frame<0.5f)
  //{
	 // cudaMat4 cudaprojection=utilityCore::glmMat4ToCudaMat4(projection);
	 // utilityCore::printCudaMat4(cudaprojection);

	 // printf("primitives: %d\n",ibosize/3);
	 // triangle* localPrimitives=new triangle[ibosize/3];
	 // cudaMemcpy(localPrimitives,primitives,ibosize/3*sizeof(triangle),cudaMemcpyDeviceToHost);
	 // for(int i=0;i<ibosize/3;i++)
	 // {
		//  localPrimitives[i].p0.x=800*(localPrimitives[i].p0.x/2.0f+0.5f);
		//  localPrimitives[i].p0.y=800*(localPrimitives[i].p0.y/2.0f+0.5f);
		//  localPrimitives[i].p1.x=800*(localPrimitives[i].p1.x/2.0f+0.5f);
		//  localPrimitives[i].p1.y=800*(localPrimitives[i].p1.y/2.0f+0.5f);
		//  localPrimitives[i].p2.x=800*(localPrimitives[i].p2.x/2.0f+0.5f);
		//  localPrimitives[i].p2.y=800*(localPrimitives[i].p2.y/2.0f+0.5f);

		//  printf("Primitive %d : p1(%f %f %f), p2(%f %f %f), p3(%f %f %f)\n",i,localPrimitives[i].p0.x,localPrimitives[i].p0.y,localPrimitives[i].p0.z
		//	  ,localPrimitives[i].p1.x,localPrimitives[i].p1.y,localPrimitives[i].p1.z
		//	  ,localPrimitives[i].p2.x,localPrimitives[i].p2.y,localPrimitives[i].p2.z);

	 // }
	 // delete localPrimitives;
  //}

  ///////BACK CULLING////////////

  int primitivenum=ibosize/3;
#ifdef BACKCULLING
  thrust::device_ptr<triangle> iteratorStart(primitives);
  thrust::device_ptr<triangle> iteratorEnd = iteratorStart + primitivenum;
  iteratorEnd = thrust::remove_if(iteratorStart, iteratorEnd, terminated());
  primitivenum = (int)(iteratorEnd - iteratorStart);
#endif
  //////BACKCULLING END/////////////
#ifdef ANTIALIAS
  fragment* anti_alias_depthbuffer = NULL;
  glm::vec2 resolution2(resolution.x*2,resolution.y*2);
  dim3 fullBlocksPerGrid2((int)ceil(float(resolution2.x)/float(tileSize)), (int)ceil(float(resolution2.y)/float(tileSize)));
  dim3 threadsPerBlock2(tileSize,tileSize);
  cudaMalloc((void**)&anti_alias_depthbuffer, 4*(int)resolution.x*(int)resolution.y*sizeof(fragment));
  clearDepthBuffer<<<fullBlocksPerGrid2, threadsPerBlock2>>>(resolution2, anti_alias_depthbuffer,frag);
  cudaDeviceSynchronize();
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, primitivenum, anti_alias_depthbuffer, resolution2);
  
  
#else
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, primitivenum, depthbuffer, resolution);
#endif

  //pixel_level_rasterization<<<fullBlocksPerGrid, threadsPerBlock>>>(primitives, ibosize/3, depthbuffer, resolution);

  cudaDeviceSynchronize();
  checkCUDAError("Kernel failed 3!");
  //------------------------------
  //fragment shader
  //------------------------------

  float minz=100000;float maxz=0;
  minz=0.0f;
  maxz=1.0f;
  //fragment* tmpDbf=new fragment[(int)(resolution.x*resolution.y)];
  //cudaMemcpy(tmpDbf,depthbuffer,resolution.x*resolution.y*sizeof(fragment),cudaMemcpyDeviceToHost);
  //for(int i=0;i<resolution.x*resolution.y;i++)
  //{
	 // float tmp=tmpDbf[i].position.z;
	 // if(tmp<minz) minz=tmp;
	 // if(tmp>maxz && tmp<998) maxz=tmp;
  //}
  //delete tmpDbf;
  //printf("minz=%f, maxz=%f\n",minz,maxz);
#ifdef ANTIALIAS
  //printf("%d %d, %d %d\n",fullBlocksPerGrid.x,fullBlocksPerGrid.y, fullBlocksPerGrid2.x,fullBlocksPerGrid2.y);
  fragmentShadeKernel<<<fullBlocksPerGrid2, threadsPerBlock2>>>(anti_alias_depthbuffer, resolution2,minz,maxz);
  cudaDeviceSynchronize();
  converge<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, anti_alias_depthbuffer);
  cudaDeviceSynchronize();
  cudaFree(anti_alias_depthbuffer);
#else
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution,minz,maxz);
#endif
  cudaDeviceSynchronize();
    checkCUDAError("Kernel failed 4!");

  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
  
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

  cudaDeviceSynchronize();
    checkCUDAError("Kernel failed 5!");

  kernelCleanup();

  checkCUDAError("Kernel failed 6!");
}

void kernelCleanup(){
  cudaFree( primitives );
  cudaFree( device_vbo );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}

