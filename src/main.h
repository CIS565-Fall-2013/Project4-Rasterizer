// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef MAIN_H
#define MAIN_H

#ifdef __APPLE__
	#include <GL/glfw.h>
#else
	#include <GL/glew.h>
	#include <GL/glut.h>
#endif

#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <time.h>
#include "glslUtility.h"
#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "rasterizeKernels.h"
#include "utilities.h"
#include "ObjCore/objloader.h"
#include "camera.h"

#if CUDA_VERSION >= 5000
    #include <helper_cuda.h>
    #include <helper_cuda_gl.h>
    #define compat_getMaxGflopsDeviceId() gpuGetMaxGflopsDeviceId() 
#else
    #include <cutil_inline.h>
    #include <cutil_gl_inline.h>
    #define compat_getMaxGflopsDeviceId() cutGetMaxGflopsDeviceId()
#endif

using namespace std;

//-------------------------------
//------------GL STUFF-----------
//-------------------------------
int frame;
int fpstracker;
double seconds;
int fps = 0;
GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
const char *attributeLocations[] = { "Position", "Tex" };
GLuint pbo = (GLuint)NULL;
GLuint displayImage;
uchar4 *dptr;

obj* mesh;

float* vbo;
int vbosize;
float* cbo;
int cbosize;
int* ibo;
int ibosize;
float* nbo;
int nbosize;

float fovy = 40.0f;
float zNear = 0.5f;
float zFar = 5.0f;

glm::mat4 model;
glm::mat4 view;
glm::mat4 projection;
glm::vec3 cameraPosition(0.0,0.2,2.5);
glm::vec3 cameraLookAt(0.0,0.2,0.0);



//-------------------------------
//----------CUDA STUFF-----------
//-------------------------------

int width=1280; int height=960;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv);

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda();

#ifdef __APPLE__
	void display();
#else
	void display();
	void keyboard(unsigned char key, int x, int y);
	void mouseClick(int button, int state, int x, int y);
	void mouseMove(int x, int y);
#endif

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

#ifdef __APPLE__
	void init();
#else
	void init(int argc, char* argv[]);
#endif

void initPBO(GLuint* pbo);
void initCuda();
void initTextures();
void initVAO();
GLuint initShader(const char *vertexShaderPath, const char *fragmentShaderPath);

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda();
void deletePBO(GLuint* pbo);
void deleteTexture(GLuint* tex);
void shut_down(int return_code);

#endif