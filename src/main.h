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
#include "glm/gtc/matrix_transform.hpp"
#include "rasterizeKernels.h"
#include "rasterizeStructs.h"
#include "utilities.h"
#include "ObjCore/objloader.h"

#if CUDA_VERSION >= 5000
    #include <helper_cuda.h>
    #include <helper_cuda_gl.h>
    #define compat_getMaxGflopsDeviceId() gpuGetMaxGflopsDeviceId() 
#else
    #include <cutil_inline.h>
    #include <cutil_gl_inline.h>
    #define compat_getMaxGflopsDeviceId() cutGetMaxGflopsDeviceId()
#endif

#define RGBONLY 0 // used for determining whether the CBO should be equal to the size of VBO or just 9

using namespace std;
using glm::vec2;
using glm::vec3;
using glm::mat4;
using glm::perspective;
using glm::lookAt;
using glm::rotate;

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
float* nbo;
int nbosize;
int* ibo;
int ibosize;
light* lights;
int lightsize;
camera* cam;

// camera parameters
float fovy = 60.0f;
float zNear = 0.10f;
float zFar = 5.0f;
vec3 center = vec3(0);

// interactive camera via mouse movements
float zDistance = 1;
float alpha = 0.0f;
float beta = 0.0f;
bool altLmbDown = false;
vec2 lastMousePosition = vec2(0,0);

//-------------------------------
//----------CUDA STUFF-----------
//-------------------------------

int width=800; int height=800;

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
	void mouseMove(int x, int y);
	void mouseButton(int button, int state, int x, int y);
	void updateCamera();
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
void initCamera();
void initLights();
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