#ifndef _VARIABLES_H
#define _VARIABLES_H

#include<GL/freeglut.h>
//-------------------------------
//------------GL STUFF-----------
//-------------------------------
extern int frame;
extern int fpstracker;
extern double seconds;
extern int fps;
extern GLuint positionLocation;
extern GLuint texcoordsLocation;
extern const char *attributeLocations[];
extern GLuint pbo;
extern GLuint displayImage;
extern uchar4 *dptr;
extern HostStat statVal;
extern ObjModel* mesh;
//extern float* vbo;
//extern int vbosize;
//extern float* cbo;
//extern int cbosize;
//extern float* tbo;
//extern int tbosize;
//extern float* nbo;
//extern int nbosize;
//extern int* ibo;
//extern int* nibo;
//extern int* tibo;
//extern int ibosize;
extern Param param;

//CUDA Resources
extern cudaGraphicsResource* cudaPboRc;
extern size_t cudaRcSize;


extern int widt;
extern int height;

extern glm::vec4 lightPos;
extern glm::vec4 lightPos2;

#endif