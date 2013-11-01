#ifndef _STATVAL_H
#define _STATVAL_H

#include "glm/glm.hpp"
#include "ObjCore/obj.h"

struct VertUniform
{
    glm::mat4 viewingMat;
    glm::mat4 projMat;
    glm::mat4 normalMat;
};

struct FragUniform
{

};

struct HostStat
{
    glm::vec4 initialEyePos;
    glm::vec3 eyePos;
    glm::vec3 eyeLook;
    glm::vec3 upDir;
    
    float FOV;
    float nearp;
    float farp;
    float aspect;
};

typedef struct{
    float* vbo;
    int vbosize;
    float* cbo;
    int cbosize;
    float* nbo;
    int nbosize;
    float* tbo;
    int tbosize;
    int* ibo;
    int* nibo;
    int* tibo;
    int ibosize;

    Group* groups;
    int numGroup;
}Param;

#endif