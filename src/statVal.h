#ifndef _STATVAL_H
#define _STATVAL_H

#include "glm/glm.hpp"

struct VertUniform
{
    glm::mat4 viewingMat;
    glm::mat4 projMat;
};

struct FragUniform
{

};

struct HostStat
{
    glm::vec3 eyePos;
    glm::vec3 eyeLook;
    glm::vec3 upDir;
    
    float FOV;
    float nearp;
    float farp;
    float aspect;
};

#endif