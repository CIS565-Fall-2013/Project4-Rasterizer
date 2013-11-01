#ifndef CAMERA_H
#define CAMERA_H

#include "glm/glm.hpp"
#include <GL/glew.h>
#include <GL/glut.h>


class CameraController
{
public:
	CameraController(glm::vec3 cameraPosition, glm::vec3 lookAt, glm::vec2 res)
		:center(lookAt),
		 radius( glm::length(lookAt-cameraPosition) ),
		 resolution(res)
	{
		theta = atan2f(cameraPosition.z,cameraPosition.x);
		phi = atan2f(cameraPosition.y, radius);
	}

	glm::vec3 getCameraPosition()
	{
		float x = radius*sinf(theta)*sinf(phi);
		float z = radius*cosf(phi);
		float y = radius*cosf(theta)*sinf(phi);

		//return glm::vec3(0,0,radius);
		return glm::vec3(radius*cos(theta),y,radius*sin(theta));
	}

	glm::vec3 getLookAtPosition()
	{
		return center;
	}

	void rotate (glm::vec2 delta)
	{
		if ( abs(delta.x) > abs(delta.y))
		{
			theta-=delta.x/resolution.x;
		}
		else
		{
			phi-=2.0f*delta.y/resolution.y;
		}

		if ( phi>PI || phi<0.0f)
			phi = 0.0f;

		//if ( theta>2*PI || theta<0.0f)
		//	theta = 0.0f;

	}

	void zoom( int delta)
	{
		radius= max(0.2f,radius+2*delta/resolution.y);
	}

	void pan( glm::vec2 delta)
	{
		if( abs(delta.x) > abs(delta.y))
			center.x+= -0.0001f*delta.x;
		else
			center.y+= 0.0001f*delta.y;
	}
	
	void mouseClick(int button, int state, int x, int y)
	{
		currentMouseX = x;
		currentMouseY = y;
		currentMouseBtn = button;
	}

	void mouseMove(int x, int y)
	{
		if(currentMouseBtn == GLUT_LEFT_BUTTON)
		{
			rotate(glm::vec2(x-currentMouseX,y-currentMouseY));
		}
		else if (currentMouseBtn == GLUT_MIDDLE_BUTTON)
		{
			int yDelta = y-currentMouseY;
			zoom(yDelta);
			
		}
		else if (currentMouseBtn == GLUT_RIGHT_BUTTON)
		{
			pan(glm::vec2(x-currentMouseX,y-currentMouseY));
			
		}
	}

private:
	int currentMouseX;
	int currentMouseY;
	int currentMouseBtn;
	glm::vec3 center;
	float phi;
	float theta;
	float radius;
	glm::vec2 resolution;
};


#endif