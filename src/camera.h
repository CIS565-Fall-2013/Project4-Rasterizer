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
		glm::vec3 relCameraPos = cameraPosition - center;


		theta = atan2f(relCameraPos.z,relCameraPos.x);
		if (theta < 0.0f)
			theta = 2*PI+ theta;
		phi = acosf( relCameraPos.y/radius);
		originalPhi = phi;
		originalTheta  = theta;
		originalCenter = center;
		originalRadius = radius;

	}

	glm::vec3 getCameraPosition()
	{
		float x = radius*sinf(phi)*cosf(theta);
		float y = radius*cosf(phi);
		float z = radius*sinf(theta)*sinf(phi);

		return center + glm::vec3(x,y,z);
		
	}

	glm::vec3 getLookAtPosition()
	{
		return center;
	}

	void rotate (glm::vec2 delta)
	{
		const float epsilon = 0.0001f;
		if ( abs(delta.x) > abs(delta.y))
		{
			theta+= float(delta.x)/resolution.x;
		}
		else
		{
			phi+=2.0f*float(delta.y)/resolution.y;
			if (phi > PI -epsilon )
				phi = PI-epsilon;

			else if (phi<epsilon)
				phi = epsilon;
		}
	}

	void zoom( int delta)
	{
		radius= max(0.2f,radius+2*(float)delta/resolution.y);
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
		else if (currentMouseBtn == GLUT_RIGHT_BUTTON)
		{
			int yDelta = y-currentMouseY;
			zoom(yDelta);
			
		}
		else if (currentMouseBtn == GLUT_MIDDLE_BUTTON)
		{
			pan(glm::vec2(x-currentMouseX,y-currentMouseY));
			
		}
	}

	void reset()
	{
		theta = originalTheta;
		phi = originalPhi;
		radius = originalRadius;
		center = originalCenter;
	}

private:
	int currentMouseX;
	int currentMouseY;
	int currentMouseBtn;
	glm::vec3 center;
	glm::vec3 originalCenter;
	float originalTheta;
	float originalPhi;
	float originalRadius;
	float phi;
	float theta;
	float radius;
	glm::vec2 resolution;
};


#endif