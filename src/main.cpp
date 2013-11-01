// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include "main.h"

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv){

  bool loadedScene = false;
  for(int i=1; i<argc; i++){
    string header; string data;
    istringstream liness(argv[i]);
    getline(liness, header, '='); getline(liness, data, '=');
    if(strcmp(header.c_str(), "mesh")==0){
      //renderScene = new scene(data);
      mesh = new obj();
      objLoader* loader = new objLoader(data, mesh);
      mesh->buildVBOs();
      delete loader;
      loadedScene = true;
    }
  }

  if(!loadedScene){
    cout << "Usage: mesh=[obj file]" << endl;
    return 0;
  }

  frame = 0;
  seconds = time (NULL);
  fpstracker = 0;

  // Launch CUDA/GL
  #ifdef __APPLE__
  // Needed in OSX to force use of OpenGL3.2 
  glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
  glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
  glfwOpenWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  init();
  #else
  init(argc, argv);
  #endif

  initCamera();
  initLights();
  initCuda();

  initVAO();
  initTextures();

  GLuint passthroughProgram;
  passthroughProgram = initShader("shaders/passthroughVS.glsl", "shaders/passthroughFS.glsl");

  glUseProgram(passthroughProgram);
  glActiveTexture(GL_TEXTURE0);

  #ifdef __APPLE__
    // send into GLFW main loop
    while(1){
      display();
      if (glfwGetKey(GLFW_KEY_ESC) == GLFW_PRESS || !glfwGetWindowParam( GLFW_OPENED )){
          kernelCleanup();
          cudaDeviceReset(); 
          exit(0);
      }
    }

    glfwTerminate();
  #else
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
	glutMouseFunc(mouseButton);
	glutMotionFunc(mouseMove);

    glutMainLoop();
  #endif
  kernelCleanup();
  return 0;
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda()
{
	//////////////////////
	// Timing cuda call //
	//////////////////////
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
	dptr=NULL;

	vbo = mesh->getVBO();
	vbosize = mesh->getVBOsize();

	nbo = mesh->getNBO();
	nbosize = mesh->getNBOsize();

#if RGBONLY == 1
	float newcbo[] = {0.0, 1.0, 0.0, 
					0.0, 0.0, 1.0, 
					1.0, 0.0, 0.0};
	cbo = newcbo;
	cbosize = 9;
#elif RGBONLY == 0
	vec3 defaultColor(0.5f, 0.5f, 0.5f);
	mesh->setColor(defaultColor);
	cbo = mesh->getCBO();
	cbosize = mesh->getCBOsize();
#endif

	ibo = mesh->getIBO();
	ibosize = mesh->getIBOsize();

	cudaGLMapBufferObject((void**)&dptr, pbo);

	updateCamera();

	cudaRasterizeCore(cam, dptr, glm::vec2(width, height), frame, vbo, vbosize, cbo, cbosize, ibo, ibosize, nbo, nbosize, lights, lightsize);
	cudaGLUnmapBufferObject(pbo);

	vbo = NULL;
	cbo = NULL;
	ibo = NULL;

	frame++;
	fpstracker++;

	//////////////////////
	// Timing cuda call //
	//////////////////////
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("runCuda runtime: %3.1f ms \n", time);
}

#ifdef __APPLE__

  void display(){
      runCuda();
      time_t seconds2 = time (NULL);

      if(seconds2-seconds >= 1){

        fps = fpstracker/(seconds2-seconds);
        fpstracker = 0;
        seconds = seconds2;

      }

      string title = "CIS565 Rasterizer | "+ utilityCore::convertIntToString((int)fps) + "FPS";

      glfwSetWindowTitle(title.c_str());


      glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);
      glBindTexture(GL_TEXTURE_2D, displayImage);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, 
            GL_RGBA, GL_UNSIGNED_BYTE, NULL);


      glClear(GL_COLOR_BUFFER_BIT);   

      // VAO, shader program, and texture already bound
      glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

      glfwSwapBuffers();
  }

#else

	void display()
	{
		runCuda();
		time_t seconds2 = time (NULL);

		if(seconds2-seconds >= 1){

			fps = fpstracker/(seconds2-seconds);
			fpstracker = 0;
			seconds = seconds2;
		}

		string title = "CIS565 Rasterizer | "+ utilityCore::convertIntToString((int)fps) + "FPS";
		glutSetWindowTitle(title.c_str());

		glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, displayImage);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, 
			GL_RGBA, GL_UNSIGNED_BYTE, NULL);

		glClear(GL_COLOR_BUFFER_BIT);   

		// VAO, shader program, and texture already bound
		glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

		glutPostRedisplay();
		glutSwapBuffers();
	}

	void keyboard(unsigned char key, int x, int y)
	{
		switch (key) 
		{
			case(27):
			shut_down(1);    
			break;
		}
	}

	void mouseButton(int button, int state, int x, int y)
	{
		// if both the left mouse button and ALT key is down
		int specialKey = glutGetModifiers();
		if (button == GLUT_LEFT_BUTTON && specialKey == GLUT_ACTIVE_ALT) 
		{
			lastMousePosition = vec2(x,y);
			printf("(%d,%d)\n", lastMousePosition.x, lastMousePosition.y);
			altLmbDown = true;
			altRmbDown = false;
		}
		else if (button == GLUT_RIGHT_BUTTON && specialKey == GLUT_ACTIVE_ALT)
		{
			lastMousePosition = vec2(x,y);
			altRmbDown = true;
			altLmbDown = false;
		}
		else
		{
			altRmbDown = false;
			altLmbDown = false;
		}
	}

	void mouseMove(int x, int y)
	{
		if (altLmbDown)
		{
			float speed = 0.5f;
			vec2 position = vec2(x,y);
			vec2 delta = (position - lastMousePosition) * speed;

			alpha -= delta.x;
			while (alpha < 0) 
			{
				alpha += 360;
			}
			while (alpha >= 360)
			{
				alpha -= 360;
			}

			beta -= delta.y;

			if (beta < -90)
				beta = -90;
			if (beta > 90)
				beta = 90;

			// update lastMousePosition
			lastMousePosition = position;
		}

		if (altRmbDown)
		{
			float speed = 0.01f;
			vec2 position = vec2(x,y);
			vec2 delta = (position - lastMousePosition) * speed;

			if (delta.x > 0)
				cam->position.z += 0.1f;
			else if (delta.x < 0)
				cam->position.z -= 0.1f;

			lastMousePosition = position;
		}
	}

	void updateCamera()
	{
		// update position
		//cam->position = vec3(cam->position.x, cam->position.y, zDistance);

		// update orientation
		mat4 cameraTransform(1);
		cameraTransform = rotate(cameraTransform, alpha, vec3(0,1,0));
		cameraTransform = rotate(cameraTransform, beta, vec3(1,0,0));

		vec4 pos4 = (cameraTransform * vec4(cam->position, 1.0));
		vec4 up4 = (cameraTransform * vec4(cam->up, 0.0));
		vec4 viewDir4 = (cameraTransform * vec4(0,0,-1, 0.0));


		cam->position = vec3(pos4.x, pos4.y, pos4.z);
		cam->up = vec3(up4.x, up4.y, up4.z);
		
		viewDir = vec3(viewDir4.x, viewDir4.y, viewDir4.z);
		mat4 view = glm::lookAt(cam->position, cam->position + viewDir, cam->up); // LOOK: Center is now position + view direction
		//mat4 view = glm::lookAt(cam->position, cam->position + viewDir, cam->up); // LOOK: Center is now position + view direction
		cam->view = view;

		if (alpha != 0)
			printf("alpha = %f\n", alpha);

		if (beta != 0)
			printf("beta = %f\n", beta);

		alpha = 0;
		beta = 0;
		

		zTranslateDistance = 0;
	}

#endif
  
//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

#ifdef __APPLE__
  void init(){

    if (glfwInit() != GL_TRUE){
      shut_down(1);      
    }

    // 16 bit color, no depth, alpha or stencil buffers, windowed
    if (glfwOpenWindow(width, height, 5, 6, 5, 0, 0, 0, GLFW_WINDOW) != GL_TRUE){
      shut_down(1);
    }

    // Set up vertex array object, texture stuff
    initVAO();
    initTextures();
  }
#else
  void init(int argc, char* argv[]){
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(width, height);
    glutCreateWindow("CIS565 Rasterizer");

    // Init GLEW
    glewInit();
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
      /* Problem: glewInit failed, something is seriously wrong. */
      std::cout << "glewInit failed, aborting." << std::endl;
      exit (1);
    }

    initVAO();
    initTextures();
  }
#endif

void initPBO(GLuint* pbo){
  if (pbo) {
    // set up vertex data parameter
    int num_texels = width*height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    
    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1,pbo);
    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject( *pbo );
  }
}

void initCuda(){
  // Use device with highest Gflops/s
  cudaGLSetGLDevice( compat_getMaxGflopsDeviceId() );

  initPBO(&pbo);

  // Clean up on program exit
  atexit(cleanupCuda);

  runCuda();
}

void initCamera()
{
	cam = new camera();
	vec3 up = vec3(0,1,0);
	vec3 cameraPosition = vec3(0, 0, 1.0);
	mat4 projection = glm::perspective(-fovy, float(width)/float(height), zNear, zFar); // LOOK: Passed in -fovy to have the image rightside up
    mat4 view = glm::lookAt(cameraPosition, center, up);
	cam->zFar = zFar;
	cam->zNear = zNear;
	cam->fovy = fovy;
	cam->position = cameraPosition;
	cam->projection = projection;
	cam->view = view;
	cam->resolution = vec2(width, height);
	cam->up = up;
}

void initLights()
{
	lightsize = 4;
	lights = new light[lightsize];

	// first light
	lights[0].color = vec3(1,0,0);
	lights[0].position = vec3(0, 0.f, 2.f);

	// second light
	lights[1].color = vec3(0,1,0);
	lights[1].position = vec3(-3.f, 5.f, 1.f);

	// third light
	lights[2].color = vec3(0,0,1);
	lights[2].position = vec3(3.f, 5.f, 1.f);

	// fourth light
	lights[3].color = vec3(1,1,1);
	lights[3].position = vec3(0.0f, 2.5f, 2.f);

	//// first light
	//lights[0].color = vec3(0,1,1);
	//lights[0].position = vec3(0, 0.f, 2.f);

	//// second light
	//lights[1].color = vec3(0,1,0);
	//lights[1].position = vec3(-3.f, 5.f, 1.f);

	//// third light
	//lights[2].color = vec3(0,0,1);
	//lights[2].position = vec3(3.f, 5.f, 1.f);

	//// fourth light
	//lights[3].color = vec3(1,0,0);
	//lights[3].position = vec3(0.0f, 2.5f, 2.f);
}

void initTextures(){
    glGenTextures(1,&displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
        GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void){
    GLfloat vertices[] =
    { 
        -1.0f, -1.0f, 
         1.0f, -1.0f, 
         1.0f,  1.0f, 
        -1.0f,  1.0f, 
    };

    GLfloat texcoords[] = 
    { 
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);
    
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0); 
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader(const char *vertexShaderPath, const char *fragmentShaderPath){
    GLuint program = glslUtility::createProgram(vertexShaderPath, fragmentShaderPath, attributeLocations, 2);
    GLint location;

    glUseProgram(program);
    
    if ((location = glGetUniformLocation(program, "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }

    return program;
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda(){
  if(pbo) deletePBO(&pbo);
  if(displayImage) deleteTexture(&displayImage);
}

void deletePBO(GLuint* pbo){
  if (pbo) {
    // unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(*pbo);
    
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glDeleteBuffers(1, pbo);
    
    *pbo = (GLuint)NULL;
  }
}

void deleteTexture(GLuint* tex){
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}
 
void shut_down(int return_code){
	delete cam;
	delete[] lights;
	kernelCleanup();
	cudaDeviceReset();
	#ifdef __APPLE__
	glfwTerminate();
	#endif
	exit(return_code);
}
