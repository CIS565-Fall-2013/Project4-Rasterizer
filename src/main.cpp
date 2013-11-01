// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include "main.h"
#include <GL/glut.h>

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

  initCuda();

  initVAO();
  initTextures();

  initCamera();

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
	glutMouseFunc(GLUTMouse);
	glutMotionFunc(GLUTMotion);

    glutMainLoop();
  #endif
  kernelCleanup();
  return 0;
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda(){
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
	dptr=NULL;

	vbo = mesh->getVBO();
	vbosize = mesh->getVBOsize();

	float newcbo[] = {0.0, 1.0, 0.0, 
					0.0, 0.0, 1.0, 
					1.0, 0.0, 0.0};
	cbo = newcbo;
	cbosize = 9;

	ibo = mesh->getIBO();
	ibosize = mesh->getIBOsize();

	nbo = mesh->getNBO();
	nbosize = mesh->getNBOsize();

	// set up model matrix
	glm::mat4 modelMatrix(1.0f);
	
	// set up view matrix
	glm::mat4 viewMatrix = glm::lookAt(camera_eye, mesh_center, camera_up);

	// set up perspective matrix
	float fovY = 180.0*camera_yfov/PI;
	float aspectRatio = width/height;
	float near = 0.001f*mesh_size;
	float far  = 100.0f*mesh_size;
	glm::mat4 projectionMatrix = glm::perspective(fovY, aspectRatio, near, far);

	// setup light rig
	glm::vec3 light(mesh_center.x + 1.5f*mesh_size, mesh_center.y + 4.0f*mesh_size, mesh_center.z + 1.5f*mesh_size);

	// run rasterizer
	cudaGLMapBufferObject((void**)&dptr, pbo);
	cudaRasterizeCore(dptr, glm::vec2(width, height), frame, vbo, vbosize, cbo, cbosize, ibo, ibosize, nbo, nbosize, 
					  modelMatrix, viewMatrix, projectionMatrix, light);
	cudaGLUnmapBufferObject(pbo);

	vbo = NULL;
	cbo = NULL;
	ibo = NULL;

	frame++;
	fpstracker++;

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

  void display(){
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

glm::vec3 vecRotate(glm::vec3 vector, glm::vec3 axis, float theta)
{
  // Rotate vector counterclockwise around axis (looking at axis end-on) (rz(xaxis) = yaxis)
  // From Goldstein: v' = v cos t + a (v . a) [1 - cos t] - (v x a) sin t 
  const float cos_theta = cos(theta);
  const float dot = glm::dot(vector,axis);
  glm::vec3 return_vector(vector);
  glm::vec3 cross = glm::cross(vector, axis);
  return_vector *= cos_theta;
  return_vector += axis * dot * (1.0f - cos_theta);
  return_vector -= cross * sin(theta); 
  return return_vector;
}

void GLUTMouse(int button, int state, int x, int y)
{  
	// Process mouse button event
	if (state == GLUT_DOWN) {
	if (button == GLUT_LEFT_BUTTON) {
	}
	else if (button == GLUT_MIDDLE_BUTTON) {
	}
	else if (button == GLUT_RIGHT_BUTTON) {
	}
	}

	// Remember button state 
	int b = (button == GLUT_LEFT_BUTTON) ? 0 : ((button == GLUT_MIDDLE_BUTTON) ? 1 : 2);
	GLUTbutton[b] = (state == GLUT_DOWN) ? 1 : 0;

	// Remember modifiers 
	GLUTmodifiers = glutGetModifiers();

	// Remember mouse position 
	GLUTmouse[0] = x;
	GLUTmouse[1] = y;

	// Redraw
	glutPostRedisplay();
}

void GLUTMotion(int x, int y) {
  
	// Compute mouse movement
	int dx = x - GLUTmouse[0];
	int dy = y - GLUTmouse[1];
  
	// Process mouse motion event
	if ((dx != 0) || (dy != 0)) {
		if ((GLUTbutton[0] && (GLUTmodifiers & GLUT_ACTIVE_SHIFT)) || GLUTbutton[1]) {
			// Scale world 
			float factor = (float) dx / GLUTwindow_width;
			factor += (float) dy / GLUTwindow_height;
			factor = exp(2.0 * factor);
			factor = (factor - 1.0) / factor;
			glm::vec3 translation = (mesh_center - camera_eye) * factor;
			camera_eye += translation;
			camera_zoom = glm::length(camera_eye);
			glutPostRedisplay();
		}
		else if (GLUTbutton[0]) {
			// Rotate world
			dx = -dx;
			float length = glm::distance(mesh_center, camera_eye) * 2.0f * tan(camera_yfov);
			float vx = length * (float) dx / GLUTwindow_width;
			float vy = length * (float) dy / GLUTwindow_height;
			glm::vec3 camera_right = glm::cross(camera_up, camera_towards);
			glm::vec3 translation = -((camera_right * vx) + (camera_up * vy));
			camera_eye += translation;
			camera_eye = glm::normalize(camera_eye) * camera_zoom;
			camera_towards = glm::normalize(camera_eye - mesh_center);
			camera_up -= glm::dot(camera_up, camera_towards)*camera_towards;
			glutPostRedisplay();
		}
	}

	// Remember mouse position 
	GLUTmouse[0] = x;
	GLUTmouse[1] = y;
}

void initCamera(void) {
	
	// set up view matrix
	float *bbox = mesh->getBoundingBox();
	float xmin = bbox[X_MIN]; float xmax = bbox[X_MAX];
	float ymin = bbox[Y_MIN]; float ymax = bbox[Y_MAX];
	float zmin = bbox[Z_MIN]; float zmax = bbox[Z_MAX];
	float xcenter = (xmax-xmin)/2.0f;
	float ycenter = (ymax-ymin)/2.0f;
	float zcenter = (zmax-zmin)/2.0f;
	
	mesh_size = glm::sqrt(pow(xmax-xmin, 2) + pow(xmax-xmin, 2) + pow(xmax-xmin, 2));
	mesh_center = glm::vec3(0,ycenter,0);
	camera_eye = glm::vec3(0,ycenter,2*mesh_size);
	camera_towards = glm::normalize(glm::vec3(0,ycenter,0) - camera_eye);
	camera_zoom = 2.0f*mesh_size;
}

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
  kernelCleanup();
  cudaDeviceReset();
  #ifdef __APPLE__
  glfwTerminate();
  #endif
  exit(return_code);
}
