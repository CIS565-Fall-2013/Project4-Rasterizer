// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include "main.h"
#include "glm/gtc/matrix_transform.hpp"

bool first = true;
int oldX = 0, oldY = 0, dx = 0, dy = 0;	
bool leftMButtonDown = false;

float camRadius = 3.5f;
float scrollSpeed = 0.33f;
bool outline = false, camControl = false;

cbuffer constantBuffer;
glm::mat4	cameraTransform;
glm::vec3	currentLookAt =  glm::vec3 (0,0,1);
float u = 0.5;
float vvv = 0;
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
	else if (strcmp(header.c_str(), "radius")==0)
	{
      camRadius = strtod (data.c_str(), NULL);
    }
	else if (strcmp(header.c_str(), "scrollspeed")==0)
	{
      scrollSpeed = strtod (data.c_str(), NULL);
    }
	else if (strcmp(header.c_str(), "outline")==0)
	{
		if (strcmp(data.c_str(), "true")==0)
			outline = true;
    }
	else if (strcmp(header.c_str(), "cameraControl")==0)
	{
		if (strcmp(data.c_str(), "true")==0)
			camControl = true;
    }
  }

  if(!loadedScene){
    cout << "Usage: mesh=[obj file]" << endl;
	std::cin.get ();
    return 0;
  }

  frame = 0;
  seconds = time (NULL);
  fpstracker = 0;

  constantBuffer.model = glm::translate (glm::mat4 (1.0f), glm::vec3 (0,0,1.0))*glm::rotate (glm::mat4 (1.0f), 180.0f, glm::vec3 (1,0,0));
  constantBuffer.modelIT = glm::transpose (glm::inverse (constantBuffer.model));
  constantBuffer.lightPos = glm::vec4 (0, 10, -10, 1);
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
	glutMouseFunc (onButtonPress);
	glutMotionFunc (onMouseMove);

    glutMainLoop();
  #endif
  kernelCleanup();
  return 0;
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda(bool &isFirstTime){
  // Map OpenGL buffer object for writing from CUDA on a single GPU
  // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
  dptr=NULL;

  vbo = mesh->getVBO();
  vbosize = mesh->getVBOsize();

  float newcbo[] = {0.35, 0.35, 0.35, 
                    0.4, 0.4, 0.4, 
                    0.3, 0.3, 0.3};
  cbo = newcbo;
  cbosize = 9;

  ibo = mesh->getIBO();
  ibosize = mesh->getIBOsize();

  nbo = mesh->getNBO ();
  nbosize = mesh->getNBOsize ();

  constantBuffer.projection = /*glm::mat4 (1.0f);*/glm::perspective (60.0f, (float)(width/height), 0.1f, 100.0f);

  // Change u and v with respect to change in Y and X respectively.
  u = u + (float)dy / (float)(height/scrollSpeed);
  vvv = vvv - (float)dx / (float)(width/scrollSpeed);

  // Make sure u and v stay within [0,1]
  if (u < 0)
	  u = - u;
  else if (u > 1)
	  u = 1 - (u - 1);

  if (vvv < 0)
	  vvv = 1 + vvv;
  else if (vvv > 1)
	  vvv = vvv - 1;

  // Camera transformation - translation part:
  cameraTransform = glm::translate (glm::mat4 (1.0f), 
								    glm::vec3 (camRadius*sin (PI*u)*sin (2.0f*PI*vvv), 
												camRadius*cos (PI*u), 
												-camRadius*sin (PI*u)*cos (2.0f*PI*vvv)));
  glm::vec4 camOrigin = cameraTransform*glm::vec4 (0.0f, 0.0f, 0.0f, 1.0f);

  // Now for rotation:
  glm::vec3 target_lookat = glm::normalize (glm::vec3 (0.0f) - glm::vec3 (camOrigin.x, camOrigin.y, camOrigin.z));
  currentLookAt = glm::normalize (currentLookAt);
  // Refer Real-Time Rendering sec. "Quaternion Transforms", for explanation of following:
  glm::mat4 rotationMat (1.0f);
  glm::vec3 v = glm::cross (currentLookAt, target_lookat);
  float e = glm::dot (currentLookAt, target_lookat);
  float vSqrMagnitude = glm::dot (v, v);
  if (vSqrMagnitude > 0.001f)
  {
	  float h = (1-e) / vSqrMagnitude;
	  rotationMat [0][0] = e + h*(v.x*v.x);		rotationMat [0][1] = h*v.x*v.y + v.z;	rotationMat [0][2] = h*v.x*v.z - v.y;	rotationMat [0][3] = 0;
	  rotationMat [1][0] = h*v.x*v.y - v.z;		rotationMat [1][1] = e + h*(v.y*v.y);	rotationMat [1][2] = h*v.y*v.z + v.x;	rotationMat [1][3] = 0;
	  rotationMat [2][0] = h*v.x*v.z + v.y;		rotationMat [2][1] = h*v.y*v.z - v.x;	rotationMat [2][2] = e + h*(v.z*v.z);	rotationMat [2][3] = 0;
	  rotationMat [3][0] = 0;					rotationMat [3][1] = 0;					rotationMat [3][2] = 0;					rotationMat [3][3] = 1;
  }
  cameraTransform = cameraTransform * glm::transpose (rotationMat);
  
  currentLookAt = target_lookat;	// Why? Because they're both in world space!

  glm::vec4 center =  /*camOrigin + */(/*cameraTransform * */glm::vec4 (0,0,0,0));
  glm::vec4 up =  /*glm::normalize(cameraTransform * */glm::vec4 (0,1,0,0)/*)*/;

  if (camControl)
	  constantBuffer.view = glm::lookAt (glm::vec3 (camOrigin.x, camOrigin.y, camOrigin.z), 
									 glm::vec3 (center.x, center.y, center.z), 
									 glm::vec3 (up.x, up.y, up.z));
  else
	  constantBuffer.view = glm::lookAt (glm::vec3 (0.0f, 0.0f, 0.0f),/*glm::vec3 (camOrigin.x, camOrigin.y, camOrigin.z)*/ 
										glm::vec3 (0.0f,0.0f,1.0f), 
										glm::vec3 (0.0f,1.0f,0.0f));
  cudaGLMapBufferObject((void**)&dptr, pbo);
  cudaRasterizeCore(dptr, glm::vec2(width, height), frame, vbo, vbosize, cbo, cbosize, ibo, ibosize, 
					nbo, nbosize, isFirstTime, constantBuffer);
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
    runCuda(first);
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
	   case 'o':
	   case 'O':
		   outline = !outline;
		   break;
    }
  }

  void onButtonPress (int button, int state, int x, int y)
  {
	  if ((button == GLUT_LEFT_BUTTON) && (state == GLUT_DOWN))
	  {
		  oldX = x;	oldY = y;
		  leftMButtonDown = true;
	  }
	  else
	  {
		  leftMButtonDown = false;
		  dx = 0;
		  dy = 0;
	  }
  }

  void onMouseMove (int x, int y)
  {
	  if (leftMButtonDown)
	  {
		  dx = x - oldX;
		  dy = y - oldY;
	  }
	  else
	  {
		  dx = 0;
		  dy = 0;
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

  runCuda(first);
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
