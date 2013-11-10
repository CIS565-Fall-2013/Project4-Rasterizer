// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include "main.h"
#include "glm/gtx/transform.hpp"

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
      obj* mesh = new obj();
      objLoader* loader = new objLoader(data, mesh);
      mesh->buildVBOs();
			meshes.push_back(mesh);
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
		glutMouseFunc(mousePress);
		glutMotionFunc(mouseMove);

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

	float wood[] = {245.0/255.0, 222.0/255.0, 179.0/255.0};

  float blue[] = {0.4, 0.7, 1.0};

	float pink[] = {1.0, 166.0/255.0, 186.0/255.0};
	
	float white[] = {1.0, 1.0, 1.0};

	float gold[] = {1.0, 0.77, 0.03};

	float green[] = {50.0/255.0, 205.0/255.0, 50.0/255.0};

	clearBuffers(glm::vec2(width, height));

	//------------------------------
  //draw the caps
  //------------------------------
  vbo = meshes[12]->getVBO();
  vbosize = meshes[12]->getVBOsize();
	cbo = wood;
  cbosize = 3;
  nbo = meshes[12]->getNBO();
  nbosize = meshes[12]->getNBOsize();
  ibo = meshes[12]->getIBO();
  ibosize = meshes[12]->getIBOsize();

  cudaRasterizeCore(glm::vec2(width, height), eye, center, vbo, vbosize, cbo, cbosize, nbo, nbosize, ibo, ibosize, false, false, 0);

	//------------------------------
  //draw the portals
  //------------------------------
	//portal 1
	vbo = meshes[0]->getVBO();
	vbosize = meshes[0]->getVBOsize();
	ibo = meshes[0]->getIBO();
  ibosize = meshes[0]->getIBOsize();
	drawToStencilBuffer(glm::vec2(width, height), eye, center, vbo, vbosize, ibo, ibosize, 1);

	//portal 2
	vbo = meshes[1]->getVBO();
	vbosize = meshes[1]->getVBOsize();
	ibo = meshes[1]->getIBO();
  ibosize = meshes[1]->getIBOsize();
	drawToStencilBuffer(glm::vec2(width, height), eye, center, vbo, vbosize, ibo, ibosize, 2);

	//portal 3
	vbo = meshes[2]->getVBO();
	vbosize = meshes[2]->getVBOsize();
	ibo = meshes[2]->getIBO();
  ibosize = meshes[2]->getIBOsize();
	drawToStencilBuffer(glm::vec2(width, height), eye, center, vbo, vbosize, ibo, ibosize, 3);

	//portal 4
	vbo = meshes[3]->getVBO();
	vbosize = meshes[3]->getVBOsize();
	ibo = meshes[3]->getIBO();
  ibosize = meshes[3]->getIBOsize();
	drawToStencilBuffer(glm::vec2(width, height), eye, center, vbo, vbosize, ibo, ibosize, 4);

	//------------------------------
  //draw box 1
  //------------------------------
	clearOnStencil(glm::vec2(width, height), 1);

	vbo = meshes[4]->getVBO();
  vbosize = meshes[4]->getVBOsize();
	cbo = wood;
  cbosize = 3;
  nbo = meshes[4]->getNBO();
  nbosize = meshes[4]->getNBOsize();
  ibo = meshes[4]->getIBO();
  ibosize = meshes[4]->getIBOsize();

  cudaRasterizeCore(glm::vec2(width, height), eye, center, vbo, vbosize, cbo, cbosize, nbo, nbosize, ibo, ibosize, true, false, 1);

	//------------------------------
  //draw bunny
  //------------------------------
	vbo = meshes[5]->getVBO();
  vbosize = meshes[5]->getVBOsize();
	cbo = blue;
  cbosize = 3;
  nbo = meshes[5]->getNBO();
  nbosize = meshes[5]->getNBOsize();
  ibo = meshes[5]->getIBO();
  ibosize = meshes[5]->getIBOsize();

  cudaRasterizeCore(glm::vec2(width, height), eye, center, vbo, vbosize, cbo, cbosize, nbo, nbosize, ibo, ibosize, true, true, 1);

	//------------------------------
  //draw box 2
  //------------------------------
	clearOnStencil(glm::vec2(width, height), 2);

	vbo = meshes[6]->getVBO();
  vbosize = meshes[6]->getVBOsize();
	cbo = wood;
  cbosize = 3;
  nbo = meshes[6]->getNBO();
  nbosize = meshes[6]->getNBOsize();
  ibo = meshes[6]->getIBO();
  ibosize = meshes[6]->getIBOsize();

  cudaRasterizeCore(glm::vec2(width, height), eye, center, vbo, vbosize, cbo, cbosize, nbo, nbosize, ibo, ibosize, true, false, 2);

	//------------------------------
  //draw dragon
  //------------------------------
	vbo = meshes[7]->getVBO();
  vbosize = meshes[7]->getVBOsize();
	cbo = gold;
  cbosize = 3;
  nbo = meshes[7]->getNBO();
  nbosize = meshes[7]->getNBOsize();
  ibo = meshes[7]->getIBO();
  ibosize = meshes[7]->getIBOsize();

  cudaRasterizeCore(glm::vec2(width, height), eye, center, vbo, vbosize, cbo, cbosize, nbo, nbosize, ibo, ibosize, true, true, 2);

	//------------------------------
  //draw box 3
  //------------------------------
	clearOnStencil(glm::vec2(width, height), 3);

	vbo = meshes[8]->getVBO();
  vbosize = meshes[8]->getVBOsize();
	cbo = wood;
  cbosize = 3;
  nbo = meshes[8]->getNBO();
  nbosize = meshes[8]->getNBOsize();
  ibo = meshes[8]->getIBO();
  ibosize = meshes[8]->getIBOsize();

  cudaRasterizeCore(glm::vec2(width, height), eye, center, vbo, vbosize, cbo, cbosize, nbo, nbosize, ibo, ibosize, true, false, 3);

	//------------------------------
  //draw cow
  //------------------------------
	vbo = meshes[9]->getVBO();
  vbosize = meshes[9]->getVBOsize();
	cbo = green;
  cbosize = 3;
  nbo = meshes[9]->getNBO();
  nbosize = meshes[9]->getNBOsize();
  ibo = meshes[9]->getIBO();
  ibosize = meshes[9]->getIBOsize();

  cudaRasterizeCore(glm::vec2(width, height), eye, center, vbo, vbosize, cbo, cbosize, nbo, nbosize, ibo, ibosize, true, true, 3);

	//------------------------------
  //draw box 4
  //------------------------------
	clearOnStencil(glm::vec2(width, height), 4);

	vbo = meshes[10]->getVBO();
  vbosize = meshes[10]->getVBOsize();
	cbo = wood;
  cbosize = 3;
  nbo = meshes[10]->getNBO();
  nbosize = meshes[10]->getNBOsize();
  ibo = meshes[10]->getIBO();
  ibosize = meshes[10]->getIBOsize();

  cudaRasterizeCore(glm::vec2(width, height), eye, center, vbo, vbosize, cbo, cbosize, nbo, nbosize, ibo, ibosize, true, false, 4);

	//------------------------------
  //draw buddha
  //------------------------------
	vbo = meshes[11]->getVBO();
  vbosize = meshes[11]->getVBOsize();
	cbo = pink;
  cbosize = 3;
  nbo = meshes[11]->getNBO();
  nbosize = meshes[11]->getNBOsize();
  ibo = meshes[11]->getIBO();
  ibosize = meshes[11]->getIBOsize();

  cudaRasterizeCore(glm::vec2(width, height), eye, center, vbo, vbosize, cbo, cbosize, nbo, nbosize, ibo, ibosize, true, true, 4);

	dptr=NULL;
	cudaGLMapBufferObject((void**)&dptr, pbo);
	renderToPBO(dptr, glm::vec2(width, height), eye);
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

	void mousePress(int button, int state, int x, int y) {
		if (state == GLUT_DOWN) {
			buttonPressed = button;
			prevX = x;
			prevY = y;
		}
		else {
			buttonPressed = -1;
			prevX = -1;
			prevY = -1;
		}
	}

	void mouseMove(int x, int y) {
		x = max(0, x);
		x = min(x, width);
		y = max(0, y);
		y = min(y, height);
		int offsetX = x - prevX;
		int offsetY = y - prevY;
		prevX = x;
		prevY = y;

		glm::vec4 teye;
		glm::mat4 rotation;
		glm::vec3 axis;
		glm::vec3 step;

		switch (buttonPressed) {
		case(GLUT_LEFT_BUTTON):
			teye = glm::vec4(eye - center, 1);
			axis = glm::normalize(glm::cross(glm::vec3(0,1,0), eye-center));
			rotation = glm::rotate((float)(-360.0f/width*offsetX), 0.0f, 1.0f, 0.0f) * glm::rotate((float)(-360.0f/width*offsetY), axis.x, axis.y, axis.z);
			teye = rotation * teye;
			eye = glm::vec3(teye);
			eye = eye + center;
			break;
		case(GLUT_MIDDLE_BUTTON): //need revise
			eye += glm::vec3(-0.002, 0, 0) * (float)offsetX;
			eye += glm::vec3(0, 0.002, 0) * (float)offsetY;
			center += glm::vec3(-0.002, 0, 0) * (float)offsetX;
			center += glm::vec3(0, 0.002, 0) * (float)offsetY;
			break;
		case(GLUT_RIGHT_BUTTON): //need revise
			if (glm::distance(center, eye) > 0.01 || (offsetX < 0 && glm::distance(center, eye) < 20)) {
				step = 0.01f * glm::normalize(center - eye);
				eye += step * (float)offsetX;
			}
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

		buttonPressed = -1; //no mouse button is pressed
		prevX = -1;
		prevY = -1;

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

	initBuffers(glm::vec2(width, height));
	initLights();

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
	freeBuffers();
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
	meshes.clear();
  #ifdef __APPLE__
  glfwTerminate();
  #endif
  exit(return_code);
}
