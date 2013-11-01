// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include "main.h"

glm::fquat computeArcballRot(glm::vec3 point0, glm::vec3 point1)
{
	glm::fquat rotQuat = glm::fquat();
	glm::vec3 crossProduct = glm::cross(startP0, endP1);
	rotQuat.x = crossProduct.x;
	rotQuat.y = crossProduct.y;
	rotQuat.z = crossProduct.z;
	rotQuat.w = glm::dot(startP0, endP1);
	return rotQuat;
}

glm::vec3 computeSphereCoords(int screenX, int screenY, int screenWidth, int screenHeight)
{
  glm::vec2 center = glm::vec2(screenWidth/2, screenHeight/2);
	float radius = min(screenWidth/2, screenHeight/2);
	glm::vec3 pt = glm::vec3(0.0f); //point that we are returning

	pt.x = (screenX - center.x)/radius;
	pt.y = (screenY - center.y)/radius;
	float r = pt.x*pt.x + pt.y*pt.y;
	if(r > 1.0){
		float s = 1.0/sqrt(r);
		pt.x = s*pt.x;
		pt.y = s*pt.y;
		pt.z = 0.0;
	}
	else
		pt.z = sqrt(1.0 - r);

	return pt;
}

void mousePress(int button, int state, int x, int y)
{
	if(button == 3){
		position += 0.1f*currView;  
	} else if (button == 4){
		position -= 0.1f*currView;
	}
	if(state == GLUT_DOWN){ //pressed
		startP0 = computeSphereCoords(x, y, width, height);
		endP1 = startP0; 
		arcballRotOn = true;
	} else{ //released
		qstart = computeArcballRot(startP0, endP1)*qstart; 
		arcballRotOn = false;
	}
}

void mouseMove(int x, int y)
{
    endP1 = computeSphereCoords(x, y, width, height);
}


//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv){

	 qstart.x = 0;
	qstart.y = 0;
	qstart.z = 0;
	qstart.w = 1;
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

  //myColorReader = new colorReader(mesh->getCBOsize(), "../../objs/colors_90_deg.ncolors");

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
  dptr=NULL;

  vbo = mesh->getVBO();

  vbosize = mesh->getVBOsize();
  nbo = mesh->getNBO();

  ////color for the triangle
  //float newcbo[] = {0.0, 1.0, 0.0, 
  //                  0.0, 0.0, 1.0, 
  //                  1.0, 0.0, 0.0};
  //cbo = newcbo;
  //cbosize = 9;
  mesh->setColor(glm::vec3(0, 1, 0));

  if(normalsAsColors){
	cbo = mesh->getNBO();
  } else {
	cbo = mesh->getCBO();
  }
  //cbo = myColorReader->getCBO();
  cbosize = mesh->getCBOsize();

  ibo = mesh->getIBO();
  ibosize = mesh->getIBOsize();

  cudaGLMapBufferObject((void**)&dptr, pbo);
  float angleDeg; 
  
  if(rotateModel)
	angleDeg = frame%360; //rotation angle of the model in degrees
  else
	angleDeg = 0;

  if(firstRun){
      startView = currView;
      startUp = currUp;
    } 

    if(arcballRotOn){
      //currView = computeArcballRot(startP0, endP1) * qstart * startView;  
      //currUp = computeArcballRot(startP0, endP1) * qstart * startUp;  
		currRot = computeArcballRot(startP0, endP1) * qstart;
    }

  cudaRasterizeCore(dptr, glm::vec2(width, height), frame, vbo, vbosize, cbo, cbosize, ibo, ibosize, nbo, angleDeg, 
	  camPos, drawLines, useShading, interpColors, useLargeStep, checkWriteCount, backfaceCull, currRot, nineteen_eighty_four);
  cudaGLUnmapBufferObject(pbo);

  vbo = NULL;
  cbo = NULL;
  ibo = NULL;
  nbo = NULL;

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
	   case(119): //W
         camPos.y += 0.1f;  
         break;
	   case(115): //S
         camPos.y -= 0.1f;  
         break;
	   case(97): //A
         camPos.x -= 0.1f;  
         break;
	   case(100): //D
         camPos.x += 0.1f;  
         break;
	   case(113): //Q
         camPos.z -= 0.1f;  
         break;
	   case(101): //E
         camPos.z += 0.1f;  
         break;
	   case(108): //l
		drawLines = !drawLines;
		break;
	   case(114): //r
		rotateModel = !rotateModel;
		break;
	   case(110): //n
		normalsAsColors = !normalsAsColors;
		break;
	   case(104): //h for sHade
		useShading = !useShading;
		break;
	   case(99): //c for Color
		interpColors = !interpColors;
		break;
	   case(116): //t for sTep;
		useLargeStep = !useLargeStep;
		break;
	   case(107): //k for checK if we write twice
		checkWriteCount = !checkWriteCount;
		break;
	   case(98): //B for backface cull
		   backfaceCull = !backfaceCull;
		   break;
	   case(56): //8 for 1984 mode
		   nineteen_eighty_four = !nineteen_eighty_four;
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
