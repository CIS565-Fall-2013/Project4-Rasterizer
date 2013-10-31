// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include "main.h"
#include "structs.h"

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

  // Initialization of camera parameters
  cam.position = glm::vec3(0.0f, 1.0f, 1.0f);
  cam.up       = glm::vec3(0.0f, 1.0f, 0.0f);
  cam.view     = glm::normalize(-cam.position);
  cam.right    = glm::normalize(glm::cross(cam.view, cam.up));
  cam.fovy     = 45.0f;

  // Initialize transformation
  model      = new glm::mat4(utilityCore::buildTransformationMatrix(glm::vec3(0.0f, -0.2f, 0.0f), glm::vec3(0.0f), glm::vec3(0.7f)));
  view       = new glm::mat4(glm::lookAt(cam.position, glm::vec3(0.0f), cam.up));
  projection = new glm::mat4(glm::perspective(cam.fovy, (float)width / height, zNear, zFar));
  transformModel2Projection  = new cudaMat4(utilityCore::glmMat4ToCudaMat4(*projection * *view * *model));

  // Initialize viewport in the model space
  viewPort   = glm::normalize(utilityCore::multiplyMat(utilityCore::glmMat4ToCudaMat4(*projection * *view), glm::vec4(cam.view, 1.0f)));

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
    glutSpecialFunc(specialFunction);
    glutMouseFunc(mouseClick);
    glutMotionFunc(mouseMotion);

    glutMainLoop();
  #endif
  kernelCleanup();
  delete model;
  delete view;
  delete projection;
  delete transformModel2Projection;
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

  // Update view and model to projection transform matrices in each step when interacting with keyboard or mouse
  *view = glm::lookAt(cam.position, glm::vec3(0.0f), cam.up);
  *transformModel2Projection = utilityCore::glmMat4ToCudaMat4(*projection * *view * *model);
  viewPort = glm::normalize(utilityCore::multiplyMat(utilityCore::glmMat4ToCudaMat4(*projection * *view), glm::vec4(cam.view, 1.0f)));

  // Transformation Feedback
  std::cout <<  "\n The model-view-projection transformation is:" << std::endl;
  utilityCore::printMat4(*projection * *view * *model);

  std::cout <<  "\n The view port in the clip space is:" << std::endl;
  utilityCore::printVec3(viewPort);

  cudaGLMapBufferObject((void**)&dptr, pbo);
  cudaRasterizeCore(dptr, glm::vec2(width, height), frame, vbo, vbosize, cbo, cbosize, ibo, ibosize, nbo, nbosize, transformModel2Projection, viewPort, antialiasing, depthFlag, flatcolorFlag, color, multicolorFlag);
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
bool pauseFlag = false;
  void display(){
    if(!pauseFlag)
      runCuda();

	time_t seconds2 = time (NULL);

    if(seconds2-seconds >= 1){

      fps = fpstracker/(seconds2-seconds);
      fpstracker = 0;
      seconds = seconds2;

    }

    string title = "CIS565 Rasterizer of Qiong Wang | "+ utilityCore::convertIntToString((int)fps) + "FPS";
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
       case(' '):
         pauseFlag = ! pauseFlag;
         break;
       case('a'):
         antialiasing = ! antialiasing;
         break;
       case('d'):
         depthFlag = ! depthFlag;
         break;
	   case('f'):
         flatcolorFlag = ! flatcolorFlag;
         break;
	   case('m'):
         multicolorFlag = ! multicolorFlag;
         break;
       case('`'):
         color = 0;
         break;
	   case('1'):
         color = 1;
         break;
	   case('2'):
         color = 2;
         break;
       case('3'):
         color = 3;
         break;
	   case('4'):
         color = 4;
         break;
       case('['):
         cam.position -= 0.1f * cam.up;
         cam.view     = glm::normalize(-cam.position);
         cam.right    = glm::normalize(glm::cross(cam.view, cam.up));
         cam.up       = glm::normalize(glm::cross(cam.right, cam.view));
         break;
       case(']'):
         cam.position += 0.1f * cam.up;
         cam.view     = glm::normalize(-cam.position);
         cam.right    = glm::normalize(glm::cross(cam.view, cam.up));
         cam.up       = glm::normalize(glm::cross(cam.right, cam.view));
         break;
    }
  }

  void specialFunction(int key, int x, int y) {
    // callback function for glutSpecialFunc
    switch (key)
    {
      case(GLUT_KEY_UP):
        cam.position -= 0.1f * cam.position;        
        break;
      case(GLUT_KEY_DOWN):
        cam.position += 0.1f * cam.position;
        break;
	  case(GLUT_KEY_LEFT):
        cam.position -= 0.1f * cam.right;
        cam.view     = glm::normalize(-cam.position);
        cam.right    = glm::normalize(glm::cross(cam.view, cam.up));
        cam.up       = glm::normalize(glm::cross(cam.right, cam.view));
        break;
	  case(GLUT_KEY_RIGHT):
        cam.position += 0.1f * cam.right;
        cam.view     = glm::normalize(-cam.position);
        cam.right    = glm::normalize(glm::cross(cam.view, cam.up));
        cam.up       = glm::normalize(glm::cross(cam.right, cam.view));
        break;
    }
    return;
  }

  // Mouse interaction code, referring to the glut example code http://graphics.stanford.edu/courses/cs248-01/OpenGLHelpSession/code_example.html
  void mouseClick(int button, int state, int x, int y) {
    if (state == GLUT_LEFT_BUTTON) {
      buttonDown = (state == GLUT_DOWN) ? true : false;
      lastX = x;
      lastY = y;
    }
  }

  glm::vec4 pos;
  void mouseMotion(int x, int y) {
    if ( x < 0 || y < 0 || x > width || y > height)
      return;
    // Using the position of the mouse to change the two rotation angles of the camera
    if (buttonDown) {
      float roll  = (x - lastX) / 5.0f;
      float pitch = - (y - lastY) / 5.0f;
      // Rotate around up axis
	  cam.right = glm::normalize(glm::cross(cam.view, cam.up));
	  cam.up    = glm::normalize(glm::cross(cam.right, cam.view));
      pos = glm::rotate(glm::mat4(1.0f), roll, cam.up) * glm::vec4(cam.position, 1.0f);
      cam.position = glm::vec3(pos.x, pos.y, pos.z) / pos.w;
	  cam.view     = glm::normalize(-cam.position);
      // Rotate around right axis
	  cam.right = glm::normalize(glm::cross(cam.view, cam.up));
	  cam.up    = glm::normalize(glm::cross(cam.right, cam.view));
      pos = glm::rotate(glm::mat4(1.0f), pitch, cam.right) * glm::vec4(cam.position, 1.0f);
      cam.position = glm::vec3(pos.x, pos.y, pos.z) / pos.w;
	  cam.view     = glm::normalize(-cam.position);
	}
    lastX = x;
    lastY = y;
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
