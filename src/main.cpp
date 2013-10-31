// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include "main.h"

using namespace utilityCore;
//-------------------------------
//-------------MAIN--------------
//-------------------------------
bool pause = false;
bool output = false;
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
	  hasTexture = loader->hasTextureFun();
	  std::cout<<"hasTexture: "<<hasTexture<<std::endl;
      mesh->buildVBOs();
      delete loader;
      loadedScene = true;
    }
  }

  if(!loadedScene){
    cout << "Usage: mesh=[obj file]" << endl;
    return 0;
  }

  if(hasTexture)
  {
	  BMP img;
	  char* filename = "../../src/cow.bmp";
	  img.ReadFromFile(filename);
	  t_width = img.TellWidth();
	  t_height = img.TellHeight();
	  textureimg = new glm::vec3[t_width * t_height];
	  for(int i = 0;i<t_width;i++)
	  {

		  for(int j = 0;j<t_height;j++)
		  {
			  float r = img(i,j)->Red/255.0f;
			  float g = img(i,j)->Green/255.0f;
			  float b = img(i,j)->Blue/255.0f;
			  textureimg[i+j*t_height] = glm::vec3(r,g,b);
		  }
	  }
  }
  else
	  getCheckerBox();

  frame = 0;
  seconds = time (NULL);
  fpstracker = 0;

  // matrix setup
  projectionM = glm::perspective(fovy,float(width)/float(height),zNear,zFar);
  viewM = glm::lookAt(cameraPostion,center,up);//eye,center,up
  modelM = glm::mat4(1.0);
  //image initialization
  images = new glm::vec3[width * height];
  for(int i = 0;i<width;++i)
	  for(int j = 0;j<height; ++j)
	  {
		  images[i+width*j] = glm::vec3(0,0,0);
	  }

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

  //ADD matrix setup	

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

    glutMainLoop();
  #endif
  kernelCleanup();
  return 0;
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda(){
	if(pause == true) return;
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

  //
  vector<vector<int>>* faceTexture = mesh->getFaceTextures();
  vector<glm::vec4>* textCoord = mesh->getTextureCoords();
  vtbosize = faceTexture->size()*3;
  vtbo = new glm::vec4[vtbosize];
  //std::cout<<faceTexture->size()<<std::endl;
  for(int i = 0; i < faceTexture->size();i++)
  {	  
	  vector<int> facetext = faceTexture->at(i);
	  vtbo[i*3] = textCoord->at(facetext[0]);
	  vtbo[i*3+1] = textCoord->at(facetext[1]);
	  vtbo[i*3+2] = textCoord->at(facetext[2]);
	  //std::cout<<facetext[0]<<" ";
  }

  modelM = glm::mat4(1.0);
  //modelM = glm::rotate(modelM,29.0f,glm::vec3(0,1,0));
  modelM = glm::rotate(modelM,(float)frame,glm::vec3(0,1,0));
  //modelM = glm::rotate(modelM,230.0f,glm::vec3(0,1,0));
  

  cudaGLMapBufferObject((void**)&dptr, pbo);
  cudaRasterizeCore(dptr, glm::vec2(width, height), frame, vbo, vbosize, cbo, cbosize, ibo, ibosize,
	  nbo,nbosize,
	  modelM,viewM,projectionM
	  ,images
	  ,cameraPostion
	  ,hasTexture
	  ,vtbo,vtbosize,textureimg,glm::vec2(t_width,t_height));

  cudaGLUnmapBufferObject(pbo);

  vbo = NULL;
  cbo = NULL;
  ibo = NULL;
  nbo = NULL;
  vtbo = NULL;

  if(output == true)
  {

	  image outputImage(width, height);
	  for(int x=0; x<width; x++){
		  for(int y=0; y<height; y++){
			  int index = x + (y * width);
			  outputImage.writePixelRGB(width-1-x,y,images[index]);
		  }
	  }
	  gammaSettings gamma;
	  gamma.applyGamma = true;
	  gamma.gamma = 1.0;
	  gamma.divisor = 1.0;
	  outputImage.setGammaSettings(gamma);
	  string filename = "screenshot.png";
	  string s;
	  stringstream out;
	  out << frame;
	  s = out.str();	  
	  utilityCore::replaceString(filename, ".bmp", "."+s+".bmp");
	  utilityCore::replaceString(filename, ".png", "."+s+".png");
	  outputImage.saveImageRGB(filename);
	  cout << "Saved frame " << s << " to " << filename << endl;
	  output = false;
  }

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

    string title = "CIS565 Rasterizer | "+ utilityCore::convertIntToString((int)fps) + "FPS | " + utilityCore::convertIntToString(frame) + "FRAME";
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
	 std::cout<<key<<std::endl;
    switch (key) 
    {
       case(27):
         shut_down(1);    
         break;
	   case('p'):
		   pause = !pause;
		   break;
	   case('c'):
		   output = true;
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


void getCheckerBox()
{
	textureimg = new glm::vec3[t_width * t_height];
	for(int i = 0;i<t_width;i++)
	{
		for(int j = 0;j<t_height;j++)
		{			
			/*if(j%40 <= 20 && i%40 <=20)
			textureimg[i+j*t_width] = glm::vec3(1,0,0);			
			else
			textureimg[i+j*t_width] = glm::vec3(1,1,0);*/
			if(i%40 <=20)
				textureimg[i+j*t_width] = glm::vec3(1,1,1);			
			else
				textureimg[i+j*t_width] = glm::vec3(0,0,0);
		}
	}
}