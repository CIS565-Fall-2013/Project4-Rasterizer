// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include "main.h"

//-------------------------------
//-------------MAIN--------------
//-------------------------------
typedef int BOOL;
  #define TRUE 1
  #define FALSE 0
  // Set up mouse call back
  static BOOL g_bButton1Down = FALSE;
  static BOOL g_bButton2Down = FALSE;
  static BOOL g_bButton3Down = FALSE;
  int bottonMask = 0;
  int mouse_old_x ;
  int mouse_old_y ;
  float spherex =0.0f,spherey = 0.0f,sphereRadius=10.0f ;
  glm::vec3 sphereCenter = glm::vec3(0,0,0) ;
  float r_head= 0.0 , r_pitch = 0.0;


glm::mat4 Projection;
glm::vec3 eye = glm::vec3(0,0,10) ;
//glm::vec3 viewDir = glm::vec3(0,-1,0); 
glm::vec3 up = glm::vec3(0,1,0); 
glm::mat4 View ;
glm::mat4 Model ;
glm::mat4 MVP ;
glm::mat4 MV ;




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
	int a;
	cin >> a ; 
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
	  glutMouseFunc(MouseButton);
      glutMotionFunc(MouseMotion);
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

// Projection matrix : 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
Projection = glm::perspective(45.0f, 1.0f, 0.1f, 100.0f);
// Camera matrixglm::vec3(0,0,10)

View       = glm::lookAt(
    eye, // Camera is at (4,3,3), in World Space3,1,10
    sphereCenter, // and looks at the origin
    up  // Head is up (set to 0,-1,0 to look upside-down)
);
// Model matrix : an identity matrix (model will be at the origin)
Model      = glm::mat4(utilityCore::buildTransformationMatrix(glm::vec3(0,0,0), glm::vec3(0,180,0), glm::vec3(5,5,5)));//glm::mat4(1.0f);  // Changes for each model !
//Model      = glm::mat4((1.0));//utilityCore::buildTransformationMatrix(glm::vec3(0,0,0), glm::vec3(0,0,0), glm::vec3(1,1,1)));
  //glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
// Our ModelViewProjection : multiplication of our 3 matrices
 MVP        = Projection * View * Model; // Remember, matrix multiplication is the other way around
 MV         = View * Model; 
//cudaMat4* MVPc = &utilityCore::glmMat4ToCudaMat4(MVP);

  cudaGLMapBufferObject((void**)&dptr, pbo);
  cudaRasterizeCore(dptr, glm::vec2(width, height), frame, vbo, vbosize, cbo, cbosize, ibo, ibosize,MVP,MV,eye);
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

  void MouseButton(int button, int state, int x, int y)
	{
		mouse_old_x = x;
		mouse_old_y = y;
	  // Respond to mouse button presses.
	  // If button1 pressed, mark this state so we know in motion function.
	  if (button == GLUT_LEFT_BUTTON)
		{
		  g_bButton1Down = (state == GLUT_DOWN) ? TRUE : FALSE;
		  bottonMask= 1;
		  //g_yClick = y - 3 * g_fViewDistance;
		}
	   else if (button == GLUT_RIGHT_BUTTON)
		{
		  g_bButton2Down = (state == GLUT_DOWN) ? TRUE : FALSE;
		  bottonMask= 2;
		  //g_yClick = y - 3 * g_fViewDistance;
		}
	    else if (button == GLUT_MIDDLE_BUTTON)
		{
		  g_bButton3Down = (state == GLUT_DOWN) ? TRUE : FALSE;
		  bottonMask= 3;
		  //g_yClick = y - 3 * g_fViewDistance;
		}
		else
		{
		   bottonMask= 0;
		}
		std::cout << button << ", " << state << std::endl;
	}

	void MouseMotion(int x, int y)
	{
		
		float dx, dy;
		dx = (float)(x - mouse_old_x);
		dy = (float)(y - mouse_old_y);
		//r_head = 0.0f ; r_pitch = 0.0f ;
		if( g_bButton1Down || g_bButton2Down || g_bButton3Down)
		{
		
		// If button1 pressed, zoom in/out if mouse is moved up/down.
		if (g_bButton1Down)  //Left mouse click drag 
		{
			spherex += dy * 0.2f;
			spherey += dx * 0.2f;
		

			r_head = glm::radians(spherex);
			r_pitch = glm::radians(spherey);
			//renderCam->positions[targetFrame].x = (sphereCenter.x + sphereRadius * glm::sin(r_head) * glm::cos(r_pitch));
			//renderCam->views[targetFrame] = glm::normalize(glm::vec3(sphereCenter - renderCam->positions[targetFrame]));
			
		}
		if (g_bButton2Down)  //Right mouse click drag 
		{
			sphereRadius-= dy * 0.01f;
		
			//int sign = 0 ;
		   // (dx>0)? sign=1 :sign = -1 ; 
			//renderCam->positions[targetFrame].z += sign * 0.2f; //(sphereCenter.z + sphereRadius * glm::cos(r_head) );
			//renderCam->views[targetFrame] = glm::normalize(glm::vec3(sphereCenter - renderCam->positions[targetFrame]));
		}
		if (g_bButton3Down) // Middle mouse click drag 
		{
			
			glm::vec3 vdir(sphereCenter -  eye);
			glm::vec3 u(glm::normalize(glm::cross(glm::normalize(vdir), up)));
			glm::vec3 v(glm::normalize(glm::cross(u, glm::normalize(vdir))));

			sphereCenter += 0.01f * (dy * v - dx * u);
			
			//renderCam->positions[targetFrame].x += dx ;//(sphereCenter.x + sphereRadius * glm::sin(r_head) * glm::cos(r_pitch));
			//renderCam->positions[targetFrame].y =(sphereCenter.y + sphereRadius * glm::sin(r_head) * glm::sin(r_pitch));
			//renderCam->positions[targetFrame].z =sphereCenter.z + sphereRadius * glm::cos(r_head) );
			//renderCam->views[targetFrame] = 	glm::normalize(glm::vec3(sphereCenter - renderCam->positions[targetFrame]));
			//renderCam->positions[0] = renderCam->positions[0] + 2.0f;
		}
		//glutPostRedisplay();

		eye.x = (sphereCenter.x + sphereRadius* glm::sin(r_head) * glm::cos(r_pitch));
		eye.y = (sphereCenter.y + sphereRadius  * glm::sin(r_head) * glm::sin(r_pitch));
		eye.z = (sphereCenter.z + sphereRadius * glm::cos(r_head) );

		//glm::vec3 viewDir = 	glm::normalize(glm::vec3(sphereCenter - eye));
			

			//cam_pos.x = lookat.x + eye_distance * glm::cos(r_head) * glm::cos(r_pitch);
   // cam_pos.y = lookat.y + eye_distance * glm::sin(r_head);
   // cam_pos.z = lookat.z + eye_distance * glm::cos(r_head) * glm::sin(r_pitch);


	/*	renderCam->positions[targetFrame].x = (sphereCenter.x + sphereRadius * glm::sin(r_head) * glm::cos(r_pitch));
		renderCam->positions[targetFrame].y = (sphereCenter.y + sphereRadius * glm::sin(r_head) * glm::sin(r_pitch));
		renderCam->positions[targetFrame].z = (sphereCenter.z + sphereRadius * glm::cos(r_head) );	*/


		mouse_old_x = x;
		mouse_old_y = y;
		}
	}

	void mouseWheel(int button, int dir, int x, int y)
{
    if (dir > 0)
    {
        // Zoom in
    }
    else
    {
        // Zoom out
    }

    return;
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
