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

	//Setup uniform variables

	//TODO: Camera movable
	u_pipelineOpts.fShaderProgram = BLINN_PHONG_SHADING;
	u_pipelineOpts.showTriangleColors = false;
	u_pipelineOpts.backfaceCulling = false;

	u_variables.blinnPhongParams  = glm::vec3(0.1,0.6,0.3);//Ambient, diffuse, specular.
	u_variables.lightPos = glm::vec4(-1.0f,1.0f,10.0f,1.0f);
	u_variables.lightColor = glm::vec3(1.0f,1.0f,1.0f);
	u_variables.diffuseColor = glm::vec3(0.8,0.8,0.8);
	u_variables.specularColor = glm::vec3(1.0,1.0,1.0);
	u_variables.shininess = 8.0f;

	u_variables.viewTransform = glm::lookAt(glm::vec3(1.0,0.0,1.0), glm::vec3(0,0,0), glm::vec3(0.0,0.0,-1.0));
	u_variables.perspectiveTransform = glm::perspective(60.0f, float(width)/float(height), 0.1f, 5.0f);

	glm::mat4 scale = glm::mat4(1.0f);
	scale[3][3] = 1.0f;
	u_variables.modelTransform = glm::rotate(scale, 90.0f, glm::vec3(1.0f,0.0f,0.0f));

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
	glutMouseFunc(mouse_click);
	glutMotionFunc(mouse_move);

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
	nbosize = mesh->getNBOsize();

	float newcbo[] = {1.0, 0.0, 0.0, 
		0.0, 1.0, 0.0, 
		0.0, 0.0, 1.0};
	if(!u_pipelineOpts.showTriangleColors){

		glm::vec3 color = mesh->getColor();
		for(int i = 0; i < 3; i++)
		{
			newcbo[3*i+0] = color.x;
			newcbo[3*i+1] = color.y;
			newcbo[3*i+2] = color.z;
		}
	}
	cbo = newcbo;
	cbosize = 9;

	ibo = mesh->getIBO();
	ibosize = mesh->getIBOsize();


	cudaGLMapBufferObject((void**)&dptr, pbo);
	cudaRasterizeCore(dptr, glm::vec2(width, height), frame, vbo, vbosize, nbo, nbosize, cbo, cbosize, ibo, ibosize, u_variables, u_pipelineOpts);
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

	glm::vec3 Right, Look;
	switch (key) 
	{
	case(27)://ESC
		shut_down(0);    
		break;
	case 'b':
		u_pipelineOpts.backfaceCulling = !u_pipelineOpts.backfaceCulling;
		break;
	case '1':
		u_pipelineOpts.fShaderProgram = AMBIENT_LIGHTING;
		break;
	case '2':
		u_pipelineOpts.fShaderProgram = DEPTH_SHADING;
		break;
	case '3':
		u_pipelineOpts.fShaderProgram = NORMAL_SHADING;
		break;
	case '4':
		u_pipelineOpts.fShaderProgram = BLINN_PHONG_SHADING;
		break;
	case 'c':
		u_pipelineOpts.showTriangleColors = !u_pipelineOpts.showTriangleColors;
		break;
	case 'a':
		u_variables.viewTransform = glm::rotate(u_variables.viewTransform, 5.0f, glm::vec3(0.0f,0.0f,1.0f));
		break;
	case 'd':
		u_variables.viewTransform = glm::rotate(u_variables.viewTransform, -5.0f, glm::vec3(0.0f,0.0f,1.0f));
		break;
	case 's':
		Right = glm::inverse(glm::mat3(u_variables.viewTransform))*glm::vec3(1.0f,0.0f,0.0f);
		u_variables.viewTransform = glm::rotate(u_variables.viewTransform, 5.0f,Right);
		break;
	case 'w':
		Right = glm::inverse(glm::mat3(u_variables.viewTransform))*glm::vec3(1.0f,0.0f,0.0f);
		u_variables.viewTransform = glm::rotate(u_variables.viewTransform, -5.0f,Right);
		break;
	case 'q':
		Look = glm::inverse(glm::mat3(u_variables.viewTransform))*glm::vec3(0.0f,0.0f,-1.0f);
		u_variables.viewTransform = glm::rotate(u_variables.viewTransform, 5.0f,Look);
		break;
	case 'e':
		Look = glm::inverse(glm::mat3(u_variables.viewTransform))*glm::vec3(0.0f,0.0f,-1.0f);
		u_variables.viewTransform = glm::rotate(u_variables.viewTransform, -5.0f,Look);
		break;
	case 'x':
		Look = glm::inverse(glm::mat3(u_variables.viewTransform))*glm::vec3(0.0f,0.0f,-1.0f);
		u_variables.viewTransform = glm::translate(u_variables.viewTransform, 0.05f*Look);
		break;		
	case 'z':
		Look = glm::inverse(glm::mat3(u_variables.viewTransform))*glm::vec3(0.0f,0.0f,-1.0f);
		u_variables.viewTransform = glm::translate(u_variables.viewTransform, -0.05f*Look);
		break;


	}
}


//MOUSE STUFF
bool dragging = false;
bool rightclick = false;
int drag_x_last = -1;
int drag_y_last = -1;
void mouse_click(int button, int state, int x, int y) {
	if(button == GLUT_LEFT_BUTTON) {
		if(state == GLUT_DOWN) {
			dragging = true;
			drag_x_last = x;
			drag_y_last = y;
		}
		else{
			dragging = false;
		}
	}
	if(button == GLUT_RIGHT_BUTTON) {
		if(state == GLUT_DOWN)
		{
			rightclick = true;
		}else{
			rightclick = false;
		}
	}
}

void mouse_move(int x, int y) {
	if(dragging) {
		glm::mat3 inv_View = glm::inverse(glm::mat3(u_variables.viewTransform));
		glm::vec3 Up = inv_View*glm::vec3(0.0f,1.0f,0.0f);
		glm::vec3 Right = inv_View*glm::vec3(1.0f,0.0f,0.0f);
		glm::vec3 Look = inv_View*glm::vec3(0.0f,0.0f,-1.0f);

		float delX = x-drag_x_last;
		float delY = y-drag_y_last;
		float rotSpeed = 0.5f;
		if(rightclick)
		{
			//Operations about view direction.
			//Rotate about view
			u_variables.viewTransform = glm::rotate(u_variables.viewTransform, rotSpeed*delX, Look);
			//Zoom
			//u_variables.viewTransform = glm::translate(u_variables.viewTransform, delY*0.005f*Look);

		}else{
			//Simple rotation
			u_variables.viewTransform = glm::rotate(u_variables.viewTransform, -rotSpeed*delX, Up);
			u_variables.viewTransform = glm::rotate(u_variables.viewTransform, -rotSpeed*delY, Right);
		}
		drag_x_last = x;
		drag_y_last = y;
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
