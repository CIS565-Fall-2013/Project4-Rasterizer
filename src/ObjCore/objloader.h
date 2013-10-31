//OBJCORE- A Obj Mesh Library by Yining Karl Li
//This file is part of OBJCORE, Coyright (c) 2012 Yining Karl Li

#ifndef OBJLOADER
#define OBJLOADER

#include <stdlib.h>
#include "obj.h"

using namespace std;



class objLoader{
private:
	obj* geomesh;
	bool hasTexture;
public:
	objLoader(string, obj*);
	~objLoader();
    
    //------------------------
    //-------GETTERS----------
    //------------------------
    
	obj* getMesh();
	bool hasTextureFun(){return hasTexture;};
};

#endif