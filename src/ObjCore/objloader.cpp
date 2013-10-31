//OBJCORE- A Obj Mesh Library by Yining Karl Li
//This file is part of OBJCORE, Coyright (c) 2012 Yining Karl Li

#include "objloader.h"
#include <iomanip>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string.h>
#include "../glm/glm.hpp" 
#include "../glm.h"
using namespace std;

ObjModel* objLoader::load( string &filename )
{
    int offset = 0;
    ObjModel* newMesh;
    GLMmodel *model = glmReadOBJ( (char*)filename.c_str() );
    if( model == NULL )
    {
        newMesh = NULL;
        return NULL;
    }
    glmUnitize( model );

    newMesh = new ObjModel();
    newMesh->numVert = model->numvertices;
    newMesh->numIdx = model->numtriangles * 3;
    newMesh->numNrml = model->numnormals;
    newMesh->numTxcoord = model->numtexcoords;
   
    newMesh->vbo = new float[ newMesh->numVert * 3 ];
    newMesh->ibo = new int[ newMesh->numIdx ];

    newMesh->nbo = new float[ newMesh->numVert * 3 ];
    newMesh->tbo = new float[ newMesh->numVert * 2];

    GLMgroup* group = model->groups;
   
    //this loop copy index data
    //and copy normal and texture data in the order indicated by the index data
    while(group)
    {
        GLMtriangle* tri;
        for( int i = 0; i < group->numtriangles; ++i )
        {
            tri = &model->triangles[group->triangles[i]];

            //First we copy the index data
            newMesh->ibo[3*offset] = tri->vindices[0]-1;
            newMesh->ibo[3*offset+1] = tri->vindices[1]-1;
            newMesh->ibo[3*offset+2] = tri->vindices[2]-1;

            //Then copy the normal data in the order determined by the index data
            if( newMesh->numNrml )
            {
                memcpy( &newMesh->nbo[ 3*newMesh->ibo[3*offset] ], &model->normals[ 3*(tri->nindices[0]) ], 3*sizeof(float) );
                memcpy( &newMesh->nbo[ 3*newMesh->ibo[3*offset+1] ], &model->normals[ 3*(tri->nindices[1]) ], 3*sizeof(float) );
                memcpy( &newMesh->nbo[ 3*newMesh->ibo[3*offset+2] ], &model->normals[ 3*(tri->nindices[2]) ], 3*sizeof(float) );
            }
            //And copy the texture coordinate data
            if( newMesh->numTxcoord )
            {
                memcpy( &newMesh->tbo[ 2*newMesh->ibo[3*offset] ], &model->texcoords[ 3*(tri->tindices[0]) ], 2*sizeof(float) );
                memcpy( &newMesh->tbo[ 2*newMesh->ibo[3*offset+1] ], &model->texcoords[ 3*(tri->tindices[1]) ], 2*sizeof(float) );
                memcpy( &newMesh->tbo[ 2*newMesh->ibo[3*offset+2] ], &model->texcoords[ 3*(tri->tindices[2]) ], 2*sizeof(float) );
            }
            offset += 1;
        }
        group = group->next;
    }

    //copy vertex data
    for( int i = 0; i < newMesh->numVert; ++i )
    {
        memcpy( &newMesh->vbo[3*i], &model->vertices[3*(i+1)], sizeof(float)*3);
        
    }


    glmDelete( model );
    cout<<newMesh->numVert<<" vertices loaded with "<<(newMesh->numIdx/3)<<" faces.\n";

    return newMesh;
}

objLoader::~objLoader(){
}

//obj* objLoader::getMesh(){
//	return geomesh;
//}
