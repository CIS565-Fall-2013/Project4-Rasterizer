#ifndef _UTIL_H
#define _UTIL_H

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


#define cudaErrorCheck( errNo ) checkError( (errNo), __FILE__, __LINE__ )

inline void checkError( cudaError_t err, const char* const filename, const int line  )
{
    if( err != cudaSuccess )
    {
        std::cerr<<"CUDA ERROR: "<<filename<<" line "<<line<<"\n";
        std::cerr<<cudaGetErrorString( err )<<"\n";
        exit(1);
    }
}



#endif