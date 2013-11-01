-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2013
-------------------------------------------------------------------------------
Due Thursday 10/31/2012
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
INTRODUCTION:
-------------------------------------------------------------------------------
For this project, I implemented a GPU rasterizer that reads and rasterizes an .obj
file. The pipeline is as follows:

1. Transform the .obj vertices from model space to screen space using a series of 
matrix transformations. 

2. Do primitive assembly, i.e. take three vertices at a time and group them into 
triangles. Calculate normals (I keep them in camera space).

3. Rasterize per primitive -- for each primitive, find out which fragments it contains. 

4. Fragment shading -- Do diffuse shading for fragments that are contained by a primitive.

I implemented the following extra features:

* Color interpolation between points on a primitive

* Back-face culling -- I throw away all triangles that don't face the eye

* Mouse based interactive camera support

-------------------------------------------------------------------------------
Images:
-------------------------------------------------------------------------------


-------------------------------------------------------------------------------
README
-------------------------------------------------------------------------------
All students must replace or augment the contents of this Readme.md in a clear 
manner with the following:

* A brief description of the project and the specific features you implemented.
* At least one screenshot of your project running.
* A 30 second or longer video of your project running.  To create the video you
  can use http://www.microsoft.com/expression/products/Encoder4_Overview.aspx 
* A performance evaluation (described in detail below).

-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
The performance evaluation is where you will investigate how to make your CUDA
programs more efficient using the skills you've learned in class. You must have
performed at least one experiment on your code to investigate the positive or
negative effects on performance. 

We encourage you to get creative with your tweaks. Consider places in your code
that could be considered bottlenecks and try to improve them. 

Each student should provide no more than a one page summary of their
optimizations along with tables and or graphs to visually explain any
performance differences.

---
SUBMISSION
---
As with the previous project, you should fork this project and work inside of
your fork. Upon completion, commit your finished project back to your fork, and
make a pull request to the master repository.  You should include a README.md
file in the root directory detailing the following

* A brief description of the project and specific features you implemented
* At least one screenshot of your project running.
* A link to a video of your raytracer running.
* Instructions for building and running your project if they differ from the
  base code.
* A performance writeup as detailed above.
* A list of all third-party code used.
* This Readme file edited as described above in the README section.

