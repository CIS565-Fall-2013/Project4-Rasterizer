-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2013
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
NOTE:
-------------------------------------------------------------------------------
This project requires an NVIDIA graphics card with CUDA capability! 
Any card with CUDA compute capability 1.1 or higher will work fine for this project.

-------------------------------------------------------------------------------
INTRODUCTION:
-------------------------------------------------------------------------------
This project is a CUDA based implementation of a simplified rasterization pipeline.

Very specifically, the following stages of the rasteriztion pipeline have been implemented,

- Vertex shader
- Primitive assembly
- Rasterization
- Fragment shader

-------------------------------------------------------------------------------
MY PIPELINE IMPLEMENTATION:
-------------------------------------------------------------------------------

![alt tag](https://raw.github.com/vimanyu/Project4-Rasterizer/master/doc_images/pipeline.png)

-------------------------------------------------------------------------------
FEATURES:
-------------------------------------------------------------------------------

* A basic rasterization pipeline from start to finish for obj models
* Parallel rasterization of each primitive
* Smooth interpolation of normals and colors using barycentric co-ordinates
* Backface culling in primitive assembly to discard back facing polygons. This is implemented using thrust's stream compaction.
* Interactive mouse movement for panning, zooming and rotating

-------------------------------------------------------------------------------
RESULTS:
-------------------------------------------------------------------------------
*COLOR INTERPOLATION* : Smooth linear interpolation of colors for each triangle based on barycentric co-ordinates
![alt tag](https://raw.github.com/vimanyu/Project4-Rasterizer/master/renders/cow_color_interp.png)

*NORMALS*: World space normals
![alt tag](https://raw.github.com/vimanyu/Project4-Rasterizer/master/renders/cow_normals.png)

*Z-DEPTH*: Z-depth in camera space
![alt tag](https://raw.github.com/vimanyu/Project4-Rasterizer/master/renders/dragon_depth.png)

*LAMBERT DIFFUSE LIGHING*: Simple lambertian shading
![alt tag](https://raw.github.com/vimanyu/Project4-Rasterizer/master/renders/dragon.png)

*BLINN PHONG LIGHTING*: Diffuse lighting with specular highlights
![alt tag](https://raw.github.com/vimanyu/Project4-Rasterizer/master/renders/buddha_spec.png)

-------------------------------------------------------------------------------
VIDEO
-------------------------------------------------------------------------------
The following is a video of the rasterizer tool in action

[![ScreenShot](https://raw.github.com/vimanyu/Project4-Rasterizer/master/doc_images/rasterizer_video_screenshot.png)](http://www.youtube.com/watch?v=s8ehsuIoL_U)

-------------------------------------------------------------------------------
BUILDING AND RUNNING CODE
-------------------------------------------------------------------------------
The code has been tested on Visual Studio 2012/2010 and cuda 5.5 on a laptop with compute capability 1.0 as well as 3.0.

Keyboard bindings for interactivity:

Key|Action
---|---
'c'| View color interpolation
'n'| View world space normals
'z'| View z-depth
'l'| Diffuse lighting
's'| Specular lighting
'r'| Reset camera

Mouse button| Action
LMB| Rotate 
MMB| Pan
RMB| Zoom

Apart from this you might encounter objs which have been built assuming z-axis as the up axis.
In this code, y-axis is the up-axis.

To correct this, you can simply, set the model matrix accordingly.

```
//Control the number of bodies in the simulation
//main.cpp (line 96)
	glm::vec3 myRotationAxis(1.0f, 0.0f, 0.0f);
	//Change the 0.0f to -90f incase you need to align with y-axis
	glm::mat4 rotationMat = glm::rotate( 0.0f, myRotationAxis )
```

If you need to render with a different color, you can alter,
```
//Control the number of bodies in the simulation
//rasterize_kernels.cu (line 345)
glm::vec3 col = glm::vec3(0.42f,0.35f,0.80f);
```
For now, these values are hard-coded but I will definitely be working on a future version that can read in a config file.

-------------------------------------------------------------------------------
PERFORMANCE ANALYSIS
-------------------------------------------------------------------------------
* Use of any third-party code must be approved by asking on Piazza.  If it is approved, all students are welcome to use it.  Generally, we approve use of third-party code that is not a core part of the project.  For example, for the ray tracer, we would approve using a third-party library for loading models, but would not approve copying and pasting a CUDA function for doing refraction.
* Third-party code must be credited in README.md.
* Using third-party code without its approval, including using another student's code, is an academic integrity violation, and will result in you receiving an F for the semester.

-------------------------------------------------------------------------------
SELF-GRADING
-------------------------------------------------------------------------------
* On the submission date, email your grade, on a scale of 0 to 100, to Liam, liamboone+cis565@gmail.edu, with a one paragraph explanation.  Be concise and realistic.  Recall that we reserve 30 points as a sanity check to adjust your grade.  Your actual grade will be (0.7 * your grade) + (0.3 * our grade).  We hope to only use this in extreme cases when your grade does not realistically reflect your work - it is either too high or too low.  In most cases, we plan to give you the exact grade you suggest.
* Projects are not weighted evenly, e.g., Project 0 doesn't count as much as the path tracer.  We will determine the weighting at the end of the semester based on the size of each project.

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

