-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------
Yuqin Shao
-------------------------------------------------------------------------------

Video Demo 
-------------------------------------------------------------------------------
http://youtu.be/4INouIToxjU
-------------------------------------------------------------------------------

Features
-------------------------------------------------------------------------------
[Basic Feature]
* Vertex Shading
* Primitive Assembly with support for triangle VBOs/IBOs
* Perspective Transformation
* Rasterization through either a scanline or a tiled approach
* Fragment Shading
* A depth buffer for storing and depth testing fragments
* Fragment to framebuffer writing
* A simple lighting/shading scheme, such as Lambert or Blinn-Phong, implemented in the fragment shader

[Additional]
* Back face culling (in primitive Assembly step)
* Anti-aliasing ( super sample one)
* Texture mapping
* Simple animation ( rotate the model around y axis based on the frame number)

ScreenShots
-------------------------------------------------------------------------------
![Alt test] (/renders/texture1.png " ")
![Alt test] (/renders/texture0.png " ")
![Alt test] (/renders/texture2.png " ")

More
-------------------------------------------------------------------------------
Color Interpolation
![Alt test] (/renders/ColorIntepolation.png " ")

Depth field render
![Alt test] (/renders/DepthRender.png "")

Anti-alising

![Alt test] (/renders/antiailising.jpg " ")


-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
* I got 18 fps after implementing the basic features.
* The speed inproves to 26 fps when back face culling was added
* Anti-aliasing decreases the speed to 24 fps
* Texture mapping decreases the speed to 20 fps

-------------------------------------------------------------------------------
THIRD PARTY CODE POLICY
-------------------------------------------------------------------------------
I used the EASYBMP library for the texture image loading.
I also used the image.h image.cpp files from last project for the saving of screenshots in run time.

