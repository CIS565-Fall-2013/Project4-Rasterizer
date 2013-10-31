-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2013
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
NOTE:
-------------------------------------------------------------------------------
NVidia Cards with compute 3.0 supported

---
Screenshots
---

*Diffuse Shading: 871k, 5k triangles*

![Screen](renders/dragon_diffuse.png)
![Screen](renders/cow_diffuse.png)

*Depth Shading: 100k triangles*

![Screen](renders/dragon.png)

*Normal Shading: 871k triangles*

![Screen](renders/dragon_Normal.png)

*Color Per Triangle: 6k triangles*

![Screen](renders/suzanne_col_per_tri.png)

*Color Per Vertex (interpolated Colors): 6k triangles*

![Screen](renders/suzanne_col_per_vert.png)

--- 
Features
---

Extra Features Impemented:
* Point Mode: Render using only points. Key 'p' to switch on and off
* Mouse Interaction: use mouse to pan and zoom
* Correct Color Interpolation: Using barycentric coordinates to interpolate colors
* Back Face Culling: Using thrust to remove faces with normals point away from the eye
* Back Face Ignoring: After calculating fragment properties, do not perform depth test in rasterization stage
* Dynamic Parallelization: use compute 3.5 to launch a kernel from within a kernel

Basic features implemented:
* Vertex Shading: Vertex transformations
* Primitive Assembly: Takes normal and color buffers and arranges them
* Rasterization: Per primitive rasterization
* Depth Testing: At rasterization stage
* Fragment shader: for different lighting techniques

---
Performance Analysis
---