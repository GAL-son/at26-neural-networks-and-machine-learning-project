# Neural Anti-Aliaser

## Project purpose
This project is created for the `Neural Networks and Machine Learning` corse at `Akdemia Tarnowska`

Purpose of this project is to create a neural network that can be used to anti-alias the edges of the objects in the scene.

## Project architecture
Project consists of two main elements:
* Renderer
* Training suite

**Renderer** is a simple vulkan 3D renderer that is used to generate training data for the neural network.
For each scene it records a camera animation (600 frames by default) as one multi-layer `.exr` per frame (1920x1080).
Each frame contains 5 buffers:
* 4 aliased buffers (1 sample per pixel):
    * Albedo (unlit, linear RGB)
    * Normals (world space)
    * Depth (linear view-space Z)
    * Motion vectors (screen space, UV units)
* 1 anti-aliased buffer:
    * Albedo rendered with SSAA x16 (4x4 box filter) as ground truth

The exact file layout, channel names, value ranges and conventions are in [renderer/OUTPUT_FORMAT.md](renderer/OUTPUT_FORMAT.md).

**Training suite** is a full training suite for CNN, with data ingestion, analysis, training, testing and visualization.

