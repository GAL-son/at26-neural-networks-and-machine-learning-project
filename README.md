# Neural Anti-Aliaser

## Project purpose
This project is created for the `Neural Networks and Machine Learning` corse at `Akdemia Tarnowska`

Purpose of this project is to create a neural network that can be used to anti-alias the edges of the objects in the scene.

## Project architecture
Project consists of two main elements:
* Renderer
* Training suite

**Renderer** is a simple vulkan 3D renderer that is used to generate training data for the neural network.
This data consists of 5 buffers for each frame:
* 4 Aliased buffers:
    * G-buffer (Color / Albedo)
    * Normal buffer
    * Depth buffer
    * Motion vectors buffer
* 1 Unaliased buffer:
    * Rendered frame (G-buffer) with SSAA x16 as ground truth

**Training suite** is a full training suite for CNN, with data ingestion, analysys, training, testing and visualization.

