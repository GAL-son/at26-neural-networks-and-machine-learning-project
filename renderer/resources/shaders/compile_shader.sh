#!/bin/bash

# Ensure the target directory exists
mkdir -p target

echo "Starting shader compilation..."

# Loop through all files in the source directory
for file in source/*; do
    if [ -f "$file" ]; then
        # Extract the filename (e.g., composite.vert)
        filename=$(basename "$file")
        
        # Append .spv to the exact filename (e.g., composite.vert.spv)
        outname="${filename}.spv"
        
        # Compile using glslc
        glslc "$file" -o "target/$outname"
        
        if [ $? -eq 0 ]; then
            echo "[SUCCESS] Compiled: $filename -> target/$outname"
        else
            echo "[ERROR] Failed to compile: $filename"
        fi
    fi
done

echo "Shader compilation finished!"