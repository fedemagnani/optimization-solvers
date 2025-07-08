#!/bin/bash

# Build script for WASM compilation
echo "Building optimization-solvers for WebAssembly..."

# Run the WASM build script
cd wasm && ./build-wasm.sh

echo "WASM build complete! Files are in the 'wasm/pkg' directory." 