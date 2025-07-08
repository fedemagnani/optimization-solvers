#!/bin/bash

# Build script for WASM compilation

echo "Building optimization-solvers for WebAssembly..."

# Install wasm-pack if not already installed
if ! command -v wasm-pack &> /dev/null; then
    echo "Installing wasm-pack..."
    cargo install wasm-pack
fi

# Build for WASM
echo "Compiling to WASM..."
wasm-pack build --target web --out-dir pkg

echo "WASM build complete! Files are in the 'pkg' directory."
echo ""
echo "To use in a web project:"
echo "1. Copy the 'pkg' directory to your web project"
echo "2. Include the generated JavaScript file in your HTML"
echo "3. Use the exported functions in your JavaScript code" 