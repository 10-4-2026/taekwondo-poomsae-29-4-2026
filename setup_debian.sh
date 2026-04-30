#!/bin/bash

# Script to install system dependencies for Debian 11 (Bullseye)
# Required for OpenCV and MediaPipe to work correctly.

echo "Updating package list..."
sudo apt-get update

echo "Installing system dependencies..."
sudo apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

echo "Dependencies installed successfully!"
echo "You can now run your streamlit app using: streamlit run app.py"
