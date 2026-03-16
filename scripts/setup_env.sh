#!/bin/bash

# SG-TERRA (SG_proj_007) Environment Setup Script
# This script creates a python virtual environment and installs dependencies.

echo "Setting up Python virtual environment for SG-TERRA..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Created virtual environment 'venv'."
else
    echo "Virtual environment 'venv' already exists."
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install requirements
echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup Complete. To activate the environment, run: source venv/bin/activate"
