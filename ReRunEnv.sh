#!/bin/bash
# This shell script allows for quick restarting of a virtual environment

# Remove the existing virtual environment directory
rm -rf env

# Install virtualenv (if needed)
apt install python3-virtualenv

# Create a new virtual environment
python3 -m venv env

# Activate the virtual environment
source env/bin/activate

# Install the required packages
pip install -r requirements.txt
