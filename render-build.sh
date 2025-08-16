#!/usr/bin/env bash

set -e

# Ensure pip, setuptools, and wheel are up-to-date BEFORE installing other requirements
python -m pip install --upgrade pip setuptools wheel

# Optionally, add extra build tools if you know you need them
# python -m pip install build

# Install all project dependencies
pip install -r requirements.txt
