#!/bin/bash

# Set the test folder path
test_folder="test/"
source env/bin/activate
# Run pytest with pytest-pythonpath
python -m pytest $test_folder

