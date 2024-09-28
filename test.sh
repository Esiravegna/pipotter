#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
# Set the test folder path
test_folder="test/"
source env/bin/activate
# Run pytest with pytest-pythonpath
python -m pytest $test_folder

