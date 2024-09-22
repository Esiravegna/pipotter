#!/bin/bash
ENV_DIR="env"  
if [ ! -d "$ENV_DIR" ]; then
    echo "Error: Virtual environment '$ENV_DIR' does not exist ❌"
    exit 1
fi
echo "Virtual environment '$ENV_DIR' : ✅"
echo "Info: Activating virtual environment..."
source "$ENV_DIR/bin/activate"
echo "✅"
REQUIREMENTS_FILE="requirements.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "'$REQUIREMENTS_FILE' not found ❌"
    exit 1
fi
echo "Info: Found requirements file '$REQUIREMENTS_FILE' ✅"
python run.py "$@"
deactivate

