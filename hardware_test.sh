#!/bin/bash
ENV_DIR="env"  
if [ ! -d "$ENV_DIR" ]; then
    echo "Error: Virtual environment '$ENV_DIR' does not exist ❌"
    exit 1
fi
echo "Virtual environment '$ENV_DIR' : ✅"
echo "Info: Activating virtual environment..."
source "$ENV_DIR/bin/activate"
echo "...✅"
PULSE_SERVER="/run/pulse/native" python ./hardware_test/test_hardware.py "$@"
deactivate

