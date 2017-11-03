#!/usr/bin/env bash
# check if conda installed
conda -v foo >/dev/null 2>&1 || { echo "I require conda but it's not installed.  Download from  https://conda.io ." >&2; exit 1; }
if [ -f settings.sh ]; then
    source settings.sh
fi
source activate pipotter
python run.py "$@"
