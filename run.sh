#!/usr/bin/env bash
# check if conda installed
# File containing the basic requirements
REQUIREMENTS_FILE="requirements.txt"

# Function to check if a Python package is installed
check_and_install() {
    PACKAGE_NAME=$1
    PACKAGE_VERSION=$2

    # Check if the package is installed
    INSTALLED_VERSION=$(pip3 show "$PACKAGE_NAME" | grep Version | awk '{print $2}')

    if [ "$INSTALLED_VERSION" == "$PACKAGE_VERSION" ]; then
        echo "$PACKAGE_NAME==$PACKAGE_VERSION is already installed."
    else
        echo "Installing $PACKAGE_NAME==$PACKAGE_VERSION..."
        pip3 install "$PACKAGE_NAME==$PACKAGE_VERSION"
    fi
}

# Loop through the requirements.txt and check each package
while IFS== read -r PACKAGE VERSION; do
    check_and_install "$PACKAGE" "$VERSION"
done < "$REQUIREMENTS_FILE"
python3 run.py "$@"
