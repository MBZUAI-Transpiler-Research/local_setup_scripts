#!/bin/bash

# Check if project name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <project_name>"
    exit 1
fi

PROJECT_NAME=$1
DOCKER_IMAGE="${PROJECT_NAME}_riscv"
DOCKER_CONTAINER="${PROJECT_NAME}_riscv_container"

# Create and move into the project directory
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME" || { echo "Error: Failed to enter project directory"; exit 1; }

echo "Setting up project: $PROJECT_NAME"

# 1. Clone fresh repository
echo "Cloning repository..."
git clone https://github.com/celine-lee/transpile.git . 
if [ $? -ne 0 ]; then
    echo "Error: Failed to clone repository"
    exit 1
fi
echo "Repository cloned successfully."

# 2. Install pip
echo "Installing pip..."

# Download get-pip.py and check if successful
curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to download get-pip.py"
    exit 1
fi

# Install pip and check if successful
python3 get-pip.py --user
if [ $? -ne 0 ]; then
    echo "Error: Failed to install pip"
    exit 1
fi

# Add pip to PATH
export PATH=$HOME/.local/bin:$PATH
if [ $? -ne 0 ]; then
    echo "Error: Failed to update PATH"
    exit 1
fi

# Upgrade pip
pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "Error: Failed to upgrade pip"
    exit 1
fi

# Verify installation
pip --version &>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Pip installation verification failed"
    exit 1
fi

python3 --version &>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Python installation verification failed"
    exit 1
fi

echo "Pip installed and upgraded successfully."

# 3. Install virtualenv
echo "Installing virtualenv..."
pip install --user virtualenv
if [ $? -ne 0 ]; then
    echo "Error: Failed to install virtualenv"
    exit 1
fi

# Create virtual environment
~/.local/bin/virtualenv venv
if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

echo "Virtual environment created and activated."

# 4. Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install required packages"
    exit 1
fi

pip install -Ue .
if [ $? -ne 0 ]; then
    echo "Error: Failed to install project in editable mode"
    exit 1
fi

python setup.py install
if [ $? -ne 0 ]; then
    echo "Error: Failed to run setup.py install"
    exit 1
fi

echo "Dependencies installed successfully."

# 5. Unzip RISC-V Verification Image
echo "Unzipping verification image..."
unzip -q verification_image.zip
if [ $? -ne 0 ]; then
    echo "Error: Failed to unzip verification image"
    exit 1
fi

# Move into verification_image directory
cd verification_image || { echo "Error: Failed to enter verification_image directory"; exit 1; }

# 6. Build the verification image
echo "Building verification image..."
bash build.sh
if [ $? -ne 0 ]; then
    echo "Error: Failed to build verification image"
    exit 1
fi

echo "Verification image built successfully."

# 7. Build the RISC-V Docker Image
echo "Building Docker image..."
docker rmi -f "$DOCKER_IMAGE" &>/dev/null
docker build -t "$DOCKER_IMAGE" .
if [ $? -ne 0 ]; then
    echo "Error: Failed to build Docker image $DOCKER_IMAGE"
    exit 1
fi

echo "Docker image $DOCKER_IMAGE built successfully."

# 8. Run the Container and Compile the C program
docker rm -f "$DOCKER_CONTAINER" &>/dev/null
docker run --name "$DOCKER_CONTAINER" -it "$DOCKER_IMAGE" bash -c "
    echo 'Downloading C program...';
    wget -q https://raw.githubusercontent.com/celine-lee/transpile/refs/heads/main/data/project-euler-c/original_c/problem12.c;
    if [ \$? -ne 0 ]; then
        echo 'Error: Failed to download C program inside container.';
        exit 1;
    fi;
    echo 'C program downloaded.';

    echo 'Compiling C program...';
    riscv64-unknown-linux-gnu-gcc -S problem12.c -o problem12.s;
    if [ \$? -ne 0 ]; then
        echo 'Error: Failed to compile assembly (.s) file.';
        exit 1;
    fi;
    
    riscv64-unknown-linux-gnu-gcc problem12.c -o problem12;
    if [ \$? -ne 0 ]; then
        echo 'Error: Failed to compile executable.';
        exit 1;
    fi;

    ls;
    echo 'C program compiled successfully.';

    echo 'Running compiled program using QEMU...';
    qemu-riscv64 ./problem12;
    if [ \$? -ne 0 ]; then
        echo 'Error: Execution failed.';
        exit 1;
    fi;

    echo 'Execution complete.';
"
if [ $? -ne 0 ]; then
    echo "Error: Failed to start container $DOCKER_CONTAINER."
    exit 1
fi

echo "Container $DOCKER_CONTAINER running, program compiled and executed."
