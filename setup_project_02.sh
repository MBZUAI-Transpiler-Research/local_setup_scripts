#!/bin/bash

# Check if project name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <project_name>"
    exit 1
fi

PROJECT_NAME=$1

# Define RISC-V Docker image/container names
DOCKER_IMAGE_RISCV="${PROJECT_NAME}_riscv"
DOCKER_CONTAINER_RISCV="${PROJECT_NAME}_riscv_container"

# Define ARM Docker image/container names
DOCKER_IMAGE_ARM="${PROJECT_NAME}_arm"
DOCKER_CONTAINER_ARM="${PROJECT_NAME}_arm_container"

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
echo "Building RISC-V Docker image..."
docker rmi -f "$DOCKER_IMAGE_RISCV" &>/dev/null
docker build -t "$DOCKER_IMAGE_RISCV" .
if [ $? -ne 0 ]; then
    echo "Error: Failed to build Docker image $DOCKER_IMAGE_RISCV"
    exit 1
fi
echo "RISC-V Docker image $DOCKER_IMAGE_RISCV built successfully."

# 8. Run the RISC-V Container and Compile the C program
docker rm -f "$DOCKER_CONTAINER_RISCV" &>/dev/null
docker run --name "$DOCKER_CONTAINER_RISCV" -it "$DOCKER_IMAGE_RISCV" bash -c "
    echo 'Downloading C program...';
    wget -q https://raw.githubusercontent.com/celine-lee/transpile/refs/heads/main/data/project-euler-c/original_c/problem12.c;
    if [ \$? -ne 0 ]; then
        echo 'Error: Failed to download C program inside container.';
        exit 1;
    fi;
    echo 'C program downloaded.';

    echo 'Compiling C program...';
    riscv64-unknown-linux-gnu-gcc -S problem12.c -o problem12.s;
    riscv64-unknown-linux-gnu-gcc problem12.c -o problem12;
    if [ \$? -ne 0 ]; then
        echo 'Error: Compilation failed.';
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
    echo "Error: Failed to start container $DOCKER_CONTAINER_RISCV."
    exit 1
fi
echo "RISC-V program executed successfully."

# Stop and save RISC-V container
echo "Stopping and committing the RISC-V container..."
docker stop "$DOCKER_CONTAINER_RISCV" &>/dev/null
docker commit "$DOCKER_CONTAINER_RISCV" "$DOCKER_IMAGE_RISCV":latest &>/dev/null
docker rm "$DOCKER_CONTAINER_RISCV" &>/dev/null
echo "RISC-V container stopped and removed."

# 9. Build the ARM Docker Image
echo "Building ARM Docker image..."
docker rmi -f "$DOCKER_IMAGE_ARM" &>/dev/null
docker build -t "$DOCKER_IMAGE_ARM" -<<EOF
FROM ubuntu:22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Update and install required dependencies
RUN apt-get update && apt-get install -y \
    gcc-aarch64-linux-gnu \
    wget \
    vim \
    build-essential \
    ninja-build \
    python3-pip \
    libglib2.0-dev \
    libpixman-1-dev \
    flex \
    bison \
    xz-utils

# Download and extract QEMU
RUN wget -q https://download.qemu.org/qemu-8.1.2.tar.xz && \
    tar xvJf qemu-8.1.2.tar.xz && \
    cd qemu-8.1.2 && \
    ./configure --enable-plugins && \
    make -j\$(nproc)

# Set working directory
WORKDIR /workspace
EOF

if [ $? -ne 0 ]; then
    echo "Error: Failed to build Docker image $DOCKER_IMAGE_ARM"
    exit 1
fi
echo "ARM Docker image $DOCKER_IMAGE_ARM built successfully."


# 10. Run the ARM Container and Compile the C program
docker rm -f "$DOCKER_CONTAINER_ARM" &>/dev/null
docker run --name "$DOCKER_CONTAINER_ARM" -it "$DOCKER_IMAGE_ARM" bash -c "
    echo 'Downloading C program...';
    wget -q https://raw.githubusercontent.com/celine-lee/transpile/refs/heads/main/data/project-euler-c/original_c/problem12.c;
    if [ \$? -ne 0 ]; then
        echo 'Error: Failed to download C program inside container.';
        exit 1;
    fi;
    echo 'C program downloaded.';

    echo 'Compiling C program for ARM...';
    aarch64-linux-gnu-gcc -S problem12.c -o problem12.s;
    aarch64-linux-gnu-gcc problem12.c -o problem12;
    if [ \$? -ne 0 ]; then
        echo 'Error: Compilation failed.';
        exit 1;
    fi;

    ls;
    echo 'C program compiled successfully.';

    echo 'Checking dynamic linker path...';
    find / -name \"ld-linux-aarch64.so.1\" 2>/dev/null;

    echo 'Running compiled ARM program using QEMU...';
    ./qemu-8.1.2/build/qemu-aarch64 -L /usr/aarch64-linux-gnu problem12;
    if [ \$? -ne 0 ]; then
        echo 'Error: Execution failed.';
        exit 1;
    fi;

    echo 'Execution complete.';
"

if [ $? -ne 0 ]; then
    echo "Error: Failed to start ARM container $DOCKER_CONTAINER_ARM."
    exit 1
fi
echo "ARM program executed successfully."


# Stop and save ARM container
echo "Stopping and committing the ARM container..."
docker stop "$DOCKER_CONTAINER_ARM" &>/dev/null
docker commit "$DOCKER_CONTAINER_ARM" "$DOCKER_IMAGE_ARM":latest &>/dev/null
docker rm "$DOCKER_CONTAINER_ARM" &>/dev/null
echo "ARM container stopped and removed."

echo "Setup complete for both RISC-V and ARM."

