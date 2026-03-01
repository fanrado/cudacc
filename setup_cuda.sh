#!/bin/bash
################################################################################
# CUDACC CUDA Setup Script
# Installs CuPy and verifies CUDA environment
################################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================================"
echo "                        CUDACC CUDA Setup"
echo "================================================================================${NC}\n"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${GREEN}✓ Virtual environment detected: $VIRTUAL_ENV${NC}"
else
    echo -e "${YELLOW}⚠ No virtual environment detected${NC}"
    echo -e "${YELLOW}  To activate your environment, run:${NC}"
    echo -e "${YELLOW}  source /media/rado/RADO/WORKSPACE/cudacc/env_cudacc/bin/activate${NC}\n"
    echo -e "${YELLOW}  Do you want to continue anyway? (yes/no)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy][Ee][Ss]$|^[Yy]$ ]]; then
        echo -e "${RED}Exiting...${NC}"
        exit 1
    fi
fi

# Check for NVIDIA GPU
echo -e "\n${YELLOW}[1/4] Checking for NVIDIA GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo -e "${GREEN}✓ NVIDIA GPU found${NC}\n"
    
    # Get CUDA version from driver
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    echo -e "${GREEN}CUDA Version from driver: ${CUDA_VERSION}${NC}"
else
    echo -e "${RED}✗ nvidia-smi not found. NVIDIA drivers may not be installed.${NC}"
    exit 1
fi

# Determine which CuPy package to install based on CUDA version
echo -e "\n${YELLOW}[2/4] Determining CuPy package...${NC}"
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)

if [ "$CUDA_MAJOR" -ge 12 ]; then
    CUPY_PACKAGE="cupy-cuda12x"
    echo -e "${GREEN}Will install: ${CUPY_PACKAGE} (for CUDA 12.x)${NC}"
elif [ "$CUDA_MAJOR" -eq 11 ]; then
    CUPY_PACKAGE="cupy-cuda11x"
    echo -e "${GREEN}Will install: ${CUPY_PACKAGE} (for CUDA 11.x)${NC}"
else
    echo -e "${RED}Unsupported CUDA version: ${CUDA_VERSION}${NC}"
    echo -e "${YELLOW}Please install CuPy manually for your CUDA version.${NC}"
    exit 1
fi

# Check if CuPy is already installed
echo -e "\n${YELLOW}[3/4] Checking for existing CuPy installation...${NC}"
if python -c "import cupy" 2>/dev/null; then
    INSTALLED_VERSION=$(python -c "import cupy; print(cupy.__version__)")
    echo -e "${GREEN}✓ CuPy ${INSTALLED_VERSION} is already installed${NC}"
    
    echo -e "${YELLOW}Do you want to reinstall/upgrade? (yes/no)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy][Ee][Ss]$|^[Yy]$ ]]; then
        echo -e "${BLUE}Skipping CuPy installation${NC}"
        SKIP_INSTALL=true
    fi
else
    echo -e "${YELLOW}CuPy is not installed${NC}"
fi

# Install CuPy
if [ "$SKIP_INSTALL" != "true" ]; then
    echo -e "\n${YELLOW}[4/4] Installing ${CUPY_PACKAGE}...${NC}"
    echo -e "${BLUE}This may take several minutes...${NC}\n"
    
    pip install --upgrade pip
    pip install ${CUPY_PACKAGE}
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}✓ CuPy installed successfully${NC}"
    else
        echo -e "\n${RED}✗ CuPy installation failed${NC}"
        echo -e "${YELLOW}You may need to install CUDA toolkit manually.${NC}"
        exit 1
    fi
fi

# Verify installation
echo -e "\n${YELLOW}Verifying CuPy installation...${NC}"
python << 'VERIFY_EOF'
import sys
try:
    import cupy as cp
    print(f"✓ CuPy version: {cp.__version__}")
    print(f"✓ CUDA available: {cp.cuda.is_available()}")
    
    if cp.cuda.is_available():
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"✓ Number of CUDA devices: {device_count}")
        
        for i in range(device_count):
            device = cp.cuda.Device(i)
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props['name'].decode('utf-8')
            mem = props['totalGlobalMem'] / (1024**3)  # Convert to GB
            print(f"  Device {i}: {name} ({mem:.2f} GB)")
        
        # Test a simple operation
        x = cp.array([1, 2, 3])
        y = cp.array([4, 5, 6])
        z = x + y
        print(f"✓ Simple GPU operation test: {z.get()}")
        print("\n✓✓✓ CUDA is fully functional! ✓✓✓")
    else:
        print("✗ CUDA is not available through CuPy")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
VERIFY_EOF

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}================================================================================"
    echo "                        CUDA Setup Complete!"
    echo "================================================================================${NC}"
    echo -e "${GREEN}You can now run the test suite with: ./run_tests.sh${NC}\n"
else
    echo -e "\n${RED}================================================================================"
    echo "                        CUDA Setup Failed"
    echo "================================================================================${NC}"
    echo -e "${RED}Please check the errors above and try manual installation.${NC}\n"
    exit 1
fi
