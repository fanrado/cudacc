#!/bin/bash
################################################################################
# CUDACC Test Suite Runner
# Runs comprehensive tests with detailed logging
################################################################################

set -e  # Exit on error (comment this out if you want to continue on failures)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OUTPUT_DIR="output_test"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo -e "${BLUE}================================================================================"
echo "                        CUDACC TEST SUITE"
echo "                        Started: $(date)"
echo -e "================================================================================${NC}\n"

# Clean and create output directory
echo -e "${YELLOW}[1/5] Setting up test environment...${NC}"
rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}

# Check Python environment
echo -e "${YELLOW}[2/5] Checking Python environment...${NC}"
python --version > ${OUTPUT_DIR}/environment_info.txt 2>&1
echo "Python Executable: $(which python)" >> ${OUTPUT_DIR}/environment_info.txt
echo -e "\n=== Installed Packages ===" >> ${OUTPUT_DIR}/environment_info.txt
pip list >> ${OUTPUT_DIR}/environment_info.txt 2>&1

# Check CUDA availability
echo -e "${YELLOW}[3/5] Checking CUDA availability...${NC}"
echo -e "\n=== NVIDIA GPU Info ===" >> ${OUTPUT_DIR}/environment_info.txt
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi >> ${OUTPUT_DIR}/environment_info.txt 2>&1
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
else
    echo "nvidia-smi not available" >> ${OUTPUT_DIR}/environment_info.txt
    echo -e "${RED}✗ No NVIDIA GPU detected${NC}"
fi

# Check if CuPy is installed
echo -e "\n=== CuPy Status ===" >> ${OUTPUT_DIR}/environment_info.txt
if python -c "import cupy" 2>/dev/null; then
    python -c "import cupy as cp; print(f'CuPy Version: {cp.__version__}'); print(f'CUDA Available: {cp.cuda.is_available()}'); print(f'Device Count: {cp.cuda.runtime.getDeviceCount()}')" >> ${OUTPUT_DIR}/environment_info.txt 2>&1
    echo -e "${GREEN}✓ CuPy is installed${NC}"
else
    echo "CuPy is NOT installed" >> ${OUTPUT_DIR}/environment_info.txt
    echo -e "${RED}✗ CuPy is NOT installed${NC}"
    echo -e "${YELLOW}  To install CuPy for CUDA 12.x, run:${NC}"
    echo -e "${YELLOW}  pip install cupy-cuda12x${NC}"
fi

echo -e "\n${YELLOW}[4/5] Running detailed tests...${NC}\n"

# Array of test files
declare -a TEST_FILES=(
    "tests/test_accelerator.py"
    "tests/test_dispatcher.py"
    "tests/test_memory.py"
    "tests/bridges/test_numpy_bridge.py"
    "tests/bridges/test_scipy_bridge.py"
    "tests/bridges/test_uproot_bridge.py"
    "tests/kernels/test_physics.py"
    "tests/kernels/test_reductions.py"
    "tests/kernels/test_transforms.py"
)

# Run each test with detailed output
for test_file in "${TEST_FILES[@]}"; do
    test_name=$(basename ${test_file} .py)
    echo -e "${BLUE}Running: ${test_file}${NC}"
    
    # Run with maximum verbosity
    set +e  # Don't exit on test failure
    python -m pytest ${test_file} \
        -vvv \
        --tb=long \
        --showlocals \
        --capture=no \
        --color=yes \
        > ${OUTPUT_DIR}/detailed_${test_name}.txt 2>&1
    
    exit_code=$?
    set -e
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}  ✓ PASSED${NC}\n"
    elif [ $exit_code -eq 5 ]; then
        echo -e "${YELLOW}  ⊘ NO TESTS COLLECTED${NC}\n"
    else
        echo -e "${RED}  ✗ FAILED (see detailed_${test_name}.txt)${NC}\n"
    fi
done

echo -e "${YELLOW}[5/5] Generating summary reports...${NC}"

# Run all tests together for overall summary
set +e
python -m pytest tests/ \
    -v \
    --tb=short \
    --color=yes \
    > ${OUTPUT_DIR}/overall_summary.txt 2>&1
set -e

# Create comprehensive test report
cat > ${OUTPUT_DIR}/test_report_${TIMESTAMP}.txt << 'REPORT_EOF'
================================================================================
                           CUDACC TEST REPORT
================================================================================

REPORT_EOF

echo "Test Run Date: $(date)" >> ${OUTPUT_DIR}/test_report_${TIMESTAMP}.txt
echo "" >> ${OUTPUT_DIR}/test_report_${TIMESTAMP}.txt

# Add environment info
cat ${OUTPUT_DIR}/environment_info.txt >> ${OUTPUT_DIR}/test_report_${TIMESTAMP}.txt

echo -e "\n\n" >> ${OUTPUT_DIR}/test_report_${TIMESTAMP}.txt
echo "================================================================================" >> ${OUTPUT_DIR}/test_report_${TIMESTAMP}.txt
echo "                           COMPONENT TEST RESULTS" >> ${OUTPUT_DIR}/test_report_${TIMESTAMP}.txt
echo "================================================================================" >> ${OUTPUT_DIR}/test_report_${TIMESTAMP}.txt

# Extract summary from each detailed test
for test_file in "${TEST_FILES[@]}"; do
    test_name=$(basename ${test_file} .py)
    echo -e "\n${test_name}:" >> ${OUTPUT_DIR}/test_report_${TIMESTAMP}.txt
    echo "-------------------" >> ${OUTPUT_DIR}/test_report_${TIMESTAMP}.txt
    
    # Get the summary line
    if [ -f "${OUTPUT_DIR}/detailed_${test_name}.txt" ]; then
        tail -1 "${OUTPUT_DIR}/detailed_${test_name}.txt" >> ${OUTPUT_DIR}/test_report_${TIMESTAMP}.txt
        
        # Count skipped tests and reasons
        grep -A 1 "SKIPPED" "${OUTPUT_DIR}/detailed_${test_name}.txt" | grep -E "reason|Skipped" >> ${OUTPUT_DIR}/test_report_${TIMESTAMP}.txt 2>/dev/null || true
        
        # Show failed test lines
        grep -E "FAILED|ERROR" "${OUTPUT_DIR}/detailed_${test_name}.txt" >> ${OUTPUT_DIR}/test_report_${TIMESTAMP}.txt 2>/dev/null || true
    fi
    echo "" >> ${OUTPUT_DIR}/test_report_${TIMESTAMP}.txt
done

# Add overall summary
echo -e "\n\n" >> ${OUTPUT_DIR}/test_report_${TIMESTAMP}.txt
echo "================================================================================" >> ${OUTPUT_DIR}/test_report_${TIMESTAMP}.txt
echo "                           OVERALL TEST SUMMARY" >> ${OUTPUT_DIR}/test_report_${TIMESTAMP}.txt
echo "================================================================================" >> ${OUTPUT_DIR}/test_report_${TIMESTAMP}.txt
tail -20 ${OUTPUT_DIR}/overall_summary.txt >> ${OUTPUT_DIR}/test_report_${TIMESTAMP}.txt

# Create a symlink to the latest report
ln -sf test_report_${TIMESTAMP}.txt ${OUTPUT_DIR}/test_report_latest.txt

echo -e "\n${GREEN}================================================================================"
echo "                        TEST SUITE COMPLETED"
echo "                        Finished: $(date)"
echo -e "================================================================================${NC}\n"

echo -e "${BLUE}Test outputs saved to: ${OUTPUT_DIR}/${NC}"
echo -e "${BLUE}Main report: ${OUTPUT_DIR}/test_report_${TIMESTAMP}.txt${NC}"
echo -e "${BLUE}Latest report link: ${OUTPUT_DIR}/test_report_latest.txt${NC}\n"

# Show quick summary
echo -e "${YELLOW}Quick Summary:${NC}"
if [ -f "${OUTPUT_DIR}/overall_summary.txt" ]; then
    tail -1 ${OUTPUT_DIR}/overall_summary.txt
fi

echo -e "\n${YELLOW}Detailed test logs:${NC}"
ls -lh ${OUTPUT_DIR}/detailed_*.txt

echo -e "\n${GREEN}Done!${NC}"
