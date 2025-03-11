import os
import sys
import time
import random

def main():
    """
    Simple test script to verify Python is working without any dependencies.
    """
    print("\n===== PYTHON TEST SCRIPT =====")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Current timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create a simple output directory
    os.makedirs("test_output", exist_ok=True)
    
    # Generate a simple text file
    with open("test_output/test_file.txt", "w") as f:
        f.write(f"Test file created at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Random number: {random.randint(1, 1000)}\n")
    
    print("\n✓ Test file created successfully at test_output/test_file.txt")
    print("✓ Python is working correctly!")
    print("===== TEST COMPLETE =====\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 