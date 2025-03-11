import sys
import torch
import matplotlib.pyplot as plt

print("=== Environment Validation ===")
print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}")
print(f"Matplotlib: {plt.__version__}")

# Test basic plotting
plt.plot([1, 2, 3], [4, 1, 2])
plt.savefig('env_test.png')
print("Validation plot saved to env_test.png") 