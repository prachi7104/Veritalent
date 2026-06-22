import sys
import os

# Ensure the repo root (c:\projects\Veritalent) is on the path
# so `import feature_lab` works from any test file.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
