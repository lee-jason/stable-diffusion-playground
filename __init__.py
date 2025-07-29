import os
import sys

# Get the directory containing this __init__.py file (project root)
project_root = os.path.dirname(os.path.abspath(__file__))

# Add project root to Python path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
