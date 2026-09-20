"""
Debug script for ray-tracing module.
Checks imports, file paths, and basic setup.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("Checking ray-tracing dependencies...")

try:
    import bifacial_radiance
    print(f"  ✅ bifacial_radiance: {bifacial_radiance.__version__}")
except ImportError:
    print("  ⚠️  bifacial_radiance not installed")

try:
    import pyradiance
    print(f"  ✅ pyradiance: available")
except ImportError:
    print("  ⚠️  pyradiance not installed")

# Check for Radiance binaries
import shutil
for cmd in ["radiance", "rfluxmtx", "gendaylit", "rtrace"]:
    path = shutil.which(cmd)
    if path:
        print(f"  ✅ {cmd}: {path}")
    else:
        print(f"  ❌ {cmd}: not found on PATH")

print("\nTo install Radiance:")
print("  Ubuntu:   sudo apt install radiance")
print("  macOS:    brew install radiance")
print("  Windows:  Download from https://www.radiance-online.org")
