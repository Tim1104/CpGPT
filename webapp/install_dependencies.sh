#!/bin/bash

# CpGPT Web Application - Dependency Installation Script
# For macOS (Apple Silicon) with Python 3.13

set -e  # Exit on error

echo "🚀 CpGPT Web Application - Dependency Installation"
echo "=================================================="
echo ""

# Check Python version
echo "📌 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $PYTHON_VERSION"
echo ""

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  Warning: This script is optimized for macOS"
    echo "   It may work on other platforms but is not tested"
    echo ""
fi

# Check if we're on Apple Silicon
ARCH=$(uname -m)
echo "📌 System architecture: $ARCH"
if [[ "$ARCH" == "arm64" ]]; then
    echo "   ✅ Apple Silicon detected"
else
    echo "   ℹ️  Not Apple Silicon (may use different optimizations)"
fi
echo ""

# Install core web dependencies
echo "📦 Installing core web dependencies..."
echo "   - FastAPI (Web framework)"
echo "   - Uvicorn (ASGI server)"
echo "   - python-multipart (File upload support)"
echo ""

pip3 install "fastapi>=0.112.0,<0.115" uvicorn python-multipart

echo ""
echo "✅ Core web dependencies installed"
echo ""

# Install visualization libraries
echo "📊 Installing visualization libraries..."
echo "   - Matplotlib (Plotting)"
echo "   - Seaborn (Statistical visualization)"
echo ""

pip3 install matplotlib seaborn

echo ""
echo "✅ Visualization libraries installed"
echo ""

# Install PDF generation (optional)
echo "📄 Installing PDF generation library..."
echo "   - WeasyPrint (HTML to PDF)"
echo ""

if pip3 install weasyprint; then
    echo ""
    echo "✅ PDF generation library installed"
else
    echo ""
    echo "⚠️  WeasyPrint installation failed (PDF export will not work)"
    echo "   You can continue without PDF export functionality"
fi
echo ""

# Check if pyarrow is needed
echo "📦 Checking Arrow format support..."
if python3 -c "import pyarrow" 2>/dev/null; then
    echo "   ✅ PyArrow already installed"
else
    echo "   ⚠️  PyArrow not installed"
    echo "   Arrow format (.arrow, .feather) will not be supported"
    echo "   Only CSV format will work"
    echo ""
    read -p "   Install PyArrow? (may take several minutes) [y/N]: " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Installing PyArrow..."
        if pip3 install pyarrow; then
            echo "   ✅ PyArrow installed"
        else
            echo "   ⚠️  PyArrow installation failed"
            echo "   You can still use CSV format"
        fi
    else
        echo "   Skipping PyArrow installation"
        echo "   Only CSV format will be supported"
    fi
fi
echo ""

# Verify installations
echo "🔍 Verifying installations..."
echo ""

python3 -c "
import sys
import importlib

packages = {
    'fastapi': 'FastAPI',
    'uvicorn': 'Uvicorn',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
}

optional_packages = {
    'weasyprint': 'WeasyPrint (PDF export)',
    'pyarrow': 'PyArrow (Arrow format)',
}

print('Core Dependencies:')
all_ok = True
for module, name in packages.items():
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f'  ✅ {name}: {version}')
    except ImportError:
        print(f'  ❌ {name}: NOT INSTALLED')
        all_ok = False

print()
print('Optional Dependencies:')
for module, name in optional_packages.items():
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f'  ✅ {name}: {version}')
    except ImportError:
        print(f'  ⚠️  {name}: NOT INSTALLED (optional)')

if not all_ok:
    print()
    print('❌ Some core dependencies are missing!')
    sys.exit(1)
"

echo ""
echo "🖥️  Testing GPU detection..."
echo ""

python3 webapp/test_gpu_detection.py

echo ""
echo "=================================================="
echo "✅ Installation Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Start the server:"
echo "     bash webapp/start_server.sh"
echo ""
echo "  2. Open your browser:"
echo "     http://localhost:8000"
echo ""
echo "  3. Upload a 935k methylation data file (CSV or Arrow)"
echo ""
echo "For more information, see:"
echo "  - webapp/README.md"
echo "  - webapp/QUICKSTART.md"
echo "  - webapp/INSTALL_MACOS.md"
echo ""

