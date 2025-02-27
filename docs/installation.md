# Installation Guide

This guide provides detailed instructions for setting up the Netcore AI Marketing Suite on your system.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment tool (optional but recommended)
- Git

## Basic Installation

1. **Clone the repository**

```
git clone https://github.com/yourusername/netcore-ai-marketing-suite.git
cd netcore-ai-marketing-suite
```

2. **Set up a virtual environment (recommended)**

```
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**

```
pip install -r requirements.txt
```

4. **Set up environment variables**

```
# Copy the example .env file
cp .env.example .env

# Edit the .env file with your preferred text editor
# Replace the placeholders with your actual credentials
```

5. **Test the installation**

```
# Run a simple example
python examples/subject_line_optimizer_demo.py
```

## Development Installation

If you're planning to contribute to the project or modify the code:

```
# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

## GPU Acceleration (Optional)

For faster LLM inference, GPU acceleration is recommended:

1. Ensure you have a CUDA-compatible GPU
2. Install the CUDA toolkit and cuDNN appropriate for your system
3. Install PyTorch with CUDA support:

```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

4. Set `USE_GPU=true` in your `.env` file

## Troubleshooting

### Common Issues

- **ImportError**: Make sure all dependencies are correctly installed
- **CUDA errors**: Check your CUDA and PyTorch versions are compatible
- **API authentication errors**: Verify your Netcore API key is correct

### Getting Help

If you encounter any issues that aren't covered here, please:

1. Check the [GitHub Issues](https://github.com/yourusername/netcore-ai-marketing-suite/issues) to see if it's a known problem
2. Submit a new issue with details about your problem 