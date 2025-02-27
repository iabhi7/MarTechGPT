# Installation Guide

## Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

## Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-marketing-suite.git
cd ai-marketing-suite
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API credentials
```

5. **Test the installation**

```bash
# Run a simple example
python examples/subject_line_optimizer_demo.py
```

## Development Installation

If you're planning to contribute to the project or modify the code:

```bash
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

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

4. Set `USE_GPU=true` in your `.env` file

## Troubleshooting

### Common Issues

- **ImportError**: Make sure all dependencies are correctly installed
- **CUDA errors**: Check your CUDA and PyTorch versions are compatible
- **API authentication errors**: Verify your API key is correct

### Getting Help

If you encounter any issues that aren't covered here, please:

1. Check the [GitHub Issues](https://github.com/yourusername/ai-marketing-suite/issues) to see if it's a known problem
2. Submit a new issue with details about your problem 