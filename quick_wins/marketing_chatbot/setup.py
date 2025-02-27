from setuptools import setup, find_packages

setup(
    name="ai-marketing-chatbot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "flask>=2.0.0",
        "numpy>=1.20.0",
        "tqdm>=4.65.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.40.0",  # For 8-bit quantization
        "pytest>=7.0.0",         # For testing
        "gunicorn>=20.1.0"       # For production deployment
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered marketing chatbot",
    keywords="marketing, chatbot, llm, ai",
    url="https://github.com/yourusername/ai-marketing-chatbot",
) 