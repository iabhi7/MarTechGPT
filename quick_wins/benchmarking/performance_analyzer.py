import time
import psutil
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import os
from quick_wins.marketing_chatbot.chatbot import MarketingChatbot

class PerformanceAnalyzer:
    """Benchmark and analyze performance of AI models"""
    
    def __init__(self):
        self.results = {}
        self.metrics = []
    
    def benchmark_model(self, model_name, quantized=True, iterations=10, warmup=2):
        """Benchmark a model's performance"""
        print(f"Benchmarking {model_name} (quantized={quantized})...")
        
        # Initialize model
        start_time = time.time()
        chatbot = MarketingChatbot(model_name=model_name, quantize=quantized)
        load_time = time.time() - start_time
        
        # Get model size
        model_size = chatbot.get_performance_metrics()["model_size_mb"]
        
        # Warmup
        for _ in range(warmup):
            chatbot.get_response("Warmup query to prime the model")
        
        # Test prompts
        test_prompts = [
            "How can I improve email deliverability?",
            "Generate an ad for a fitness app targeting young professionals",
            "What are the best practices for customer segmentation?",
            "How can I measure the ROI of my marketing campaigns?",
            "Generate subject lines for a product launch email"
        ]
        
        # Benchmark inference time
        inference_times = []
        memory_usages = []
        
        for prompt in tqdm(test_prompts * (iterations // len(test_prompts) + 1)):
            # Measure memory before
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
            
            # Run inference
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            _ = chatbot.get_response(prompt)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.time()
            
            # Measure memory after
            mem_after = process.memory_info().rss / 1024 / 1024
            memory_usage = mem_after - mem_before
            
            inference_times.append(end - start)
            memory_usages.append(memory_usage)
        
        # Calculate statistics
        avg_inference = np.mean(inference_times)
        p95_inference = np.percentile(inference_times, 95)
        max_inference = np.max(inference_times)
        avg_memory = np.mean(memory_usages)
        
        result = {
            "model_name": model_name,
            "quantized": quantized,
            "model_size_mb": model_size,
            "load_time_sec": load_time,
            "avg_inference_sec": avg_inference,
            "p95_inference_sec": p95_inference,
            "max_inference_sec": max_inference,
            "avg_memory_usage_mb": avg_memory
        }
        
        # Store results
        key = f"{model_name}_{'quantized' if quantized else 'unquantized'}"
        self.results[key] = result
        self.metrics.append(result)
        
        return result
    
    def compare_models(self, models, quantize_options=[True, False]):
        """Compare performance across multiple models with and without quantization"""
        for model in models:
            for quantize in quantize_options:
                self.benchmark_model(model, quantized=quantize)
        
        return self.get_comparison_table()
    
    def get_comparison_table(self):
        """Generate a comparison table of benchmark results"""
        if not self.metrics:
            return "No benchmark data available"
        
        df = pd.DataFrame(self.metrics)
        return df
    
    def generate_charts(self, output_dir="benchmark_results"):
        """Generate visualization charts of benchmark results"""
        if not self.metrics:
            return "No benchmark data available"
        
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(self.metrics)
        
        # Size comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(x="model_name", y="model_size_mb", hue="quantized", data=df)
        plt.title("Model Size Comparison")
        plt.ylabel("Size (MB)")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_size_comparison.png")
        
        # Inference time comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(x="model_name", y="avg_inference_sec", hue="quantized", data=df)
        plt.title("Average Inference Time Comparison")
        plt.ylabel("Time (seconds)")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/inference_time_comparison.png")
        
        # Memory usage comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(x="model_name", y="avg_memory_usage_mb", hue="quantized", data=df)
        plt.title("Average Memory Usage Comparison")
        plt.ylabel("Memory (MB)")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/memory_usage_comparison.png")
        
        print(f"Charts saved to {output_dir}/")
        return f"Charts saved to {output_dir}/"

    def visualize_comparison(self, output_dir="benchmark_results"):
        """Generate visualization charts of benchmark results"""
        plt.figure(figsize=(12, 6))
        plt.title("Platform Integration Performance")
        plt.xlabel("Model Configuration")

# Example usage
if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    
    # Compare different models
    analyzer.compare_models(
        models=["distilgpt2", "facebook/opt-125m", "mistralai/Mistral-7B-Instruct-v0.1"], 
        quantize_options=[True, False]
    )
    
    # Generate comparison table
    comparison = analyzer.get_comparison_table()
    print(comparison)
    
    # Generate charts
    analyzer.generate_charts() 