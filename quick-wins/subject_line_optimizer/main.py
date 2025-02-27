import pandas as pd
import numpy as np
from transformers import pipeline
import json
import requests
from typing import List, Dict, Any, Optional

class SubjectLineOptimizer:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt2"):
        """
        Initialize the Subject Line Optimizer.
        
        Args:
            api_key: API key for OpenAI or Netcore API integration
            model_name: Model to use for text generation
        """
        self.api_key = api_key
        self.model_name = model_name
        self.generator = pipeline('text-generation', model=model_name)
        self.metrics_cache = {}
        
    def load_historical_data(self, filepath: str = None, netcore_api: bool = False):
        """
        Load historical email campaign data either from a file or Netcore API.
        
        Args:
            filepath: Path to CSV file with historical data
            netcore_api: Whether to fetch data from Netcore API
        """
        if netcore_api and self.api_key:
            # Example of how we would connect to Netcore's API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get(
                "https://api.netcore.co.in/campaigns/email/performance", 
                headers=headers
            )
            if response.status_code == 200:
                data = response.json()
                self.historical_data = pd.DataFrame(data['campaigns'])
                print(f"Loaded {len(self.historical_data)} campaigns from Netcore API")
            else:
                print(f"Failed to load data from Netcore API: {response.status_code}")
                self.historical_data = pd.DataFrame()
        elif filepath:
            self.historical_data = pd.DataFrame(pd.read_csv(filepath))
            print(f"Loaded {len(self.historical_data)} campaigns from {filepath}")
        else:
            # Create sample data if no source is provided
            self.historical_data = pd.DataFrame({
                'subject_line': [
                    "Last chance to save 20% on your order",
                    "Your exclusive preview is waiting",
                    "Don't miss out on these deals",
                    "Thank you for your loyalty"
                ],
                'open_rate': [0.23, 0.31, 0.18, 0.45],
                'click_rate': [0.05, 0.12, 0.04, 0.15],
                'conversion_rate': [0.02, 0.04, 0.01, 0.05]
            })
            print("Created sample historical data")
            
    def analyze_top_performers(self, metric: str = "open_rate", top_n: int = 5):
        """
        Analyze top-performing subject lines based on a given metric.
        
        Args:
            metric: Metric to use for ranking (open_rate, click_rate, etc.)
            top_n: Number of top performers to return
        
        Returns:
            Dictionary with analysis results
        """
        if metric not in self.historical_data.columns:
            raise ValueError(f"Metric {metric} not found in historical data")
            
        top_performers = self.historical_data.sort_values(by=metric, ascending=False).head(top_n)
        
        # Analyze word patterns in top performers
        all_words = []
        for subject in top_performers['subject_line']:
            all_words.extend(subject.lower().split())
            
        word_freq = {}
        for word in all_words:
            if word not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1
                
        # Store in cache for generating new subject lines
        cache_key = f"{metric}_{top_n}"
        self.metrics_cache[cache_key] = {
            'top_performers': top_performers.to_dict('records'),
            'word_frequency': word_freq,
            'avg_length': top_performers['subject_line'].str.len().mean()
        }
        
        return self.metrics_cache[cache_key]
    
    def generate_subject_lines(self, 
                              product_info: Dict[str, Any], 
                              target_audience: str,
                              campaign_type: str, 
                              num_suggestions: int = 5) -> List[Dict[str, Any]]:
        """
        Generate optimized subject lines based on product info and campaign type.
        
        Args:
            product_info: Dictionary containing product details
            target_audience: Description of the target audience
            campaign_type: Type of campaign (promotional, newsletter, etc.)
            num_suggestions: Number of subject lines to generate
            
        Returns:
            List of dictionaries with subject lines and predicted metrics
        """
        # Construct prompt for the language model
        product_name = product_info.get('name', 'product')
        product_category = product_info.get('category', '')
        prompt = f"""
        Generate {num_suggestions} engaging email subject lines for a {campaign_type} campaign 
        for {product_name} in the {product_category} category. 
        Target audience: {target_audience}.
        Make them compelling and likely to have high open rates.
        Format as a numbered list.
        """
        
        # Generate subject lines
        result = self.generator(prompt, max_length=200, num_return_sequences=num_suggestions)
        generated_text = result[0]['generated_text'] if isinstance(result, list) else result['generated_text']
        
        # Parse the numbered list
        import re
        subject_lines = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|$)', generated_text, re.DOTALL)
        subject_lines = [line.strip() for line in subject_lines]
        
        # If we didn't get enough or parsing failed, try to extract lines differently
        if len(subject_lines) < num_suggestions:
            lines = generated_text.split('\n')
            subject_lines = [line.strip().replace(f"{i+1}. ", "") for i, line in enumerate(lines) 
                           if line.strip() and not line.strip().startswith("Generate")][:num_suggestions]
        
        # Predict performance metrics for each subject line
        results = []
        for subject in subject_lines:
            # Simple heuristic prediction based on historical data
            predicted_metrics = self._predict_metrics(subject)
            results.append({
                'subject_line': subject,
                'predicted_open_rate': predicted_metrics['open_rate'],
                'predicted_click_rate': predicted_metrics['click_rate'],
                'score': predicted_metrics['score'],
                'reasoning': predicted_metrics['reasoning']
            })
            
        # Sort by predicted score
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results
    
    def _predict_metrics(self, subject_line: str) -> Dict[str, float]:
        """
        Predict performance metrics for a given subject line based on historical data.
        
        Args:
            subject_line: Subject line to evaluate
            
        Returns:
            Dictionary with predicted metrics
        """
        # Use the latest analysis results or run analysis if none exists
        if not self.metrics_cache:
            self.analyze_top_performers()
        
        cache_key = list(self.metrics_cache.keys())[-1]
        analysis = self.metrics_cache[cache_key]
        
        # Simple heuristic scoring based on word patterns
        score = 0.5  # Start with a neutral score
        reasoning = []
        
        # Check for high-performing words
        subject_words = subject_line.lower().split()
        for word in subject_words:
            if word in analysis['word_frequency'] and analysis['word_frequency'][word] > 1:
                score += 0.05
                reasoning.append(f"Contains high-performing word '{word}'")
        
        # Check length (reward being close to average length of top performers)
        avg_length = analysis['avg_length']
        subject_length = len(subject_line)
        length_diff = abs(subject_length - avg_length)
        if length_diff < 5:
            score += 0.1
            reasoning.append(f"Optimal length (close to {avg_length} characters)")
        elif length_diff < 10:
            score += 0.05
            reasoning.append("Good length")
        
        # Check for personalization indicators
        if any(term in subject_line.lower() for term in ['you', 'your', 'exclusive']):
            score += 0.08
            reasoning.append("Contains personalization elements")
        
        # Check for urgency indicators
        if any(term in subject_line.lower() for term in ['limited', 'now', 'today', 'hurry', 'last chance']):
            score += 0.07
            reasoning.append("Creates a sense of urgency")
        
        # Cap the score at 0.95
        score = min(score, 0.95)
        
        # Convert score to estimated metrics
        # This is a simplified model - in practice you would use more sophisticated ML
        predicted_open_rate = score * 0.5  # Assume max open rate around 50%
        predicted_click_rate = score * 0.2  # Assume max click rate around 20%
        
        return {
            'open_rate': round(predicted_open_rate, 2),
            'click_rate': round(predicted_click_rate, 2),
            'score': round(score, 2),
            'reasoning': reasoning
        }

# Example usage
if __name__ == "__main__":
    optimizer = SubjectLineOptimizer()
    optimizer.load_historical_data()
    
    # Analyze top performers
    analysis = optimizer.analyze_top_performers(metric="open_rate", top_n=3)
    print("Top performing subject line analysis:", json.dumps(analysis, indent=2))
    
    # Generate new subject lines
    product_info = {
        "name": "Premium Fitness Tracker",
        "category": "Wearable Technology",
        "price": 129.99,
        "features": ["Heart rate monitoring", "Sleep tracking", "Water resistant"]
    }
    
    subject_lines = optimizer.generate_subject_lines(
        product_info=product_info,
        target_audience="Health-conscious professionals aged 25-45",
        campaign_type="promotional",
        num_suggestions=5
    )
    
    print("\nGenerated subject lines:")
    for idx, result in enumerate(subject_lines, 1):
        print(f"{idx}. {result['subject_line']}")
        print(f"   Predicted open rate: {result['predicted_open_rate']:.2f}")
        print(f"   Predicted click rate: {result['predicted_click_rate']:.2f}")
        print(f"   Score: {result['score']:.2f}")
        print(f"   Reasoning: {', '.join(result['reasoning'])}")
        print() 