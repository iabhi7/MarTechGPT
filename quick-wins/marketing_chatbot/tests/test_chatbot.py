import unittest
import sys
import os
import time

# Add parent directory to path to import chatbot module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chatbot import MarketingChatbot

class TestMarketingChatbot(unittest.TestCase):
    """Test cases for the MarketingChatbot class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across all tests"""
        # Use a smaller model for testing to speed up tests
        cls.chatbot = MarketingChatbot(model_name="distilgpt2", quantize=False)
    
    def test_get_response_basic(self):
        """Test that the chatbot returns a non-empty response"""
        response = self.chatbot.get_response("Hello, how can you help with marketing?")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
    
    def test_get_response_with_context(self):
        """Test that the chatbot incorporates context"""
        context = {
            "user_name": "John",
            "interests": ["tech gadgets", "fitness"]
        }
        response = self.chatbot.get_response("What products would interest me?", context)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
    
    def test_generate_ad_copy(self):
        """Test ad copy generation"""
        product_name = "Smart Fitness Watch"
        target_audience = "Health-conscious professionals"
        key_benefits = ["Tracks heart rate", "Sleep analysis", "7-day battery life"]
        
        ad_copy = self.chatbot.generate_ad_copy(product_name, target_audience, key_benefits)
        
        self.assertIsInstance(ad_copy, str)
        self.assertTrue(len(ad_copy) > 0)
        # Check if at least one of the key benefits is mentioned
        self.assertTrue(any(benefit.lower() in ad_copy.lower() for benefit in key_benefits))
    
    def test_generate_ab_test_variants(self):
        """Test A/B test variant generation"""
        variants = self.chatbot.generate_ab_test_variants(
            "Cloud Storage Service",
            "Small business owners",
            "Secure and affordable cloud storage",
            num_variants=2
        )
        
        self.assertIsInstance(variants, list)
        self.assertTrue(len(variants) > 0)
    
    def test_performance_metrics(self):
        """Test that performance metrics are returned"""
        metrics = self.chatbot.get_performance_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("model_name", metrics)
        self.assertIn("model_size_mb", metrics)

    def test_response_time(self):
        """Test that the response time is within acceptable limits"""
        start_time = time.time()
        self.chatbot.get_response("Quick response test")
        response_time = time.time() - start_time
        
        # This threshold might need adjustment based on your hardware
        self.assertLess(response_time, 10.0, "Response time exceeds 10 seconds")

if __name__ == "__main__":
    unittest.main() 