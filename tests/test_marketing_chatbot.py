import pytest
import os
import sys
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quick_wins.marketing_chatbot.chatbot import MarketingChatbot
from quick_wins.marketing_chatbot.api import app

# Fixtures
@pytest.fixture
def chatbot():
    """Test fixture for chatbot with a small test model"""
    return MarketingChatbot(model_name="distilgpt2", quantize=True)

@pytest.fixture
def client():
    """Test client for Flask API"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Unit tests
class TestMarketingChatbot:
    def test_initialization(self, chatbot):
        """Test chatbot initialization"""
        assert chatbot is not None
        assert hasattr(chatbot, 'model')
        assert hasattr(chatbot, 'tokenizer')
    
    def test_generate_ad_copy(self, chatbot):
        """Test ad copy generation"""
        ad_copy = chatbot.generate_ad_copy(
            "Cloud Storage Service", 
            "Small business owners",
            ["Secure", "Affordable", "Easy to use"]
        )
        assert isinstance(ad_copy, str)
        assert len(ad_copy) > 50
        # Check that at least one key benefit is mentioned
        assert any(benefit.lower() in ad_copy.lower() for benefit in ["secure", "affordable", "easy"])
    
    def test_generate_ab_test_variants(self, chatbot):
        """Test A/B test variant generation"""
        variants = chatbot.generate_ab_test_variants(
            "Email Marketing Platform",
            "Digital marketers",
            "Increase open rates and engagement",
            num_variants=2
        )
        assert isinstance(variants, list)
        assert len(variants) >= 2
        assert all(isinstance(v, str) for v in variants)
    
    def test_model_quantization(self):
        """Test that quantization reduces model size"""
        # Create unquantized model for comparison
        unquantized_model = MarketingChatbot(model_name="distilgpt2", quantize=False)
        quantized_model = MarketingChatbot(model_name="distilgpt2", quantize=True)
        
        # Check memory footprint (this is an approximation)
        unquantized_memory = unquantized_model.get_performance_metrics()["model_size_mb"]
        quantized_memory = quantized_model.get_performance_metrics()["model_size_mb"]
        
        # Quantized model should be smaller
        assert quantized_memory < unquantized_memory

# Integration tests
class TestMarketingChatbotAPI:
    def test_health_endpoint(self, client):
        """Test API health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        assert response.json['status'] == 'healthy'
    
    def test_chat_endpoint(self, client):
        """Test chat endpoint with valid input"""
        test_input = {
            "message": "How can I improve email open rates?",
            "context": {"user_name": "Test User", "interests": ["email marketing"]}
        }
        response = client.post('/chat', json=test_input)
        assert response.status_code == 200
        assert 'response' in response.json
        assert response.json['status'] == 'success'
    
    def test_generate_ad_endpoint(self, client):
        """Test ad generation endpoint"""
        test_input = {
            "product_name": "Marketing Analytics Tool",
            "target_audience": "CMOs and Marketing Directors",
            "key_benefits": ["Real-time insights", "Easy integration", "AI-powered"]
        }
        response = client.post('/generate_ad', json=test_input)
        assert response.status_code == 200
        assert 'ad_copy' in response.json
        assert response.json['status'] == 'success'
    
    def test_error_handling(self, client):
        """Test API error handling with invalid input"""
        # Missing required field
        test_input = {"context": {"user_name": "Test User"}}
        response = client.post('/chat', json=test_input)
        assert response.status_code == 400
        assert 'error' in response.json 