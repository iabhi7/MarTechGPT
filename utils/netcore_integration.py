import requests
import json
import pandas as pd
import os
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('netcore_integration')

class NetcoreIntegration:
    """
    Class for integrating with Netcore Cloud's API.
    Allows bidirectional data flow between Netcore and the AI modules.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.netcore.co.in"):
        """
        Initialize Netcore Integration.
        
        Args:
            api_key: Netcore API key (will use from env if not provided)
            base_url: Base URL for Netcore API
        """
        self.api_key = api_key or os.getenv("NETCORE_API_KEY")
        if not self.api_key:
            logger.warning("No Netcore API key provided. Set NETCORE_API_KEY env variable or pass api_key.")
            
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info("Netcore integration initialized")
        
    def test_connection(self) -> bool:
        """
        Test connection to Netcore API.
        
        Returns:
            True if connection is successful, False otherwise
        """
        if not self.api_key:
            logger.error("Cannot test connection. No API key provided.")
            return False
            
        try:
            response = requests.get(
                f"{self.base_url}/v1/status",
                headers=self.headers
            )
            response.raise_for_status()
            logger.info("Successfully connected to Netcore API")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to connect to Netcore API: {e}")
            return False
    
    def fetch_campaign_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch campaign data from Netcore for analysis.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with campaign data
        """
        if not self.api_key:
            logger.error("Cannot fetch data. No API key provided.")
            return pd.DataFrame()
            
        try:
            # This is a mockup of how the API call would work
            # In a real implementation, adjust the endpoint and parameters
            response = requests.get(
                f"{self.base_url}/campaigns/performance",
                headers=self.headers,
                params={
                    "start_date": start_date,
                    "end_date": end_date
                }
            )
            response.raise_for_status()
            
            # Parse response into DataFrame
            campaigns = response.json().get("campaigns", [])
            df = pd.DataFrame(campaigns)
            
            logger.info(f"Fetched {len(df)} campaigns from Netcore")
            return df
        except requests.RequestException as e:
            logger.error(f"Failed to fetch campaign data: {e}")
            # Return empty DataFrame
            return pd.DataFrame()
            
    def fetch_customer_data(self, segment: Optional[str] = None, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch customer data from Netcore for analysis.
        
        Args:
            segment: Optional segment filter
            limit: Maximum number of records to fetch
            
        Returns:
            DataFrame with customer data
        """
        if not self.api_key:
            logger.error("Cannot fetch data. No API key provided.")
            return pd.DataFrame()
            
        try:
            # This is a mockup of how the API call would work
            params = {"limit": limit}
            if segment:
                params["segment"] = segment
                
            response = requests.get(
                f"{self.base_url}/customers",
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            
            # Parse response into DataFrame
            customers = response.json().get("customers", [])
            df = pd.DataFrame(customers)
            
            logger.info(f"Fetched {len(df)} customers from Netcore")
            return df
        except requests.RequestException as e:
            logger.error(f"Failed to fetch customer data: {e}")
            # Return empty DataFrame
            return pd.DataFrame()
    
    def upload_customer_segments(self, segments: pd.DataFrame) -> bool:
        """
        Upload customer segments to Netcore for targeting.
        
        Args:
            segments: DataFrame with customer IDs and segment names
            
        Returns:
            True if upload is successful, False otherwise
        """
        if not self.api_key:
            logger.error("Cannot upload segments. No API key provided.")
            return False
            
        if 'customer_id' not in segments.columns or 'segment_name' not in segments.columns:
            logger.error("Segments DataFrame must contain 'customer_id' and 'segment_name' columns")
            return False
            
        try:
            # Convert DataFrame to expected API format
            segments_data = segments.to_dict('records')
            
            # This is a mockup of how the API call would work
            response = requests.post(
                f"{self.base_url}/segments/batch",
                headers=self.headers,
                json={"segments": segments_data}
            )
            response.raise_for_status()
            
            logger.info(f"Successfully uploaded {len(segments)} customer segments to Netcore")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to upload customer segments: {e}")
            return False
    
    def upload_content_suggestions(self, 
                                  content_type: str, 
                                  content_data: List[Dict[str, Any]],
                                  campaign_id: Optional[str] = None) -> bool:
        """
        Upload AI-generated content suggestions to Netcore.
        
        Args:
            content_type: Type of content ('email', 'push', 'sms', 'social', 'ad')
            content_data: List of content suggestions
            campaign_id: Optional campaign ID to associate with content
            
        Returns:
            True if upload is successful, False otherwise
        """
        if not self.api_key:
            logger.error("Cannot upload content. No API key provided.")
            return False
            
        try:
            # Prepare payload
            payload = {
                "content_type": content_type,
                "suggestions": content_data
            }
            
            if campaign_id:
                payload["campaign_id"] = campaign_id
                
            # This is a mockup of how the API call would work
            response = requests.post(
                f"{self.base_url}/content/suggestions",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
            logger.info(f"Successfully uploaded {len(content_data)} {content_type} content suggestions to Netcore")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to upload content suggestions: {e}")
            return False
    
    def upload_campaign_recommendations(self, 
                                      recommendations: Dict[str, Any],
                                      campaign_id: Optional[str] = None) -> bool:
        """
        Upload campaign optimization recommendations to Netcore.
        
        Args:
            recommendations: Dictionary of campaign recommendations
            campaign_id: Optional specific campaign ID
            
        Returns:
            True if upload is successful, False otherwise
        """
        if not self.api_key:
            logger.error("Cannot upload recommendations. No API key provided.")
            return False
            
        try:
            # Prepare payload
            payload = {"recommendations": recommendations}
            
            if campaign_id:
                payload["campaign_id"] = campaign_id
                
            # This is a mockup of how the API call would work
            response = requests.post(
                f"{self.base_url}/campaigns/recommendations",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
            logger.info("Successfully uploaded campaign recommendations to Netcore")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to upload campaign recommendations: {e}")
            return False
    
    def upload_churn_predictions(self, 
                               predictions: pd.DataFrame,
                               include_recommendations: bool = True) -> bool:
        """
        Upload churn predictions to Netcore for targeted campaigns.
        
        Args:
            predictions: DataFrame with customer IDs and churn probabilities
            include_recommendations: Whether to include prevention recommendations
            
        Returns:
            True if upload is successful, False otherwise
        """
        if not self.api_key:
            logger.error("Cannot upload churn predictions. No API key provided.")
            return False
            
        if 'customer_id' not in predictions.columns or 'predicted_churn_probability' not in predictions.columns:
            logger.error("Predictions DataFrame must contain 'customer_id' and 'predicted_churn_probability' columns")
            return False
            
        try:
            # Prepare data in Netcore-compatible format
            prediction_data = []
            
            for _, row in predictions.iterrows():
                prediction = {
                    "customer_id": row['customer_id'],
                    "churn_probability": float(row['predicted_churn_probability']),
                    "risk_level": row.get('risk_level', 'unknown')
                }
                
                if include_recommendations and 'recommendations' in predictions.columns:
                    prediction["recommendations"] = row['recommendations'].split("; ")
                    
                prediction_data.append(prediction)
                
            # This is a mockup of how the API call would work
            response = requests.post(
                f"{self.base_url}/predictive/churn",
                headers=self.headers,
                json={"predictions": prediction_data}
            )
            response.raise_for_status()
            
            logger.info(f"Successfully uploaded {len(predictions)} churn predictions to Netcore")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to upload churn predictions: {e}")
            return False 