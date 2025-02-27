import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import json
import os

class CustomerSegmentAnalyzer:
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        """
        Initialize the Customer Segment Analyzer.
        
        Args:
            n_clusters: Number of segments to create
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.segments = None
        self.feature_importance = {}
        
        # Initialize the text classification pipeline for analyzing customer feedback
        self.nlp_classifier = pipeline("text-classification", 
                                      model="distilbert-base-uncased-finetuned-sst-2-english")
        
    def load_data(self, 
                 filepath: Optional[str] = None, 
                 data: Optional[pd.DataFrame] = None,
                 netcore_api_key: Optional[str] = None,
                 sample_size: int = 1000) -> pd.DataFrame:
        """
        Load customer data from file or DataFrame.
        
        Args:
            filepath: Path to CSV file with customer data
            data: Pandas DataFrame with customer data
            netcore_api_key: API key for Netcore integration
            sample_size: Size of sample data to create if no data provided
            
        Returns:
            DataFrame containing the loaded customer data
        """
        if data is not None:
            self.data = data
            print(f"Data loaded with {len(self.data)} records")
            return self.data
            
        if filepath and os.path.exists(filepath):
            self.data = pd.read_csv(filepath)
            print(f"Data loaded from {filepath} with {len(self.data)} records")
            return self.data
            
        if netcore_api_key:
            # This would be replaced with actual API call to Netcore
            print("Fetching data from Netcore API (mock)")
            self.data = self._create_sample_data(sample_size)
            self.data['source'] = 'netcore_api'
            return self.data
            
        # Create sample data if no source provided
        print("Creating sample customer data")
        self.data = self._create_sample_data(sample_size)
        return self.data
    
    def _create_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Create sample customer data for demonstration purposes.
        
        Args:
            n_samples: Number of sample records to create
            
        Returns:
            DataFrame with synthetic customer data
        """
        np.random.seed(self.random_state)
        
        # Create basic customer data
        data = {
            'customer_id': [f'CUST{i:05d}' for i in range(1, n_samples + 1)],
            'recency': np.random.randint(1, 100, n_samples),  # Days since last purchase
            'frequency': np.random.randint(1, 50, n_samples),  # Number of purchases
            'monetary': np.random.normal(500, 300, n_samples),  # Total spend
            'tenure': np.random.randint(1, 730, n_samples),  # Days as customer
            'product_categories': np.random.randint(1, 8, n_samples),  # Number of categories purchased
            'discount_usage': np.random.randint(0, 15, n_samples),  # Number of discount codes used
            'email_open_rate': np.random.uniform(0, 1, n_samples),  # Email engagement
            'app_sessions': np.random.randint(0, 100, n_samples),  # App engagement
            'cart_abandonment_rate': np.random.uniform(0, 1, n_samples),  # Cart abandonments
            'returns': np.random.randint(0, 10, n_samples),  # Number of returns
        }
        
        # Add some behavioral data based on the numerics
        data['web_visits'] = (data['app_sessions'] * np.random.uniform(0.5, 3, n_samples)).astype(int)
        
        # Add some realistic-looking categorical data
        devices = ['Mobile', 'Desktop', 'Tablet']
        device_weights = [0.6, 0.3, 0.1]
        data['primary_device'] = [np.random.choice(devices, p=device_weights) for _ in range(n_samples)]
        
        channels = ['Direct', 'Organic Search', 'Paid Search', 'Social', 'Email', 'Referral']
        channel_weights = [0.2, 0.25, 0.15, 0.2, 0.15, 0.05]
        data['acquisition_channel'] = [np.random.choice(channels, p=channel_weights) for _ in range(n_samples)]
        
        # Add some customer feedback data
        feedback_templates = [
            "I love your {product}! It's {adjective}.",
            "The {product} was {adjective}, but shipping took too long.",
            "Not satisfied with my recent purchase of {product}.",
            "Your customer service is {adjective}.",
            "Will definitely buy more {product} from you.",
            "Had issues with my order but your team was {adjective} in resolving it.",
            "The mobile app is {adjective} to use."
        ]
        
        products = ['shoes', 'shirt', 'accessories', 'electronics', 'subscription']
        positive_adj = ['amazing', 'excellent', 'outstanding', 'great', 'wonderful']
        negative_adj = ['disappointing', 'frustrating', 'poor', 'terrible', 'subpar']
        
        feedbacks = []
        sentiments = []
        
        for _ in range(n_samples):
            template = np.random.choice(feedback_templates)
            product = np.random.choice(products)
            
            # Higher monetary value customers tend to have more positive feedback
            if data['monetary'][_] > 500 and np.random.random() < 0.7:
                adj = np.random.choice(positive_adj)
                sentiment = 'positive'
            else:
                # Randomly choose positive or negative
                if np.random.random() < 0.4:
                    adj = np.random.choice(negative_adj)
                    sentiment = 'negative'
                else:
                    adj = np.random.choice(positive_adj)
                    sentiment = 'positive'
                    
            feedback = template.format(product=product, adjective=adj)
            feedbacks.append(feedback)
            sentiments.append(sentiment)
            
        data['feedback'] = feedbacks
        data['sentiment'] = sentiments
        
        return pd.DataFrame(data)
    
    def preprocess(self, 
                  numerical_features: Optional[List[str]] = None,
                  categorical_features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Preprocess the data for segmentation.
        
        Args:
            numerical_features: List of numerical feature columns
            categorical_features: List of categorical feature columns
            
        Returns:
            Tuple of (processed DataFrame, list of all features used)
        """
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # If no features specified, use defaults based on data
        if numerical_features is None:
            numerical_features = [
                'recency', 'frequency', 'monetary', 'tenure', 
                'product_categories', 'discount_usage', 'email_open_rate',
                'app_sessions', 'cart_abandonment_rate', 'returns', 'web_visits'
            ]
        
        if categorical_features is None:
            categorical_features = ['primary_device', 'acquisition_channel']
            
        # Ensure all features exist in the data
        all_features = []
        for feature in numerical_features:
            if feature in self.data.columns:
                all_features.append(feature)
            else:
                print(f"Warning: Feature '{feature}' not found in data")
                
        # Handle missing values
        self.data_processed = self.data.copy()
        for feature in all_features:
            self.data_processed[feature] = self.data_processed[feature].fillna(self.data_processed[feature].median())
            
        # Scale numerical features
        self.data_scaled = self.data_processed.copy()
        self.data_scaled[all_features] = self.scaler.fit_transform(self.data_processed[all_features])
        
        # Process categorical features using one-hot encoding
        if categorical_features:
            cat_features_to_use = [f for f in categorical_features if f in self.data.columns]
            if cat_features_to_use:
                self.data_encoded = pd.get_dummies(self.data_scaled, columns=cat_features_to_use, drop_first=False)
                # Get the new column names from one-hot encoding
                new_cols = [c for c in self.data_encoded.columns if c not in self.data_scaled.columns]
                all_features.extend(new_cols)
            else:
                self.data_encoded = self.data_scaled
        else:
            self.data_encoded = self.data_scaled
            
        self.features_used = all_features
        print(f"Data preprocessed with {len(all_features)} features")
        return self.data_encoded, all_features
    
    def create_segments(self, features_to_use: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create customer segments using KMeans clustering.
        
        Args:
            features_to_use: Specific features to use for clustering
            
        Returns:
            DataFrame with original data and segment labels
        """
        if not hasattr(self, 'data_encoded') or self.data_encoded is None:
            self.preprocess()
            
        # Use specified features or all preprocessed features
        if features_to_use:
            X = self.data_encoded[features_to_use].values
            self.features_used = features_to_use
        else:
            X = self.data_encoded[self.features_used].values
            
        # Fit KMeans model
        self.model.fit(X)
        
        # Get cluster labels
        self.segments = self.data.copy()
        self.segments['segment'] = self.model.labels_
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        # Assign descriptive segment names
        self._assign_segment_names()
        
        print(f"Created {self.n_clusters} customer segments")
        return self.segments
    
    def _calculate_feature_importance(self):
        """Calculate the importance of each feature in defining the segments"""
        if not hasattr(self, 'model') or not hasattr(self, 'features_used'):
            raise ValueError("Model not trained or features not defined")
            
        # Get cluster centers
        centers = self.model.cluster_centers_
        
        # Calculate feature importance based on distance from global mean
        global_mean = np.mean(self.data_encoded[self.features_used].values, axis=0)
        
        importance = {}
        for i, feature in enumerate(self.features_used):
            # Calculate the variance of this feature across clusters
            feature_variance = np.var([center[i] for center in centers])
            importance[feature] = feature_variance
            
        # Normalize importance scores
        total = sum(importance.values())
        self.feature_importance = {k: v/total for k, v in importance.items()}
    
    def _assign_segment_names(self):
        """Assign descriptive names to segments based on their characteristics"""
        if not hasattr(self, 'segments') or self.segments is None:
            raise ValueError("Segments not created yet")
            
        # Get average values for key metrics by segment
        segment_profiles = self.segments.groupby('segment').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'email_open_rate': 'mean',
            'app_sessions': 'mean',
            'cart_abandonment_rate': 'mean'
        })
        
        # Analyze sentiment if available
        if 'sentiment' in self.segments.columns:
            sentiment_by_segment = self.segments.groupby('segment')['sentiment'].apply(
                lambda x: (x == 'positive').mean()
            )
            segment_profiles['positive_sentiment_rate'] = sentiment_by_segment
        
        # Create segment names based on key characteristics
        segment_names = {}
        
        for segment_id, profile in segment_profiles.iterrows():
            # Determine key characteristics
            if profile['monetary'] > segment_profiles['monetary'].median():
                if profile['frequency'] > segment_profiles['frequency'].median():
                    if profile['recency'] < segment_profiles['recency'].median():
                        name = "High-Value Active Customers"
                    else:
                        name = "High-Value At-Risk Customers"
                else:
                    name = "Big Spenders (Infrequent)"
            else:
                if profile['frequency'] > segment_profiles['frequency'].median():
                    if profile['recency'] < segment_profiles['recency'].median():
                        name = "Frequent Low-Value Customers"
                    else:
                        name = "Formerly Active Customers"
                else:
                    if profile['recency'] < segment_profiles['recency'].median():
                        name = "New or One-time Customers"
                    else:
                        name = "Inactive Customers"
                        
            # Consider engagement metrics
            if profile['email_open_rate'] > segment_profiles['email_open_rate'].median() and \
               profile['app_sessions'] > segment_profiles['app_sessions'].median():
                name += " (Highly Engaged)"
            elif profile['cart_abandonment_rate'] > segment_profiles['cart_abandonment_rate'].median():
                name += " (High Cart Abandonment)"
                
            segment_names[segment_id] = name
            
        # Add segment names to the data
        self.segments['segment_name'] = self.segments['segment'].map(segment_names)
        self.segment_profiles = segment_profiles
        self.segment_names = segment_names
    
    def analyze_segments(self) -> Dict[str, Any]:
        """
        Analyze the characteristics of each segment.
        
        Returns:
            Dictionary with segment analysis results
        """
        if not hasattr(self, 'segments') or self.segments is None:
            raise ValueError("Segments not created yet")
            
        # Calculate segment sizes
        segment_sizes = self.segments['segment'].value_counts(normalize=True).to_dict()
        
        # Get average values for key metrics by segment
        segment_metrics = self.segments.groupby('segment_name').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'tenure': 'mean',
            'email_open_rate': 'mean',
            'app_sessions': 'mean',
            'cart_abandonment_rate': 'mean',
            'returns': 'mean'
        }).round(2).to_dict()
        
        # Analyze popular channels by segment
        channel_analysis = {}
        if 'acquisition_channel' in self.segments.columns:
            for segment in self.segment_names.values():
                segment_data = self.segments[self.segments['segment_name'] == segment]
                channel_counts = segment_data['acquisition_channel'].value_counts(normalize=True)
                channel_analysis[segment] = channel_counts.to_dict()
                
        # Analyze sentiment by segment
        sentiment_analysis = {}
        if 'sentiment' in self.segments.columns:
            for segment in self.segment_names.values():
                segment_data = self.segments[self.segments['segment_name'] == segment]
                sentiment_counts = segment_data['sentiment'].value_counts(normalize=True)
                sentiment_analysis[segment] = sentiment_counts.to_dict()
        
        # Create marketing recommendations for each segment
        recommendations = self._generate_recommendations()
        
        # Compile the analysis results
        analysis = {
            'segment_sizes': segment_sizes,
            'segment_metrics': segment_metrics,
            'channel_analysis': channel_analysis,
            'sentiment_analysis': sentiment_analysis,
            'feature_importance': self.feature_importance,
            'recommendations': recommendations
        }
        
        return analysis
    
    def _generate_recommendations(self) -> Dict[str, List[str]]:
        """
        Generate marketing recommendations for each segment.
        
        Returns:
            Dictionary with segment names as keys and list of recommendations as values
        """
        if not hasattr(self, 'segments') or self.segments is None:
            raise ValueError("Segments not created yet")
            
        recommendations = {}
        
        for segment_id, segment_name in self.segment_names.items():
            segment_recs = []
            
            # Get segment profile
            profile = self.segment_profiles.loc[segment_id]
            
            # Add recommendations based on segment characteristics
            if "High-Value Active" in segment_name:
                segment_recs.extend([
                    "Implement a VIP loyalty program with exclusive perks",
                    "Create personalized thank-you messages from the CEO",
                    "Offer early access to new products and features",
                    "Develop a referral program with premium incentives"
                ])
                
            elif "High-Value At-Risk" in segment_name:
                segment_recs.extend([
                    "Send personalized win-back emails with special offers",
                    "Implement a customer feedback survey with incentives",
                    "Create a re-engagement campaign with high-value discounts",
                    "Assign a dedicated customer success manager"
                ])
                
            elif "Big Spenders" in segment_name:
                segment_recs.extend([
                    "Create bundle offers to increase purchase frequency",
                    "Implement a subscription model for recurring purchases",
                    "Develop personalized product recommendations",
                    "Send reminder emails for seasonal purchases"
                ])
                
            elif "Frequent Low-Value" in segment_name:
                segment_recs.extend([
                    "Create upsell campaigns for higher-value products",
                    "Implement tiered discounts based on cart value",
                    "Develop a loyalty program with points for each purchase",
                    "Send educational content about premium product benefits"
                ])
                
            elif "Formerly Active" in segment_name:
                segment_recs.extend([
                    "Launch a re-engagement campaign with personalized offers",
                    "Implement a 'We miss you' email series",
                    "Create a feedback survey to understand reasons for inactivity",
                    "Offer a special discount for returning customers"
                ])
                
            elif "New or One-time" in segment_name:
                segment_recs.extend([
                    "Create an onboarding email series",
                    "Develop a 'second purchase' discount offer",
                    "Implement personalized product recommendations",
                    "Send educational content about product benefits"
                ])
                
            elif "Inactive" in segment_name:
                segment_recs.extend([
                    "Launch a win-back campaign with high-value offers",
                    "Implement a 'Last chance' email series",
                    "Create a feedback survey with incentives",
                    "Consider removing from regular email campaigns"
                ])
                
            # Add engagement-specific recommendations
            if "Highly Engaged" in segment_name:
                segment_recs.extend([
                    "Leverage as brand ambassadors on social media",
                    "Invite to beta test new features",
                    "Create exclusive content and experiences",
                    "Develop a community program for engaged users"
                ])
                
            elif "High Cart Abandonment" in segment_name:
                segment_recs.extend([
                    "Optimize the checkout process for this segment",
                    "Implement targeted cart abandonment emails",
                    "Create special offers for abandoned cart items",
                    "Add live chat support during checkout"
                ])
                
            recommendations[segment_name] = segment_recs
            
        return recommendations
    
    def visualize_segments(self, 
                          feature_x: str = 'recency', 
                          feature_y: str = 'monetary',
                          feature_size: Optional[str] = 'frequency',
                          save_path: Optional[str] = None) -> None:
        """
        Visualize customer segments in a scatter plot.
        
        Args:
            feature_x: Feature to plot on x-axis
            feature_y: Feature to plot on y-axis
            feature_size: Feature to determine point size (optional)
            save_path: Path to save the visualization (optional)
        """
        if not hasattr(self, 'segments') or self.segments is None:
            raise ValueError("Segments not created yet")
            
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        for segment_name in self.segments['segment_name'].unique():
            segment_data = self.segments[self.segments['segment_name'] == segment_name]
            
            if feature_size is not None and feature_size in self.segments.columns:
                # Normalize size feature to reasonable point sizes
                sizes = 20 + (segment_data[feature_size] - self.segments[feature_size].min()) / \
                        (self.segments[feature_size].max() - self.segments[feature_size].min()) * 100
            else:
                sizes = 50
                
            plt.scatter(
                segment_data[feature_x],
                segment_data[feature_y],
                s=sizes,
                alpha=0.7,
                label=segment_name
            )
            
        # Add cluster centers
        if hasattr(self, 'model') and feature_x in self.features_used and feature_y in self.features_used:
            # Get indices of the features in the feature list
            x_idx = self.features_used.index(feature_x)
            y_idx = self.features_used.index(feature_y)
            
            # Get cluster centers and transform back to original scale
            centers = self.model.cluster_centers_
            # Assuming the scaler was applied to all features
            x_center_original = centers[:, x_idx] * self.scaler.scale_[x_idx] + self.scaler.mean_[x_idx]
            y_center_original = centers[:, y_idx] * self.scaler.scale_[y_idx] + self.scaler.mean_[y_idx]
            
            plt.scatter(
                x_center_original,
                y_center_original,
                s=200,
                c='black',
                marker='X',
                alpha=1,
                label='Cluster Centers'
            )
            
        plt.title(f'Customer Segments: {feature_x} vs {feature_y}', fontsize=15)
        plt.xlabel(feature_x.replace('_', ' ').title(), fontsize=12)
        plt.ylabel(feature_y.replace('_', ' ').title(), fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
            
        plt.show()
        
    def export_segments(self, 
                       filepath: str = 'customer_segments.csv',
                       netcore_api_key: Optional[str] = None) -> None:
        """
        Export customer segments to CSV or Netcore platform.
        
        Args:
            filepath: Path to save the CSV file
            netcore_api_key: API key for Netcore integration
        """
        if not hasattr(self, 'segments') or self.segments is None:
            raise ValueError("Segments not created yet")
            
        # Export to CSV
        self.segments.to_csv(filepath, index=False)
        print(f"Segments exported to {filepath}")
        
        # If Netcore API key is provided, upload to Netcore
        if netcore_api_key:
            print("Uploading segments to Netcore (example integration)")
            # This would be replaced with actual API call to Netcore
            # In a real implementation, you would use Netcore's API to upload segments
            print("Segments would be uploaded to Netcore for campaign targeting")
    
    def get_segment_for_customer(self, customer_id: str) -> Dict[str, Any]:
        """
        Get segment information for a specific customer.
        
        Args:
            customer_id: ID of the customer
            
        Returns:
            Dictionary with customer segment information
        """
        if not hasattr(self, 'segments') or self.segments is None:
            raise ValueError("Segments not created yet")
            
        if customer_id not in self.segments['customer_id'].values:
            raise ValueError(f"Customer {customer_id} not found in data")
            
        customer_data = self.segments[self.segments['customer_id'] == customer_id].iloc[0]
        segment_name = customer_data['segment_name']
        
        # Get segment-level recommendations
        if hasattr(self, 'recommendations'):
            recommendations = self.recommendations.get(segment_name, [])
        else:
            recommendations = self._generate_recommendations().get(segment_name, [])
            
        result = {
            'customer_id': customer_id,
            'segment': customer_data['segment'],
            'segment_name': segment_name,
            'key_metrics': {
                'recency': customer_data['recency'],
                'frequency': customer_data['frequency'],
                'monetary': customer_data['monetary'],
                'email_open_rate': customer_data['email_open_rate']
            },
            'recommendations': recommendations
        }
        
        return result

# Example usage
if __name__ == "__main__":
    # Initialize the segment analyzer
    analyzer = CustomerSegmentAnalyzer(n_clusters=5)
    
    # Load and preprocess data
    data = analyzer.load_data(sample_size=1000)
    analyzer.preprocess()
    
    # Create segments
    segments = analyzer.create_segments()
    
    # Analyze segments
    analysis = analyzer.analyze_segments()
    
    # Print segment summary
    for segment_name, size in analysis['segment_sizes'].items():
        segment_name_str = analyzer.segment_names[segment_name]
        print(f"Segment {segment_name} ({segment_name_str}): {size*100:.1f}% of customers")
        
    # Visualize segments
    analyzer.visualize_segments(feature_x='recency', feature_y='monetary', feature_size='frequency')
    
    # Export segments
    analyzer.export_segments() 