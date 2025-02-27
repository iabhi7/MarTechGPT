import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os
import json
from datetime import datetime, timedelta

class CampaignWorkflowOptimizer:
    def __init__(self, netcore_api_key: Optional[str] = None):
        """
        Initialize the Campaign Workflow Optimizer.
        
        Args:
            netcore_api_key: API key for Netcore integration
        """
        self.netcore_api_key = netcore_api_key
        self.model = None
        self.preprocessor = None
        self.feature_importance = {}
        self.best_workflows = {}
        
        print("Campaign Workflow Optimizer initialized")
        
    def load_campaign_data(self, 
                          filepath: Optional[str] = None, 
                          data: Optional[pd.DataFrame] = None,
                          use_netcore_api: bool = False,
                          sample_size: int = 500) -> pd.DataFrame:
        """
        Load historical campaign data.
        
        Args:
            filepath: Path to CSV file with campaign data
            data: Pandas DataFrame with campaign data
            use_netcore_api: Whether to fetch data from Netcore API
            sample_size: Size of sample data to create if no data provided
            
        Returns:
            DataFrame containing the loaded campaign data
        """
        if data is not None:
            self.data = data
            print(f"Data loaded with {len(self.data)} campaign records")
            return self.data
            
        if filepath and os.path.exists(filepath):
            self.data = pd.read_csv(filepath)
            print(f"Data loaded from {filepath} with {len(self.data)} campaign records")
            return self.data
            
        if use_netcore_api and self.netcore_api_key:
            # This would be replaced with actual API call to Netcore
            print("Fetching campaign data from Netcore API (mock)")
            self.data = self._create_sample_data(sample_size)
            self.data['source'] = 'netcore_api'
            return self.data
            
        # Create sample data if no source provided
        print("Creating sample campaign data")
        self.data = self._create_sample_data(sample_size)
        return self.data
    
    def _create_sample_data(self, n_samples: int = 500) -> pd.DataFrame:
        """
        Create sample campaign data for demonstration purposes.
        
        Args:
            n_samples: Number of sample records to create
            
        Returns:
            DataFrame with synthetic campaign data
        """
        np.random.seed(42)
        
        # Create campaign IDs and dates
        campaign_ids = [f'CAMP{i:05d}' for i in range(1, n_samples + 1)]
        
        # End dates are within the last 6 months
        end_dates = [
            (datetime.now() - timedelta(days=np.random.randint(1, 180))).strftime('%Y-%m-%d')
            for _ in range(n_samples)
        ]
        
        # Campaign durations between 1 and 30 days
        durations = np.random.randint(1, 30, n_samples)
        
        # Create start dates based on end dates and durations
        start_dates = [
            (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=duration)).strftime('%Y-%m-%d')
            for end_date, duration in zip(end_dates, durations)
        ]
        
        # Campaign types
        campaign_types = np.random.choice(
            ['promotional', 'newsletter', 'product_launch', 'seasonal', 're_engagement'],
            n_samples,
            p=[0.35, 0.25, 0.15, 0.15, 0.1]
        )
        
        # Target segments
        target_segments = np.random.choice(
            ['all_customers', 'high_value', 'new_customers', 'inactive', 'cart_abandoners'],
            n_samples,
            p=[0.2, 0.25, 0.2, 0.15, 0.2]
        )
        
        # Channels used
        email_usage = []
        sms_usage = []
        push_usage = []
        in_app_usage = []
        social_usage = []
        
        for c_type in campaign_types:
            # Different campaign types have different channel preferences
            if c_type == 'promotional':
                email_usage.append(np.random.choice([0, 1], p=[0.1, 0.9]))
                sms_usage.append(np.random.choice([0, 1], p=[0.5, 0.5]))
                push_usage.append(np.random.choice([0, 1], p=[0.7, 0.3]))
                in_app_usage.append(np.random.choice([0, 1], p=[0.7, 0.3]))
                social_usage.append(np.random.choice([0, 1], p=[0.5, 0.5]))
            elif c_type == 'newsletter':
                email_usage.append(np.random.choice([0, 1], p=[0.05, 0.95]))
                sms_usage.append(np.random.choice([0, 1], p=[0.9, 0.1]))
                push_usage.append(np.random.choice([0, 1], p=[0.7, 0.3]))
                in_app_usage.append(np.random.choice([0, 1], p=[0.5, 0.5]))
                social_usage.append(np.random.choice([0, 1], p=[0.7, 0.3]))
            elif c_type == 'product_launch':
                email_usage.append(np.random.choice([0, 1], p=[0.1, 0.9]))
                sms_usage.append(np.random.choice([0, 1], p=[0.6, 0.4]))
                push_usage.append(np.random.choice([0, 1], p=[0.4, 0.6]))
                in_app_usage.append(np.random.choice([0, 1], p=[0.3, 0.7]))
                social_usage.append(np.random.choice([0, 1], p=[0.2, 0.8]))
            elif c_type == 'seasonal':
                email_usage.append(np.random.choice([0, 1], p=[0.2, 0.8]))
                sms_usage.append(np.random.choice([0, 1], p=[0.4, 0.6]))
                push_usage.append(np.random.choice([0, 1], p=[0.5, 0.5]))
                in_app_usage.append(np.random.choice([0, 1], p=[0.5, 0.5]))
                social_usage.append(np.random.choice([0, 1], p=[0.3, 0.7]))
            else:  # re_engagement
                email_usage.append(np.random.choice([0, 1], p=[0.1, 0.9]))
                sms_usage.append(np.random.choice([0, 1], p=[0.5, 0.5]))
                push_usage.append(np.random.choice([0, 1], p=[0.3, 0.7]))
                in_app_usage.append(np.random.choice([0, 1], p=[0.6, 0.4]))
                social_usage.append(np.random.choice([0, 1], p=[0.8, 0.2]))
        
        # Number of steps in the workflow (1-6)
        workflow_steps = np.random.randint(1, 7, n_samples)
        
        # Send time optimization usage
        send_time_optimization = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        
        # A/B testing usage
        ab_testing = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
        
        # AI content generation
        ai_content = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
        
        # Personalization level (0-5)
        personalization_level = np.random.randint(0, 6, n_samples)
        
        # Create performance metrics with some relationships to the features
        # Base conversion rate influenced by campaign type and target segment
        base_conversion = np.zeros(n_samples)
        
        for i, (c_type, segment) in enumerate(zip(campaign_types, target_segments)):
            # Different campaign types have different base conversion rates
            if c_type == 'promotional':
                base_conversion[i] = 0.02
            elif c_type == 'newsletter':
                base_conversion[i] = 0.01
            elif c_type == 'product_launch':
                base_conversion[i] = 0.03
            elif c_type == 'seasonal':
                base_conversion[i] = 0.025
            else:  # re_engagement
                base_conversion[i] = 0.015
                
            # Adjust based on segment
            if segment == 'high_value':
                base_conversion[i] *= 1.5
            elif segment == 'new_customers':
                base_conversion[i] *= 0.8
            elif segment == 'inactive':
                base_conversion[i] *= 0.5
            elif segment == 'cart_abandoners':
                base_conversion[i] *= 2.0
        
        # Adjustments based on other features
        conversion_multiplier = np.ones(n_samples)
        
        # More channels = better conversion (up to a point)
        for i in range(n_samples):
            channel_count = email_usage[i] + sms_usage[i] + push_usage[i] + in_app_usage[i] + social_usage[i]
            if channel_count == 1:
                conversion_multiplier[i] *= 1.0
            elif channel_count == 2:
                conversion_multiplier[i] *= 1.2
            elif channel_count == 3:
                conversion_multiplier[i] *= 1.3
            elif channel_count == 4:
                conversion_multiplier[i] *= 1.35
            else:  # all 5 channels
                conversion_multiplier[i] *= 1.4
        
        # Send time optimization helps
        conversion_multiplier[send_time_optimization == 1] *= 1.15
        
        # A/B testing helps
        conversion_multiplier[ab_testing == 1] *= 1.1
        
        # AI content helps
        conversion_multiplier[ai_content == 1] *= 1.2
        
        # More personalization helps
        for i, level in enumerate(personalization_level):
            conversion_multiplier[i] *= (1 + level * 0.05)
        
        # More workflow steps help, but with diminishing returns
        for i, steps in enumerate(workflow_steps):
            conversion_multiplier[i] *= (1 + min(steps * 0.03, 0.15))
        
        # Final conversion rate with some randomness
        conversion_rate = base_conversion * conversion_multiplier * np.random.uniform(0.8, 1.2, n_samples)
        conversion_rate = np.clip(conversion_rate, 0.001, 0.25)  # Reasonable range
        
        # Click-through rate is related to conversion rate but higher
        ctr = conversion_rate * np.random.uniform(3, 5, n_samples)
        ctr = np.clip(ctr, 0.01, 0.5)  # Reasonable range
        
        # Open rate is related to CTR but higher
        open_rate = ctr * np.random.uniform(2, 3, n_samples)
        open_rate = np.clip(open_rate, 0.1, 0.7)  # Reasonable range
        
        # Revenue per conversion
        revenue_per_conversion = np.random.normal(50, 20, n_samples)
        revenue_per_conversion = np.clip(revenue_per_conversion, 10, 150)
        
        # Calculate revenue
        audience_size = np.random.randint(1000, 50000, n_samples)
        revenue = audience_size * conversion_rate * revenue_per_conversion
        
        # Cost per channel
        email_cost = np.random.uniform(0.01, 0.05, n_samples)
        sms_cost = np.random.uniform(0.05, 0.1, n_samples)
        push_cost = np.random.uniform(0.01, 0.03, n_samples)
        in_app_cost = np.random.uniform(0.02, 0.06, n_samples)
        social_cost = np.random.uniform(0.1, 0.5, n_samples)
        
        # Calculate campaign cost
        campaign_cost = (
            audience_size * (
                email_usage * email_cost +
                sms_usage * sms_cost +
                push_usage * push_cost +
                in_app_usage * in_app_cost +
                social_usage * social_cost / 10  # Social is usually per impression not per user
            )
        )
        
        # Calculate ROI
        roi = (revenue - campaign_cost) / campaign_cost
        roi = np.clip(roi, -0.9, 10)  # Reasonable range
        
        # Create the DataFrame
        data = {
            'campaign_id': campaign_ids,
            'campaign_type': campaign_types,
            'start_date': start_dates,
            'end_date': end_dates,
            'duration_days': durations,
            'target_segment': target_segments,
            'email_used': email_usage,
            'sms_used': sms_usage,
            'push_used': push_usage,
            'in_app_used': in_app_usage,
            'social_used': social_usage,
            'workflow_steps': workflow_steps,
            'send_time_optimization': send_time_optimization,
            'ab_testing': ab_testing,
            'ai_content': ai_content,
            'personalization_level': personalization_level,
            'audience_size': audience_size,
            'open_rate': open_rate,
            'click_through_rate': ctr,
            'conversion_rate': conversion_rate,
            'revenue': revenue,
            'cost': campaign_cost,
            'roi': roi
        }
        
        return pd.DataFrame(data)
    
    def train_model(self, target_metric: str = 'roi', test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train a machine learning model to predict campaign performance.
        
        Args:
            target_metric: Target metric to optimize (roi, conversion_rate, etc.)
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with model training results
        """
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("No data loaded. Call load_campaign_data() first.")
            
        if target_metric not in self.data.columns:
            raise ValueError(f"Target metric '{target_metric}' not found in data.")
            
        print(f"Training model to predict {target_metric}...")
        
        # Define features
        categorical_features = ['campaign_type', 'target_segment']
        binary_features = ['email_used', 'sms_used', 'push_used', 'in_app_used', 
                          'social_used', 'send_time_optimization', 'ab_testing', 'ai_content']
        numerical_features = ['duration_days', 'workflow_steps', 'personalization_level', 'audience_size']
        
        # Define preprocessor
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        numerical_transformer = StandardScaler()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
                ('num', numerical_transformer, numerical_features),
                ('bin', 'passthrough', binary_features)
            ])
        
        # Create and train pipeline
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Split data
        X = self.data[categorical_features + numerical_features + binary_features]
        y = self.data[target_metric]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"Model trained. R² on training data: {train_score:.4f}, R² on test data: {test_score:.4f}")
        
        # Store model and preprocessor
        self.model = model
        self.preprocessor = preprocessor
        
        # Calculate feature importance
        feature_names = (
            categorical_features + 
            numerical_features + 
            binary_features
        )
        
        # For RandomForest, we need to extract the actual estimator
        regressor = model.named_steps['regressor']
        importances = regressor.feature_importances_
        
        # Create a dictionary of feature importances
        self.feature_importance = dict(zip(feature_names, importances))
        
        # Sort by importance
        self.feature_importance = {k: v for k, v in sorted(
            self.feature_importance.items(), 
            key=lambda item: item[1], 
            reverse=True
        )}
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': self.feature_importance,
            'target_metric': target_metric
        }
    
    def optimize_workflow(self,
                         campaign_type: str,
                         target_segment: str,
                         audience_size: int,
                         constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize a campaign workflow for the given parameters.
        
        Args:
            campaign_type: Type of campaign
            target_segment: Target customer segment
            audience_size: Size of the target audience
            constraints: Dictionary of constraints (e.g., max budget, required channels)
            
        Returns:
            Dictionary with optimized workflow parameters
        """
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("No trained model available. Call train_model() first.")
            
        print(f"Optimizing workflow for {campaign_type} campaign targeting {target_segment}...")
        
        # Set default constraints if none provided
        if constraints is None:
            constraints = {}
            
        # Generate candidate workflows
        candidates = self._generate_candidate_workflows(
            campaign_type=campaign_type,
            target_segment=target_segment,
            audience_size=audience_size,
            constraints=constraints
        )
        
        # Predict performance for each candidate
        predictions = []
        for candidate in candidates:
            # Convert candidate to DataFrame for prediction
            candidate_df = pd.DataFrame([candidate])
            
            # Make prediction
            prediction = self.model.predict(candidate_df)[0]
            predictions.append(prediction)
            
        # Find the best candidate
        best_idx = np.argmax(predictions)
        best_candidate = candidates[best_idx]
        best_prediction = predictions[best_idx]
        
        # Construct result
        result = {
            'workflow': best_candidate,
            'predicted_performance': best_prediction,
            'num_candidates': len(candidates),
            'campaign_type': campaign_type,
            'target_segment': target_segment
        }
        
        # Store in best workflows
        key = f"{campaign_type}_{target_segment}"
        self.best_workflows[key] = result
        
        return result
    
    def _generate_candidate_workflows(self,
                                    campaign_type: str,
                                    target_segment: str,
                                    audience_size: int,
                                    constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate candidate workflow configurations for optimization.
        
        Args:
            campaign_type: Type of campaign
            target_segment: Target customer segment
            audience_size: Size of the target audience
            constraints: Dictionary of constraints
            
        Returns:
            List of candidate workflow dictionaries
        """
        candidates = []
        
        # Define ranges for each parameter
        duration_options = constraints.get('duration_options', [3, 5, 7, 14, 21, 30])
        workflow_step_options = constraints.get('workflow_step_options', [1, 2, 3, 4, 5, 6])
        personalization_options = constraints.get('personalization_options', [0, 1, 2, 3, 4, 5])
        
        # Channel constraints
        required_channels = constraints.get('required_channels', [])
        excluded_channels = constraints.get('excluded_channels', [])
        
        # Other feature constraints
        require_send_time_optimization = constraints.get('require_send_time_optimization', False)
        require_ab_testing = constraints.get('require_ab_testing', False)
        require_ai_content = constraints.get('require_ai_content', False)
        
        # Generate all possible channel combinations
        all_channels = ['email', 'sms', 'push', 'in_app', 'social']
        available_channels = [ch for ch in all_channels if ch not in excluded_channels]
        
        # Start with required channels
        channel_combinations = []
        
        # Helper to generate all combinations
        def generate_combinations(channels, start=0, current=[]):
            # Add current combination if it includes all required channels
            if all(req in [ch.replace('_used', '') for ch in current] for req in required_channels):
                channel_combinations.append(current.copy())
                
            # Generate remaining combinations
            for i in range(start, len(channels)):
                current.append(f"{channels[i]}_used")
                generate_combinations(channels, i + 1, current)
                current.pop()
        
        generate_combinations(available_channels)
        
        # If no valid combinations (e.g., due to constraints), use just the required channels
        if not channel_combinations and required_channels:
            channel_combination = [f"{ch}_used" for ch in required_channels]
            channel_combinations.append(channel_combination)
        
        # Generate candidates for each channel combination
        for channel_combo in channel_combinations:
            for duration in duration_options:
                for workflow_steps in workflow_step_options:
                    for personalization_level in personalization_options:
                        # Create candidate
                        candidate = {
                            'campaign_type': campaign_type,
                            'target_segment': target_segment,
                            'duration_days': duration,
                            'workflow_steps': workflow_steps,
                            'personalization_level': personalization_level,
                            'audience_size': audience_size,
                            'email_used': 1 if 'email_used' in channel_combo else 0,
                            'sms_used': 1 if 'sms_used' in channel_combo else 0,
                            'push_used': 1 if 'push_used' in channel_combo else 0,
                            'in_app_used': 1 if 'in_app_used' in channel_combo else 0,
                            'social_used': 1 if 'social_used' in channel_combo else 0,
                            'send_time_optimization': 1 if require_send_time_optimization else np.random.choice([0, 1]),
                            'ab_testing': 1 if require_ab_testing else np.random.choice([0, 1]),
                            'ai_content': 1 if require_ai_content else np.random.choice([0, 1])
                        }
                        
                        candidates.append(candidate)
        
        print(f"Generated {len(candidates)} candidate workflows")
        return candidates
    
    def visualize_feature_importance(self, top_n: int = 10, save_path: Optional[str] = None) -> None:
        """
        Visualize feature importance from the trained model.
        
        Args:
            top_n: Number of top features to display
            save_path: Path to save the visualization
        """
        if not self.feature_importance:
            raise ValueError("No feature importance available. Train a model first.")
            
        # Get top N features
        top_features = list(self.feature_importance.keys())[:top_n]
        top_importances = [self.feature_importance[f] for f in top_features]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.barh(top_features, top_importances)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Top Features for Campaign Performance')
        plt.gca().invert_yaxis()  # Display highest importance at the top
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
            
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str = 'campaign_optimizer_model.joblib') -> None:
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("No trained model available to save.")
            
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'best_workflows': self.best_workflows
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'campaign_optimizer_model.joblib') -> None:
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            raise ValueError(f"Model file not found: {filepath}")
            
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_importance = model_data.get('feature_importance', {})
        self.best_workflows = model_data.get('best_workflows', {})
        
        print(f"Model loaded from {filepath}")
    
    def export_workflow_recommendations(self, 
                                      filepath: str = 'workflow_recommendations.json',
                                      netcore_format: bool = False) -> None:
        """
        Export workflow recommendations to a file or Netcore format.
        
        Args:
            filepath: Path to save the recommendations
            netcore_format: Whether to format for Netcore API
        """
        if not self.best_workflows:
            raise ValueError("No workflow recommendations available.")
            
        if netcore_format:
            # Format for Netcore API
            netcore_recommendations = []
            
            for key, workflow in self.best_workflows.items():
                campaign_type, target_segment = key.split('_', 1)
                
                channels = []
                if workflow['workflow'].get('email_used', 0) == 1:
                    channels.append('email')
                if workflow['workflow'].get('sms_used', 0) == 1:
                    channels.append('sms')
                if workflow['workflow'].get('push_used', 0) == 1:
                    channels.append('push')
                if workflow['workflow'].get('in_app_used', 0) == 1:
                    channels.append('in-app')
                if workflow['workflow'].get('social_used', 0) == 1:
                    channels.append('social')
                    
                netcore_rec = {
                    'campaign_type': campaign_type,
                    'segment': target_segment,
                    'duration': workflow['workflow'].get('duration_days', 7),
                    'recommended_channels': channels,
                    'workflow_steps': workflow['workflow'].get('workflow_steps', 3),
                    'features': {
                        'send_time_optimization': workflow['workflow'].get('send_time_optimization', 0) == 1,
                        'ab_testing': workflow['workflow'].get('ab_testing', 0) == 1,
                        'ai_content': workflow['workflow'].get('ai_content', 0) == 1,
                        'personalization_level': workflow['workflow'].get('personalization_level', 3)
                    },
                    'predicted_performance': {
                        'metric': workflow.get('target_metric', 'roi'),
                        'value': workflow.get('predicted_performance', 0)
                    }
                }
                
                netcore_recommendations.append(netcore_rec)
                
            with open(filepath, 'w') as f:
                json.dump({'recommendations': netcore_recommendations}, f, indent=4)
        else:
            # Export as-is
            with open(filepath, 'w') as f:
                json.dump(self.best_workflows, f, indent=4)
                
        print(f"Workflow recommendations exported to {filepath}")

# Example usage
if __name__ == "__main__":
    # Initialize the optimizer
    optimizer = CampaignWorkflowOptimizer()
    
    # Load data
    optimizer.load_campaign_data(sample_size=500)
    
    # Train model
    training_results = optimizer.train_model(target_metric='roi')
    
    # Print feature importance
    print("\nFeature Importance:")
    for feature, importance in list(training_results['feature_importance'].items())[:5]:
        print(f"{feature}: {importance:.4f}")
        
    # Visualize feature importance
    optimizer.visualize_feature_importance(top_n=8)
    
    # Optimize workflows for different scenarios
    promotional_workflow = optimizer.optimize_workflow(
        campaign_type='promotional',
        target_segment='high_value',
        audience_size=10000,
        constraints={
            'required_channels': ['email'],
            'require_ai_content': True
        }
    )
    
    reengagement_workflow = optimizer.optimize_workflow(
        campaign_type='re_engagement',
        target_segment='inactive',
        audience_size=5000,
        constraints={
            'workflow_step_options': [3, 4, 5, 6],  # More complex workflows
            'require_send_time_optimization': True
        }
    )
    
    # Print recommendations
    print("\nOptimized Promotional Campaign Workflow:")
    print(f"Channels: " + ", ".join([
        ch.replace('_used', '') for ch, val in promotional_workflow['workflow'].items()
        if ch.endswith('_used') and val == 1
    ]))
    print(f"Workflow Steps: {promotional_workflow['workflow']['workflow_steps']}")
    print(f"Personalization Level: {promotional_workflow['workflow']['personalization_level']}")
    print(f"AI Content: {'Yes' if promotional_workflow['workflow']['ai_content'] == 1 else 'No'}")
    print(f"Predicted ROI: {promotional_workflow['predicted_performance']:.2f}")
    
    # Export recommendations
    optimizer.export_workflow_recommendations(netcore_format=True) 