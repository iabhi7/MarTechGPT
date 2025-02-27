import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import joblib
import json
import os
from datetime import datetime, timedelta

class CustomerChurnPredictor:
    def __init__(self, netcore_api_key: Optional[str] = None):
        """
        Initialize the Customer Churn Predictor.
        
        Args:
            netcore_api_key: API key for Netcore integration
        """
        self.netcore_api_key = netcore_api_key
        self.model = None
        self.preprocessor = None
        self.feature_importance = {}
        self.risk_score_thresholds = {
            'high': 0.7,
            'medium': 0.4,
            'low': 0.2
        }
        
        print("Customer Churn Predictor initialized")
        
    def load_customer_data(self, 
                         filepath: Optional[str] = None, 
                         data: Optional[pd.DataFrame] = None,
                         use_netcore_api: bool = False,
                         sample_size: int = 1000) -> pd.DataFrame:
        """
        Load customer behavioral data.
        
        Args:
            filepath: Path to CSV file with customer data
            data: Pandas DataFrame with customer data
            use_netcore_api: Whether to fetch data from Netcore API
            sample_size: Size of sample data to create if no data provided
            
        Returns:
            DataFrame containing the loaded customer data
        """
        if data is not None:
            self.data = data
            print(f"Data loaded with {len(self.data)} customer records")
            return self.data
            
        if filepath and os.path.exists(filepath):
            self.data = pd.read_csv(filepath)
            print(f"Data loaded from {filepath} with {len(self.data)} customer records")
            return self.data
            
        if use_netcore_api and self.netcore_api_key:
            # This would be replaced with actual API call to Netcore
            print("Fetching customer data from Netcore API (mock)")
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
        np.random.seed(42)
        
        # Customer IDs
        customer_ids = [f'CUST{i:05d}' for i in range(1, n_samples + 1)]
        
        # Last activity date
        today = datetime.now()
        
        # We'll create a mix of active and inactive customers
        # About 20% will be churned (no activity in 90 days)
        active_mask = np.random.choice([True, False], n_samples, p=[0.8, 0.2])
        
        last_activity_dates = []
        for is_active in active_mask:
            if is_active:
                # Active customers have activity within the last 90 days
                days_ago = np.random.randint(0, 90)
            else:
                # Inactive customers have no activity for 90+ days
                days_ago = np.random.randint(90, 365)
                
            last_date = today - timedelta(days=days_ago)
            last_activity_dates.append(last_date.strftime('%Y-%m-%d'))
        
        # Days since account creation (tenure)
        tenure_days = np.random.randint(1, 1000, n_samples)
        
        # Purchase frequency (purchases per month)
        purchase_frequency = np.random.exponential(scale=1.5, size=n_samples)
        purchase_frequency = np.clip(purchase_frequency, 0, 10)
        
        # Average order value
        avg_order_value = np.random.normal(loc=50, scale=30, size=n_samples)
        avg_order_value = np.clip(avg_order_value, 5, 200)
        
        # Email engagement metrics
        email_open_rate = np.random.beta(2, 5, n_samples)  # Beta distribution gives values between 0-1
        email_click_rate = email_open_rate * np.random.beta(1.5, 6, n_samples)  # Click rate is lower than open rate
        
        # Product categories purchased
        product_categories = np.random.randint(1, 5, n_samples)
        
        # Cart abandonment rate
        cart_abandonment_rate = np.random.beta(3, 2, n_samples)
        
        # Customer support interactions
        support_tickets = np.random.poisson(lam=0.5, size=n_samples)
        
        # Discount usage
        discount_usage = np.random.poisson(lam=2, size=n_samples)
        
        # App usage
        has_mobile_app = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
        app_sessions_per_month = np.zeros(n_samples)
        app_sessions_per_month[has_mobile_app == 1] = np.random.poisson(lam=5, size=sum(has_mobile_app))
        
        # Customer segments
        segments = np.random.choice(
            ['high_value', 'regular', 'occasional', 'new'],
            n_samples,
            p=[0.2, 0.5, 0.2, 0.1]
        )
        
        # Geography
        regions = np.random.choice(
            ['north', 'south', 'east', 'west', 'central'],
            n_samples,
            p=[0.2, 0.2, 0.2, 0.2, 0.2]
        )
        
        # Payment method
        payment_methods = np.random.choice(
            ['credit_card', 'debit_card', 'netbanking', 'upi', 'wallet'],
            n_samples,
            p=[0.3, 0.25, 0.2, 0.15, 0.1]
        )
        
        # Create churn label (target variable)
        # Churn is influenced by:
        # 1. Days since last activity (primary factor)
        # 2. Email engagement (lower engagement = higher churn)
        # 3. App usage (no app = higher churn)
        # 4. Support tickets (more tickets = higher churn)
        # 5. Purchase frequency (lower frequency = higher churn)
        
        days_since_activity = [(today - datetime.strptime(date, '%Y-%m-%d')).days for date in last_activity_dates]
        
        # Calculate churn probability
        churn_prob = np.zeros(n_samples)
        
        # Days since activity strongly influences churn
        for i, days in enumerate(days_since_activity):
            if days < 30:
                churn_prob[i] += 0.1
            elif days < 60:
                churn_prob[i] += 0.3
            elif days < 90:
                churn_prob[i] += 0.5
            else:
                churn_prob[i] += 0.9
                
        # Low email engagement increases churn
        churn_prob[email_open_rate < 0.1] += 0.2
        
        # No app usage increases churn
        churn_prob[has_mobile_app == 0] += 0.15
        
        # Many support tickets increases churn
        churn_prob[support_tickets > 2] += 0.2
        
        # Low purchase frequency increases churn
        churn_prob[purchase_frequency < 0.5] += 0.15
        
        # Add some randomness
        churn_prob += np.random.uniform(-0.1, 0.1, n_samples)
        
        # Cap between 0 and 1
        churn_prob = np.clip(churn_prob, 0, 1)
        
        # Convert to binary label
        churn = (churn_prob > 0.5).astype(int)
        
        # Create DataFrame
        data = {
            'customer_id': customer_ids,
            'last_activity_date': last_activity_dates,
            'days_since_activity': days_since_activity,
            'tenure_days': tenure_days,
            'purchase_frequency': purchase_frequency,
            'avg_order_value': avg_order_value,
            'email_open_rate': email_open_rate,
            'email_click_rate': email_click_rate,
            'product_categories': product_categories,
            'cart_abandonment_rate': cart_abandonment_rate,
            'support_tickets': support_tickets,
            'discount_usage': discount_usage,
            'has_mobile_app': has_mobile_app,
            'app_sessions_per_month': app_sessions_per_month,
            'segment': segments,
            'region': regions,
            'payment_method': payment_methods,
            'churn_probability': churn_prob,
            'churn': churn
        }
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, target_col: str = 'churn') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the customer data for modeling.
        
        Args:
            target_col: Name of the target column
            
        Returns:
            Tuple of preprocessed features and target
        """
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("No data loaded. Call load_customer_data() first.")
            
        # Make a copy to avoid modifying original
        df = self.data.copy()
        
        # Remove ID columns and the target from features
        features = df.drop(columns=['customer_id', target_col, 'last_activity_date', 'churn_probability'], errors='ignore')
        
        # Get target
        if target_col in df.columns:
            target = df[target_col]
        else:
            raise ValueError(f"Target column '{target_col}' not found in data")
            
        # Define preprocessing for numerical and categorical features
        numeric_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = features.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create preprocessor
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Store feature names for later
        self.feature_names = numeric_features + categorical_features
        
        # Return preprocessed data and target
        return features, target
        
    def train_model(self, 
                   model_type: str = 'random_forest',
                   test_size: float = 0.2,
                   random_state: int = 42) -> Dict[str, Any]:
        """
        Train a churn prediction model.
        
        Args:
            model_type: Type of model to train ('random_forest' or 'gradient_boosting')
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with model training results
        """
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("No data loaded. Call load_customer_data() first.")
            
        # Preprocess data
        features, target = self.preprocess_data()
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state, stratify=target
        )
        
        # Create and fit preprocessor on training data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=random_state,
                class_weight='balanced'
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Fit model
        self.model.fit(X_train_processed, y_train)
        
        # Evaluate on test set
        y_pred_proba = self.model.predict_proba(X_test_processed)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            # Get feature names from preprocessor
            # This is a bit complex due to one-hot encoding
            feature_names = []
            
            # For numeric features, use the original names
            numeric_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
            feature_names.extend(numeric_features)
            
            # For categorical features, get the one-hot encoded names
            categorical_features = features.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if categorical_features:
                # Get the OneHotEncoder
                ohe = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
                
                # Get the categories
                cat_features = []
                for i, cat_feat in enumerate(categorical_features):
                    for cat in ohe.categories_[i]:
                        cat_features.append(f"{cat_feat}_{cat}")
                
                feature_names.extend(cat_features)
            
            # Map importance to feature names
            importances = self.model.feature_importances_
            
            # If lengths don't match (due to how sklearn's ColumnTransformer works),
            # just use indices
            if len(importances) != len(feature_names):
                feature_names = [f"feature_{i}" for i in range(len(importances))]
                
            self.feature_importance = {
                name: importance for name, importance in zip(feature_names, importances)
            }
            
            # Sort by importance
            self.feature_importance = {
                k: v for k, v in sorted(
                    self.feature_importance.items(), 
                    key=lambda item: item[1], 
                    reverse=True
                )
            }
            
        # Store model training results
        results = {
            'model_type': model_type,
            'auc_score': auc_score,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'feature_importance': self.feature_importance
        }
        
        print(f"Model trained successfully. AUC: {auc_score:.4f}")
        return results
    
    def predict_churn_risk(self, 
                         customer_data: Union[pd.DataFrame, Dict[str, Any]],
                         threshold: float = 0.5) -> Dict[str, Any]:
        """
        Predict churn risk for a customer or set of customers.
        
        Args:
            customer_data: DataFrame or dictionary with customer data
            threshold: Threshold for classifying as churn
            
        Returns:
            Dictionary with prediction results
        """
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("No trained model available. Call train_model() first.")
            
        # Convert dictionary to DataFrame if needed
        if isinstance(customer_data, dict):
            customer_data = pd.DataFrame([customer_data])
            
        # Store customer IDs if available
        customer_ids = None
        if 'customer_id' in customer_data.columns:
            customer_ids = customer_data['customer_id'].tolist()
        else:
            customer_ids = [f"customer_{i}" for i in range(len(customer_data))]
            
        # Prepare data for prediction
        features = customer_data.drop(columns=['customer_id', 'churn', 'last_activity_date', 'churn_probability'], errors='ignore')
        
        # Transform features
        features_processed = self.preprocessor.transform(features)
        
        # Make predictions
        churn_probabilities = self.model.predict_proba(features_processed)[:, 1]
        churn_predictions = (churn_probabilities > threshold).astype(int)
        
        # Categorize risk levels
        risk_levels = []
        for prob in churn_probabilities:
            if prob >= self.risk_score_thresholds['high']:
                risk_levels.append('high')
            elif prob >= self.risk_score_thresholds['medium']:
                risk_levels.append('medium')
            elif prob >= self.risk_score_thresholds['low']:
                risk_levels.append('low')
            else:
                risk_levels.append('very_low')
                
        # Compile results
        results = {
            'customer_ids': customer_ids,
            'churn_probabilities': churn_probabilities.tolist(),
            'churn_predictions': churn_predictions.tolist(),
            'risk_levels': risk_levels
        }
        
        # Add individual customer predictions
        customer_predictions = []
        for i in range(len(customer_ids)):
            customer_predictions.append({
                'customer_id': customer_ids[i],
                'churn_probability': churn_probabilities[i],
                'churn_prediction': bool(churn_predictions[i]),
                'risk_level': risk_levels[i]
            })
            
        results['customer_predictions'] = customer_predictions
        
        return results
    
    def generate_churn_prevention_recommendations(self, 
                                                customer_id: str,
                                                churn_probability: float) -> List[str]:
        """
        Generate recommendations for preventing churn based on risk level.
        
        Args:
            customer_id: ID of the customer
            churn_probability: Predicted churn probability
            
        Returns:
            List of recommendation strings
        """
        # Get customer data
        if not hasattr(self, 'data') or self.data is None or customer_id not in self.data['customer_id'].values:
            return ["Customer data not available for generating detailed recommendations."]
            
        customer_data = self.data[self.data['customer_id'] == customer_id].iloc[0]
        
        # Define risk level
        if churn_probability >= self.risk_score_thresholds['high']:
            risk_level = 'high'
        elif churn_probability >= self.risk_score_thresholds['medium']:
            risk_level = 'medium'
        elif churn_probability >= self.risk_score_thresholds['low']:
            risk_level = 'low'
        else:
            risk_level = 'very_low'
            
        # Generate recommendations based on risk level and customer characteristics
        recommendations = []
        
        # High-risk recommendations
        if risk_level == 'high':
            recommendations.append("Immediate outreach: Send a personalized message from account manager")
            
            # Check if inactive for a long time
            if customer_data['days_since_activity'] > 60:
                recommendations.append("Re-engagement campaign: Special offer for returning customers")
                
            # Check app usage
            if customer_data['has_mobile_app'] == 0:
                recommendations.append("Promote mobile app: Highlight convenience and exclusive app-only deals")
            elif customer_data['app_sessions_per_month'] < 2:
                recommendations.append("App engagement: Push notification with personalized content")
                
            # Check email engagement
            if customer_data['email_open_rate'] < 0.1:
                recommendations.append("Email re-permission campaign: Update communication preferences")
                
            # Check support issues
            if customer_data['support_tickets'] > 2:
                recommendations.append("Customer success check-in: Address any ongoing service issues")
                
            recommendations.append("Loyalty incentive: Offer special discount or loyalty program upgrade")
            
        # Medium-risk recommendations
        elif risk_level == 'medium':
            recommendations.append("Proactive outreach: Send satisfaction survey with incentive for completion")
            
            # Check email engagement
            if customer_data['email_open_rate'] < 0.2:
                recommendations.append("Email relevance: Segment into more targeted email campaigns")
                
            # Check app usage
            if customer_data['has_mobile_app'] == 1 and customer_data['app_sessions_per_month'] < 5:
                recommendations.append("App re-engagement: Highlight new features or content")
                
            # Check purchase patterns
            if customer_data['purchase_frequency'] < 1:
                recommendations.append("Reactivation offer: Personalized product recommendations")
                
            recommendations.append("Education campaign: Share tips and best practices for product usage")
            
        # Low-risk recommendations
        elif risk_level == 'low':
            recommendations.append("Engagement campaign: Cross-sell or upsell related products")
            
            # Reward loyalty if long-tenured
            if customer_data['tenure_days'] > 365:
                recommendations.append("Loyalty recognition: Thank you message with small gift or perk")
                
            # Increase engagement
            recommendations.append("Community engagement: Invite to webinar or product feedback group")
            
        # Very low-risk recommendations
        else:
            recommendations.append("Maintain relationship: Regular check-ins and product updates")
            recommendations.append("Referral request: Invite to refer colleagues or friends")
            recommendations.append("Advocacy opportunity: Invite to case study or testimonial")
            
        return recommendations
    
    def visualize_churn_factors(self, top_n: int = 10, save_path: Optional[str] = None) -> None:
        """
        Visualize the most important factors influencing churn.
        
        Args:
            top_n: Number of top features to display
            save_path: Path to save the visualization
        """
        if not self.feature_importance:
            raise ValueError("No feature importance available. Train a model first.")
            
        # Get top N features
        top_features = list(self.feature_importance.items())[:top_n]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Create bar chart
        features = [feat[0] for feat in top_features]
        importances = [feat[1] for feat in top_features]
        
        bars = plt.barh(features, importances, color='steelblue')
        
        # Add labels
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Features Influencing Customer Churn', fontsize=15)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                     f'{width:.3f}', ha='left', va='center', fontsize=10)
            
        plt.gca().invert_yaxis()  # Display highest importance at the top
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
            
        plt.tight_layout()
        plt.show()
        
    def export_high_risk_customers(self, 
                                 filepath: str = 'high_risk_customers.csv',
                                 risk_level: str = 'high',
                                 include_recommendations: bool = True) -> pd.DataFrame:
        """
        Export a list of high-risk customers for targeted interventions.
        
        Args:
            filepath: Path to save the CSV
            risk_level: Minimum risk level to include ('high', 'medium', 'low')
            include_recommendations: Whether to include churn prevention recommendations
            
        Returns:
            DataFrame with high-risk customers
        """
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("No trained model available. Call train_model() first.")
            
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("No data available. Load data first.")
            
        # Define threshold based on risk level
        if risk_level == 'high':
            threshold = self.risk_score_thresholds['high']
        elif risk_level == 'medium':
            threshold = self.risk_score_thresholds['medium']
        elif risk_level == 'low':
            threshold = self.risk_score_thresholds['low']
        else:
            raise ValueError(f"Invalid risk level: {risk_level}")
            
        # Get features for prediction
        features = self.data.drop(columns=['customer_id', 'churn', 'last_activity_date', 'churn_probability'], errors='ignore')
        
        # Transform features
        features_processed = self.preprocessor.transform(features)
        
        # Make predictions
        churn_probabilities = self.model.predict_proba(features_processed)[:, 1]
        
        # Add predictions to data
        results = self.data.copy()
        results['predicted_churn_probability'] = churn_probabilities
        
        # Filter by risk threshold
        high_risk = results[results['predicted_churn_probability'] >= threshold].copy()
        
        # Sort by churn probability (highest first)
        high_risk = high_risk.sort_values('predicted_churn_probability', ascending=False)
        
        # Add risk level
        high_risk['risk_level'] = high_risk['predicted_churn_probability'].apply(
            lambda p: 'high' if p >= self.risk_score_thresholds['high'] else
                     'medium' if p >= self.risk_score_thresholds['medium'] else
                     'low' if p >= self.risk_score_thresholds['low'] else 'very_low'
        )
        
        # Add recommendations if requested
        if include_recommendations:
            recommendations = []
            for _, row in high_risk.iterrows():
                customer_recs = self.generate_churn_prevention_recommendations(
                    row['customer_id'], row['predicted_churn_probability']
                )
                recommendations.append("; ".join(customer_recs))
                
            high_risk['recommendations'] = recommendations
            
        # Save to CSV
        high_risk.to_csv(filepath, index=False)
        print(f"Exported {len(high_risk)} high-risk customers to {filepath}")
        
        return high_risk

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = CustomerChurnPredictor()
    
    # Load data
    data = predictor.load_customer_data(sample_size=1000)
    
    # Train model
    results = predictor.train_model(model_type='random_forest')
    
    # Display model performance
    print(f"AUC Score: {results['auc_score']:.4f}")
    print("\nTop Churn Factors:")
    for feature, importance in list(results['feature_importance'].items())[:5]:
        print(f"- {feature}: {importance:.4f}")
        
    # Visualize churn factors
    predictor.visualize_churn_factors()
    
    # Make predictions for a sample customer
    sample_customer = data.iloc[0].to_dict()
    prediction = predictor.predict_churn_risk(sample_customer)
    
    print(f"\nSample Customer {sample_customer['customer_id']}:")
    print(f"Churn Probability: {prediction['churn_probabilities'][0]:.2f}")
    print(f"Risk Level: {prediction['risk_levels'][0]}")
    
    # Generate recommendations
    recommendations = predictor.generate_churn_prevention_recommendations(
        sample_customer['customer_id'], prediction['churn_probabilities'][0]
    )
    
    print("\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
        
    # Export high-risk customers
    high_risk = predictor.export_high_risk_customers(risk_level='medium')
    print(f"\nIdentified {len(high_risk)} customers at medium or higher risk of churn") 