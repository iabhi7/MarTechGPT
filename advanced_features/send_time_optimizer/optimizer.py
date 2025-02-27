import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os
from datetime import datetime, timedelta
import pytz
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('send_time_optimizer')

class SendTimeOptimizer:
    """
    AI-powered send time optimization engine.
    Determines the optimal time to send marketing communications
    based on historical engagement data.
    """
    
    def __init__(self, 
                historical_data_path: Optional[str] = None,
                api_key: Optional[str] = None,
                use_gpu: bool = False):
        """Initialize the send time optimizer"""
        self.historical_data = self._load_historical_data(historical_data_path)
        self.api_key = api_key
        self.use_gpu = use_gpu
        self.model = None
        self.preprocessor = None
        self.feature_importance = {}
        self.model_performance = {}
        self.time_segments = {
            'early_morning': (5, 8),
            'morning': (8, 11),
            'lunch': (11, 14),
            'afternoon': (14, 17),
            'evening': (17, 20),
            'night': (20, 23),
            'late_night': (23, 5)
        }
        self.day_mapping = {
            0: 'Monday',
            1: 'Tuesday',
            2: 'Wednesday',
            3: 'Thursday',
            4: 'Friday',
            5: 'Saturday',
            6: 'Sunday'
        }
        logger.info("Send Time Optimizer initialized")
        
    def _load_historical_data(self, data_path: Optional[str] = None):
        """Load historical engagement data"""
        if data_path and os.path.exists(data_path):
            return pd.read_csv(data_path)
        else:
            print("No historical data found. Using default optimization strategy.")
            return None
    
    def _validate_and_preprocess_data(self):
        """Validate and preprocess the loaded data."""
        required_columns = ['customer_id', 'send_timestamp', 'open_timestamp', 'click_timestamp']
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in self.historical_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Convert timestamps to datetime if they're not already
        for col in ['send_timestamp', 'open_timestamp', 'click_timestamp']:
            if self.historical_data[col].dtype != 'datetime64[ns]':
                self.historical_data[col] = pd.to_datetime(self.historical_data[col])
                
        # Calculate engagement metrics
        self.historical_data['open_delay'] = (self.historical_data['open_timestamp'] - self.historical_data['send_timestamp']).dt.total_seconds() / 60
        self.historical_data['click_delay'] = (self.historical_data['click_timestamp'] - self.historical_data['send_timestamp']).dt.total_seconds() / 60
        
        # Filter out invalid delays (negative or extremely long)
        self.historical_data = self.historical_data[self.historical_data['open_delay'] >= 0]
        self.historical_data = self.historical_data[self.historical_data['open_delay'] <= 24*60]  # Max 24 hours
        
        # Extract time features
        self.historical_data['send_hour'] = self.historical_data['send_timestamp'].dt.hour
        self.historical_data['send_day'] = self.historical_data['send_timestamp'].dt.dayofweek
        self.historical_data['send_weekend'] = self.historical_data['send_day'].apply(lambda x: 1 if x >= 5 else 0)
        
        logger.info(f"Data validated and preprocessed: {len(self.historical_data)} valid records")
    
    def _create_sample_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Create sample data for demonstration purposes."""
        customer_ids = [f"CUST{i:05d}" for i in range(1, 501)]
        
        # Generate random timestamps within the last 90 days
        now = datetime.now()
        start_date = now - timedelta(days=90)
        
        data = []
        for _ in range(n_samples):
            customer_id = np.random.choice(customer_ids)
            send_timestamp = start_date + timedelta(
                days=np.random.randint(0, 90),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )
            
            # Simulate engagement patterns (people are more likely to engage during certain hours)
            send_hour = send_timestamp.hour
            
            # Model different engagement patterns for different times of day
            if 7 <= send_hour <= 10:  # Morning
                open_delay_mean = 30  # mins
            elif 11 <= send_hour <= 14:  # Lunch
                open_delay_mean = 60
            elif 17 <= send_hour <= 21:  # Evening
                open_delay_mean = 40
            else:  # Late night/early morning
                open_delay_mean = 180
                
            # Add some randomness to engagement delays
            open_delay = max(0, np.random.normal(open_delay_mean, open_delay_mean/2))
            click_delay = open_delay + max(0, np.random.normal(15, 10))
            
            # Create timestamps
            open_timestamp = send_timestamp + timedelta(minutes=open_delay)
            click_timestamp = send_timestamp + timedelta(minutes=click_delay)
            
            data.append({
                'customer_id': customer_id,
                'send_timestamp': send_timestamp,
                'open_timestamp': open_timestamp,
                'click_timestamp': click_timestamp,
                'campaign_type': np.random.choice(['promotional', 'newsletter', 'transactional']),
                'device': np.random.choice(['mobile', 'desktop', 'tablet'], p=[0.6, 0.3, 0.1])
            })
            
        return pd.DataFrame(data)
    
    def prepare_features(self, include_campaign_type: bool = True, 
                        include_device: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for training the model.
        
        Args:
            include_campaign_type: Whether to include campaign type as a feature
            include_device: Whether to include device as a feature
            
        Returns:
            Tuple of (X, y) where X is the feature matrix and y is the target
        """
        if not hasattr(self, 'historical_data'):
            raise ValueError("No data loaded. Call load_historical_data first.")
            
        # Group by customer to get their engagement patterns
        customer_data = []
        
        for customer_id, group in self.historical_data.groupby('customer_id'):
            # Calculate average engagement metrics per hour and day of week
            for day in range(7):  # 0-6 for Monday-Sunday
                for hour in range(24):
                    day_hour_data = group[(group['send_day'] == day) & (group['send_hour'] == hour)]
                    
                    if len(day_hour_data) > 0:
                        avg_open_delay = day_hour_data['open_delay'].mean()
                        engagement_count = len(day_hour_data)
                    else:
                        avg_open_delay = None
                        engagement_count = 0
                        
                    # Create a record for each customer-day-hour combination
                    record = {
                        'customer_id': customer_id,
                        'day_of_week': day,
                        'hour': hour,
                        'avg_open_delay': avg_open_delay,
                        'engagement_count': engagement_count
                    }
                    
                    # Add campaign type and device if requested
                    if include_campaign_type and 'campaign_type' in group.columns:
                        # Get most common campaign type for this customer
                        record['campaign_type'] = group['campaign_type'].mode()[0]
                        
                    if include_device and 'device' in group.columns:
                        # Get most common device for this customer
                        record['device'] = group['device'].mode()[0]
                        
                    customer_data.append(record)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(customer_data)
        
        # Handle missing values - replace NaN with high values to discourage selecting those times
        features_df['avg_open_delay'] = features_df['avg_open_delay'].fillna(24*60)  # 24 hours
        
        # Prepare features (X) and target (y)
        # Target is the average open delay (lower is better)
        y = features_df['avg_open_delay']
        
        # Features are everything except the target and identifiers
        X = features_df.drop(['avg_open_delay'], axis=1)
        
        logger.info(f"Prepared {len(X)} feature records across {features_df['customer_id'].nunique()} customers")
        return X, y
    
    def train_model(self, algorithm: str = 'random_forest', test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the send time optimization model.
        
        Args:
            algorithm: Algorithm to use ('random_forest' or 'gradient_boosting')
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with model performance metrics
        """
        X, y = self.prepare_features()
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Get customer IDs for later and drop from features
        train_customers = X_train['customer_id'].copy()
        test_customers = X_test['customer_id'].copy()
        X_train = X_train.drop('customer_id', axis=1)
        X_test = X_test.drop('customer_id', axis=1)
        
        # Define preprocessing steps
        categorical_features = ['day_of_week']
        if 'campaign_type' in X_train.columns:
            categorical_features.append('campaign_type')
        if 'device' in X_train.columns:
            categorical_features.append('device')
            
        numeric_features = [col for col in X_train.columns if col not in categorical_features]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Choose algorithm
        if algorithm == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif algorithm == 'gradient_boosting':
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train model
        logger.info(f"Training {algorithm} model on {len(X_train)} records")
        pipeline.fit(X_train, y_train)
        
        # Evaluate model
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        logger.info(f"Model trained. Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
        
        # Save model and preprocessor
        self.model = pipeline
        self.preprocessor = preprocessor
        
        # Get feature importance if applicable
        if algorithm == 'random_forest' or algorithm == 'gradient_boosting':
            # Get feature names after preprocessing
            feature_names = (
                numeric_features +
                pipeline.named_steps['preprocessor']
                .named_transformers_['cat']
                .get_feature_names_out(categorical_features).tolist()
            )
            
            # Get feature importances
            importances = pipeline.named_steps['model'].feature_importances_
            self.feature_importance = dict(zip(feature_names, importances))
            
        # Save performance metrics
        self.model_performance = {
            'algorithm': algorithm,
            'train_rmse_minutes': train_rmse,
            'test_rmse_minutes': test_rmse,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'feature_importance': self.feature_importance
        }
        
        return self.model_performance
    
    def predict_optimal_times(self, customer_ids: Union[str, List[str]], 
                            campaign_type: Optional[str] = None,
                            device: Optional[str] = None,
                            n_recommendations: int = 3) -> pd.DataFrame:
        """
        Predict optimal send times for specified customers.
        
        Args:
            customer_ids: Single customer ID or list of customer IDs
            campaign_type: Type of campaign (optional)
            device: Device type (optional)
            n_recommendations: Number of time recommendations to return
            
        Returns:
            DataFrame with optimal send times for each customer
        """
        if not self.model:
            raise ValueError("Model not trained. Call train_model first.")
            
        # Ensure customer_ids is a list
        if isinstance(customer_ids, str):
            customer_ids = [customer_ids]
            
        # Build prediction matrix - all possible day/hour combinations for each customer
        prediction_data = []
        for customer_id in customer_ids:
            for day in range(7):
                for hour in range(24):
                    record = {
                        'customer_id': customer_id,
                        'day_of_week': day,
                        'hour': hour,
                        'engagement_count': 1  # Placeholder
                    }
                    
                    if campaign_type:
                        record['campaign_type'] = campaign_type
                    elif 'campaign_type' in self.model[0].transformers_[1][2]:
                        # Use most common campaign type from training data
                        customer_data = self.historical_data[self.historical_data['customer_id'] == customer_id]
                        if len(customer_data) > 0 and 'campaign_type' in customer_data.columns:
                            record['campaign_type'] = customer_data['campaign_type'].mode()[0]
                        else:
                            record['campaign_type'] = self.historical_data['campaign_type'].mode()[0]
                            
                    if device:
                        record['device'] = device
                    elif 'device' in self.model[0].transformers_[1][2]:
                        # Use most common device from training data
                        customer_data = self.historical_data[self.historical_data['customer_id'] == customer_id]
                        if len(customer_data) > 0 and 'device' in customer_data.columns:
                            record['device'] = customer_data['device'].mode()[0]
                        else:
                            record['device'] = self.historical_data['device'].mode()[0]
                            
                    prediction_data.append(record)
                    
        # Convert to DataFrame
        pred_df = pd.DataFrame(prediction_data)
        
        # Save customer IDs and drop from features
        pred_customer_ids = pred_df['customer_id'].copy()
        X_pred = pred_df.drop('customer_id', axis=1)
        
        # Make predictions
        logger.info(f"Predicting optimal send times for {len(customer_ids)} customers")
        predicted_delays = self.model.predict(X_pred)
        
        # Add predictions to the DataFrame
        pred_df['predicted_delay'] = predicted_delays
        
        # Get top N recommendations for each customer
        results = []
        
        for customer_id in customer_ids:
            customer_pred = pred_df[pred_df['customer_id'] == customer_id]
            
            # Sort by predicted delay (ascending)
            customer_pred = customer_pred.sort_values('predicted_delay')
            
            # Get top N recommendations
            top_recommendations = customer_pred.head(n_recommendations)
            
            for _, row in top_recommendations.iterrows():
                day_name = self.day_mapping[row['day_of_week']]
                hour = int(row['hour'])
                
                # Get time segment name
                time_segment = None
                for segment, (start, end) in self.time_segments.items():
                    if start <= hour < end or (segment == 'late_night' and (hour >= start or hour < end)):
                        time_segment = segment
                        break
                        
                results.append({
                    'customer_id': customer_id,
                    'day_of_week': row['day_of_week'],
                    'day_name': day_name,
                    'hour': hour,
                    'time_segment': time_segment,
                    'predicted_engagement_delay_minutes': row['predicted_delay'],
                    'formatted_time': f"{day_name}, {hour}:00"
                })
                
        return pd.DataFrame(results)
    
    def schedule_campaign_sends(self, campaign_id: str, 
                            customer_ids: List[str], 
                            campaign_type: Optional[str] = None,
                            api_key: Optional[str] = None,
                            default_send_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Schedule a campaign to be sent at optimal times for each customer.
        
        Args:
            campaign_id: Campaign ID
            customer_ids: List of customer IDs
            campaign_type: Type of campaign
            api_key: API key for integration
            default_send_time: Default send time if predictions can't be made
            
        Returns:
            Dictionary with scheduling details
        """
        # Get optimal send times for all customers
        send_time_predictions = self.predict_optimal_times(
            customer_ids=customer_ids,
            campaign_type=campaign_type,
            n_recommendations=1
        )
        
        # Prepare scheduling data
        current_date = datetime.now().date()
        scheduling_data = []
        
        for _, row in send_time_predictions.iterrows():
            customer_id = row['customer_id']
            day_of_week = int(row['day_of_week'])
            hour = int(row['hour'])
            
            # Calculate the next occurrence of this day of week
            days_ahead = day_of_week - current_date.weekday()
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
                
            next_occurrence = current_date + timedelta(days=days_ahead)
            send_datetime = datetime.combine(next_occurrence, 
                                            datetime.min.time()) + timedelta(hours=hour)
            
            scheduling_data.append({
                'customer_id': customer_id,
                'send_datetime': send_datetime,
                'day_name': row['day_name'],
                'hour': hour
            })
            
        # If API key is provided, upload schedule
        if api_key:
            schedule_uploaded = self._upload_schedule_to_api(
                api_key=api_key,
                campaign_id=campaign_id,
                scheduling_data=scheduling_data
            )
        else:
            schedule_uploaded = False
            
        # Return scheduling information
        return {
            'campaign_id': campaign_id,
            'total_recipients': len(customer_ids),
            'schedule_created': True,
            'schedule_uploaded': schedule_uploaded,
            'scheduled_sends': scheduling_data
        }
    
    def _upload_schedule_to_api(self, api_key: str,
                              campaign_id: str,
                              scheduling_data: List[Dict[str, Any]]) -> bool:
        """Upload send schedule to API."""
        try:
            from utils.api_integration import APIIntegration
            
            api_client = APIIntegration(api_key=api_key)
            
            # Format data for API
            schedule_data = [{
                'customer_id': item['customer_id'],
                'send_time': item['send_datetime'].isoformat()
            } for item in scheduling_data]
            
            # Upload schedule
            result = api_client.upload_campaign_schedule(
                campaign_id=campaign_id,
                schedule_data=schedule_data
            )
            
            logger.info(f"Uploaded schedule for campaign {campaign_id} to API")
            return True
        except Exception as e:
            logger.error(f"Failed to upload schedule to API: {e}")
            return False
    
    def visualize_optimal_times(self, data: Optional[pd.DataFrame] = None,
                              title: Optional[str] = None) -> plt.Figure:
        """
        Visualize optimal send times in a heatmap.
        
        Args:
            data: Prediction data from predict_optimal_times
            title: Custom title for the plot
            
        Returns:
            Matplotlib figure
        """
        if data is None:
            if not hasattr(self, 'historical_data'):
                raise ValueError("No data available. Call load_historical_data first.")
                
            # Get sample customers
            sample_customers = self.historical_data['customer_id'].unique()[:10]
            data = self.predict_optimal_times(customer_ids=sample_customers)
            
        # Prepare data for heatmap
        heatmap_data = pd.DataFrame(
            index=list(self.day_mapping.values()),
            columns=range(24),
            data=0
        )
        
        # Get best prediction for each customer 
        best_times = data.loc[data.groupby('customer_id')['predicted_engagement_delay_minutes'].idxmin()]
        
        # Count occurrences of each day-hour combination
        for _, row in best_times.iterrows():
            day_name = row['day_name']
            hour = int(row['hour'])
            heatmap_data.loc[day_name, hour] += 1
            
        # Create heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d')
        
        plt.title(title or 'Optimal Send Times Heatmap')
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Week')
        plt.tight_layout()
        
        return plt.gcf()
    
    def export_optimal_times(self, customer_ids: List[str],
                          output_format: str = 'csv',
                          file_path: Optional[str] = None,
                          api_format: bool = False) -> Union[pd.DataFrame, str]:
        """
        Export optimal send times for customers.
        
        Args:
            customer_ids: List of customer IDs
            output_format: 'csv', 'json', or 'dataframe'
            file_path: Path to save the export file
            api_format: Whether to format for API import
            
        Returns:
            DataFrame or path to saved file
        """
        # Get optimal send times
        optimal_times = self.predict_optimal_times(customer_ids=customer_ids)
        
        # Format for API if requested
        if api_format:
            # Use only the best time for each customer
            best_times = optimal_times.loc[optimal_times.groupby('customer_id')['predicted_engagement_delay_minutes'].idxmin()]
            
            # Format for API
            api_data = []
            current_date = datetime.now().date()
            
            for _, row in best_times.iterrows():
                customer_id = row['customer_id']
                day_of_week = int(row['day_of_week'])
                hour = int(row['hour'])
                
                # Calculate the next occurrence of this day of week
                days_ahead = day_of_week - current_date.weekday()
                if days_ahead <= 0:  # Target day already happened this week
                    days_ahead += 7
                    
                next_occurrence = current_date + timedelta(days=days_ahead)
                send_datetime = datetime.combine(next_occurrence, 
                                                datetime.min.time()) + timedelta(hours=hour)
                
                api_data.append({
                    'customer_id': customer_id,
                    'optimal_send_time': send_datetime.isoformat(),
                    'day_of_week': row['day_name'],
                    'hour_of_day': hour
                })
                
            optimal_times = pd.DataFrame(api_data)
        
        # Export based on format
        if output_format == 'dataframe':
            return optimal_times
        elif output_format == 'csv':
            if not file_path:
                file_path = f"optimal_send_times_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            optimal_times.to_csv(file_path, index=False)
            return file_path
        elif output_format == 'json':
            if not file_path:
                file_path = f"optimal_send_times_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            optimal_times.to_json(file_path, orient='records')
            return file_path
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def save_model(self, directory: str = "models"):
        """Save the trained model to disk."""
        if not self.model:
            raise ValueError("No trained model to save.")
            
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(directory, f"send_time_optimizer_model_{timestamp}.joblib")
        
        # Save model
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'performance': self.model_performance,
            'feature_importance': self.feature_importance,
            'time_segments': self.time_segments,
            'day_mapping': self.day_mapping
        }
        
        metadata_path = os.path.join(directory, f"send_time_optimizer_metadata_{timestamp}.joblib")
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Model metadata saved to {metadata_path}")
        
        return {
            'model_path': model_path,
            'metadata_path': metadata_path
        }
    
    def load_model(self, model_path: str, metadata_path: Optional[str] = None):
        """Load a saved model from disk."""
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        if metadata_path:
            metadata = joblib.load(metadata_path)
            self.model_performance = metadata.get('performance', {})
            self.feature_importance = metadata.get('feature_importance', {})
            self.time_segments = metadata.get('time_segments', self.time_segments)
            self.day_mapping = metadata.get('day_mapping', self.day_mapping)
            logger.info(f"Model metadata loaded from {metadata_path}")

# Example usage
if __name__ == "__main__":
    optimizer = SendTimeOptimizer()
    
    # Load sample data
    data = optimizer.load_historical_data()
    
    # Train model
    performance = optimizer.train_model(algorithm='random_forest')
    print(f"Model RMSE: {performance['test_rmse_minutes']:.2f} minutes")
    
    # Get top feature importances
    top_features = sorted(performance['feature_importance'].items(), 
                          key=lambda x: x[1], reverse=True)[:5]
    print("\nTop feature importances:")
    for feature, importance in top_features:
        print(f"- {feature}: {importance:.4f}")
    
    # Predict optimal times for sample customers
    sample_customers = data['customer_id'].unique()[:5]
    optimal_times = optimizer.predict_optimal_times(
        customer_ids=sample_customers,
        campaign_type='promotional',
        n_recommendations=3
    )
    
    print("\nSample optimal send times:")
    for _, row in optimal_times.iterrows():
        print(f"Customer {row['customer_id']}: {row['formatted_time']} "
              f"(predicted delay: {row['predicted_engagement_delay_minutes']:.1f} minutes)")
    
    # Visualize optimal times
    optimizer.visualize_optimal_times(optimal_times, 
                                      title="Optimal Send Times for Sample Customers")
    plt.show() 