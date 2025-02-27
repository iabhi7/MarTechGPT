"""
Send Time Optimization Demo
---------------------------

This example demonstrates how to use the SendTimeOptimizer to determine
the optimal time to send marketing communications to customers.
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the optimizer
from advanced_features.send_time_optimizer.optimizer import SendTimeOptimizer

def main():
    print("⏰ API Send Time Optimization Demo")
    print("=" * 50)
    
    # Initialize the optimizer
    optimizer = SendTimeOptimizer()
    
    # Load historical data (will use sample data for demo)
    print("\n1️⃣ Loading and analyzing historical engagement data...")
    data = optimizer.load_historical_data()
    
    print(f"   Loaded data for {data['customer_id'].nunique()} customers")
    print(f"   Total engagement records: {len(data)}")
    
    # Train the model
    print("\n2️⃣ Training the send time optimization model...")
    performance = optimizer.train_model(algorithm='random_forest')
    
    print(f"   Model trained with accuracy of {performance['test_rmse_minutes']:.2f} minutes RMSE")
    
    # Display top feature importances
    print("\n   Top factors affecting optimal send time:")
    top_features = sorted(performance['feature_importance'].items(), 
                        key=lambda x: x[1], reverse=True)[:5]
    for feature, importance in top_features:
        print(f"   - {feature}: {importance:.4f}")
    
    # Get some sample customers for predictions
    sample_customers = data['customer_id'].unique()[:5]
    
    # Predict optimal send times
    print("\n3️⃣ Predicting optimal send times for sample customers...")
    optimal_times = optimizer.predict_optimal_times(
        customer_ids=sample_customers,
        campaign_type='promotional',
        n_recommendations=3
    )
    
    # Display results
    print("\n   Sample customer optimal send times:")
    
    for customer_id in sample_customers:
        customer_times = optimal_times[optimal_times['customer_id'] == customer_id]
        print(f"\n   Customer: {customer_id}")
        
        for i, (_, time) in enumerate(customer_times.iterrows(), 1):
            print(f"     Option {i}: {time['formatted_time']} "
                f"(predicted engagement delay: {time['predicted_engagement_delay_minutes']:.1f} minutes)")
    
    # Visualize optimal times
    print("\n4️⃣ Visualizing optimal send times...")
    fig = optimizer.visualize_optimal_times(
        data=optimal_times,
        title="Optimal Send Times for Sample Customers"
    )
    
    # Save the visualization
    img_path = "optimal_send_times.png"
    fig.savefig(img_path)
    print(f"   Heatmap saved to {img_path}")
    
    # Schedule campaign sends (simulation)
    print("\n5️⃣ Scheduling a campaign with optimal send times...")
    
    # All customers in the dataset
    all_customers = data['customer_id'].unique().tolist()
    
    # Schedule sends for up to 100 customers
    target_customers = all_customers[:min(100, len(all_customers))]
    
    schedule = optimizer.schedule_campaign_sends(
        campaign_id="DEMO-CAMPAIGN-001",
        customer_ids=target_customers,
        campaign_type="promotional"
    )
    
    print(f"   Campaign scheduled for {len(schedule['scheduled_sends'])} recipients")
    print(f"   Send times span from {min(s['send_datetime'] for s in schedule['scheduled_sends'])} "
         f"to {max(s['send_datetime'] for s in schedule['scheduled_sends'])}")
    
    # Export optimal times
    print("\n6️⃣ Exporting optimal send times...")
    export_path = optimizer.export_optimal_times(
        customer_ids=target_customers,
        output_format='csv',
        api_format=True
    )
    
    print(f"   Optimal send times exported to {export_path}")
    print("   This file can be imported directly to AI Cloud for campaign scheduling")
    
    # Save the model
    print("\n7️⃣ Saving the trained model...")
    save_result = optimizer.save_model()
    
    print(f"   Model saved to {save_result['model_path']}")
    print(f"   Model metadata saved to {save_result['metadata_path']}")
    
    print("\n✅ Send time optimization complete!")
    print("=" * 50)
    print("Benefits of time-optimized campaigns:")
    print("- Increased open rates by 15-25%")
    print("- Improved click-through rates by 10-20%")
    print("- Better customer experience by respecting individual engagement patterns")
    print("- Reduced unsubscribe rates through more relevant timing")
    
if __name__ == "__main__":
    main() 