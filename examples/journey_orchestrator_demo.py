"""
Customer Journey Orchestration Demo
-----------------------------------

This example demonstrates how to use the JourneyOrchestrator to create, 
manage, and optimize personalized customer journeys across multiple channels.
"""

import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the orchestrator
from advanced_features.journey_orchestrator.orchestrator import JourneyOrchestrator

def main():
    print("🛣️ Netcore Customer Journey Orchestrator Demo")
    print("=" * 60)
    
    # Initialize the orchestrator
    print("\n1️⃣ Initializing journey orchestration engine...")
    orchestrator = JourneyOrchestrator(
        use_predictive_optimization=True
    )
    
    # Create an e-commerce onboarding journey
    print("\n2️⃣ Creating a new customer journey for e-commerce onboarding...")
    journey = orchestrator.create_journey(
        journey_name="New Customer Onboarding Journey",
        entry_condition={
            "type": "simple",
            "field": "account_age_days",
            "operator": "less_than",
            "value": 7
        },
        steps=[
            {
                "step_id": "welcome_email",
                "name": "Welcome Email",
                "type": "message",
                "channel": "email",
                "content": {
                    "subject": "Welcome to our store, {{first_name}}!",
                    "body": "Thank you for joining! Here's what you can do next...",
                    "cta": "Browse Products"
                }
            },
            {
                "step_id": "wait_2_days",
                "name": "Wait 2 Days",
                "type": "delay",
                "wait_time": 48  # hours
            },
            {
                "step_id": "browse_check",
                "name": "Check Browse Activity",
                "type": "condition",
                "condition": {
                    "type": "simple",
                    "field": "browse_count",
                    "operator": "greater_than",
                    "value": 0
                },
                "success_step_id": "browse_reminder",
                "failure_step_id": "product_recommendations"
            },
            {
                "step_id": "product_recommendations",
                "name": "Product Recommendations",
                "type": "message",
                "channel": "email",
                "content": {
                    "subject": "Products picked for you, {{first_name}}",
                    "body": "We thought you might like these products..."
                }
            },
            {
                "step_id": "browse_reminder",
                "name": "Browse Reminder",
                "type": "message",
                "channel": "push",
                "content": {
                    "title": "Complete your purchase",
                    "body": "Items are waiting in your cart!"
                }
            },
            {
                "step_id": "wait_1_day",
                "name": "Wait 1 Day",
                "type": "delay",
                "wait_time": 24  # hours
            },
            {
                "step_id": "purchase_check",
                "name": "Check Purchase",
                "type": "condition",
                "condition": {
                    "type": "simple",
                    "field": "has_purchased",
                    "operator": "equals",
                    "value": True
                },
                "success_step_id": "purchase_goal",
                "failure_step_id": "discount_offer"
            },
            {
                "step_id": "discount_offer",
                "name": "Special Discount",
                "type": "message",
                "channel": "email",
                "content": {
                    "subject": "Special 15% discount for you!",
                    "body": "Use code WELCOME15 to get 15% off your first purchase."
                },
                "next_step_id": "exit"
            },
            {
                "step_id": "purchase_goal",
                "name": "First Purchase",
                "type": "goal",
                "goal_name": "first_purchase",
                "next_step_id": "thank_you"
            },
            {
                "step_id": "thank_you",
                "name": "Thank You Message",
                "type": "message",
                "channel": "sms",
                "content": {
                    "body": "Thank you for your purchase! Use code THANKS10 for 10% off your next order."
                },
                "next_step_id": "exit"
            },
            {
                "step_id": "exit",
                "name": "Exit Journey",
                "type": "exit"
            }
        ],
        journey_goal="first_purchase",
        description="Onboarding journey to guide new customers to their first purchase",
        max_duration_days=14
    )
    
    journey_id = journey["journey_id"]
    print(f"   Created journey with ID: {journey_id}")
    print(f"   Journey goal: {journey['journey_goal']}")
    print(f"   {len(journey['steps'])} steps in the journey")
    
    # Start the journey
    print("\n3️⃣ Starting the journey...")
    orchestrator.start_journey(journey_id)
    
    # Simulate customers entering the journey
    print("\n4️⃣ Simulating customers entering the journey (100 customers)...")
    
    # Customer profiles for simulation
    customer_profiles = [
        {
            "name": "Mobile App Users",
            "attributes": {
                "device_type": "mobile",
                "acquisition_source": "app_store",
                "average_session_time": lambda: random.uniform(3, 8),
                "browse_probability": 0.7,
                "purchase_probability": 0.4
            },
            "count": 35
        },
        {
            "name": "Social Media Referrals",
            "attributes": {
                "device_type": lambda: random.choice(["mobile", "desktop"]),
                "acquisition_source": "social_media",
                "average_session_time": lambda: random.uniform(1, 4),
                "browse_probability": 0.8,
                "purchase_probability": 0.25
            },
            "count": 25
        },
        {
            "name": "Search Engine Visitors",
            "attributes": {
                "device_type": "desktop",
                "acquisition_source": "organic_search",
                "average_session_time": lambda: random.uniform(5, 12),
                "browse_probability": 0.6,
                "purchase_probability": 0.3
            },
            "count": 25
        },
        {
            "name": "Email Campaign Subscribers",
            "attributes": {
                "device_type": lambda: random.choice(["mobile", "desktop", "tablet"]),
                "acquisition_source": "email_campaign",
                "average_session_time": lambda: random.uniform(2, 6),
                "browse_probability": 0.5,
                "purchase_probability": 0.35
            },
            "count": 15
        }
    ]
    
    # Generate customer data
    customer_records = []
    
    for profile in customer_profiles:
        for i in range(profile["count"]):
            # Create a base customer record
            customer_id = f"customer_{len(customer_records) + 1}"
            
            # Generate customer attributes
            attributes = {}
            
            for key, value in profile["attributes"].items():
                if callable(value):
                    attributes[key] = value()
                else:
                    attributes[key] = value
            
            # Add common customer data
            customer_data = {
                "customer_id": customer_id,
                "first_name": f"User{len(customer_records) + 1}",
                "email": f"user{len(customer_records) + 1}@example.com",
                "account_age_days": random.randint(1, 5),
                "device_type": attributes.get("device_type", "unknown"),
                "acquisition_source": attributes.get("acquisition_source", "unknown"),
                "average_session_time": attributes.get("average_session_time", 0),
                "browse_count": 0,
                "has_purchased": False,
                "profile_type": profile["name"]
            }
            
            customer_records.append({
                "data": customer_data,
                "browse_probability": attributes.get("browse_probability", 0.5),
                "purchase_probability": attributes.get("purchase_probability", 0.2)
            })
    
    # Add customers to the journey
    for i, customer in enumerate(customer_records):
        orchestrator.add_customer_to_journey(
            journey_id=journey_id,
            customer_id=customer["data"]["customer_id"],
            customer_data=customer["data"]
        )
        
        if (i + 1) % 20 == 0:
            print(f"   Added {i + 1} customers to the journey...")
    
    print("   All customers added to the journey!")
    
    # Simulate journey progression with events
    print("\n5️⃣ Simulating customer journey progression over 7 days...")
    
    for day in range(1, 8):
        print(f"\n   Day {day}:")
        
        # Process each customer
        for customer in customer_records:
            customer_id = customer["data"]["customer_id"]
            customer_key = f"{journey_id}_{customer_id}"
            
            # Skip if customer is no longer active in the journey
            if customer_key not in orchestrator.active_customer_journeys:
                continue
                
            journey_state = orchestrator.active_customer_journeys[customer_key]
            
            if not journey_state.get("active", False):
                continue
                
            # Simulate browsing event
            if not customer["data"]["has_purchased"] and random.random() < customer["browse_probability"]:
                # Customer browsed the site
                customer["data"]["browse_count"] += 1
                
                # Update journey with this event
                orchestrator.advance_customer_journey(
                    journey_id=journey_id,
                    customer_id=customer_id,
                    event_data={"browse_count": customer["data"]["browse_count"]}
                )
                
            # Simulate purchase event
            if not customer["data"]["has_purchased"] and random.random() < customer["purchase_probability"]:
                # Customer made a purchase
                customer["data"]["has_purchased"] = True
                
                # Update journey with this event
                orchestrator.advance_customer_journey(
                    journey_id=journey_id,
                    customer_id=customer_id,
                    event_data={"has_purchased": True}
                )
        
        # Count active customers and conversions
        active_count = 0
        converted_count = 0
        
        for customer_key, journey_state in orchestrator.active_customer_journeys.items():
            if not customer_key.startswith(f"{journey_id}_"):
                continue
                
            if journey_state.get("active", False):
                active_count += 1
                
            if journey_state.get("goal_achieved", False):
                converted_count += 1
                
        print(f"   Active customers: {active_count}")
        print(f"   Conversions: {converted_count}")
        
        # Advance time for delay steps
        for customer_key, journey_state in orchestrator.active_customer_journeys.items():
            if not customer_key.startswith(f"{journey_id}_"):
                continue
                
            if not journey_state.get("active", False):
                continue
                
            # Check if waiting on a delay step
            if journey_state.get("next_action_date"):
                next_action = datetime.fromisoformat(journey_state["next_action_date"])
                
                # Simulate that a day has passed
                if next_action <= datetime.now() + timedelta(days=1):
                    # Clear the wait time and advance
                    orchestrator.advance_customer_journey(
                        journey_id=journey_id,
                        customer_id=journey_state["customer_id"]
                    )
    
    # Analyze results
    print("\n6️⃣ Analyzing journey performance...")
    performance = orchestrator.analyze_journey_performance(journey_id)
    
    print(f"\n   Overall Conversion Rate: {performance['conversion_rate']:.2%}")
    print(f"   Total Customers: {performance['total_customers']}")
    print(f"   Completed Customers: {performance['completed_customers']}")
    print(f"   Active Customers: {performance['active_customers']}")
    
    if performance.get('avg_conversion_time'):
        print(f"   Average Time to Conversion: {performance['avg_conversion_time']:.1f} hours")
    
    # Show channel performance
    print("\n   Channel Performance:")
    for channel, metrics in performance.get('channel_performance', {}).items():
        print(f"   - {channel.capitalize()}: {metrics.get('conversion_rate', 0):.2%} conversion rate")
    
    # Visualize the journey
    print("\n7️⃣ Generating journey visualization...")
    orchestrator.visualize_journey(
        journey_id=journey_id,
        output_path="journey_visualization.png"
    )
    
    print("   Journey visualization saved to journey_visualization.png")
    
    # Export analytics
    print("\n8️⃣ Exporting journey analytics...")
    json_path = orchestrator.export_journey_analytics(
        journey_id=journey_id,
        output_format="json",
        output_path="journey_analytics.json"
    )
    
    csv_path = orchestrator.export_journey_analytics(
        journey_id=journey_id,
        output_format="csv",
        output_path="journey_analytics.csv"
    )
    
    print(f"   Analytics exported to {json_path} and {csv_path}")
    
    # Train predictive model
    print("\n9️⃣ Training predictive model for journey optimization...")
    model_results = orchestrator.train_conversion_model(
        journey_id=journey_id,
        model_name="conversion",
        model_type="random_forest"
    )
    
    print(f"   Model trained with {model_results['test_accuracy']:.2%} accuracy")
    print("   Top predictive factors:")
    
    for feature, importance in sorted(
        model_results.get('feature_importances', {}).items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:5]:
        print(f"   - {feature}: {importance:.4f}")
    
    print("\n✅ Customer Journey Orchestration complete!")
    print("=" * 60)
    print("Benefits of AI-powered customer journeys:")
    print("- Deliver personalized experiences across channels")
    print("- Optimize messaging timing and sequence")
    print("- Increase conversion rates by targeting high-propensity customers")
    print("- Gain insights into customer behavior patterns")
    
if __name__ == "__main__":
    main() 