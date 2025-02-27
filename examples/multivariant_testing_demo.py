"""
Multi-Variant Testing Demo
--------------------------

This example demonstrates how to use the MultiVariantTester to set up,
run, and analyze sophisticated multi-variant tests beyond simple A/B testing.
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

# Import the tester
from advanced_features.multivariant_testing.variant_optimizer import MultiVariantTester

def main():
    print("🧪 Netcore Multi-Variant Testing Demo")
    print("=" * 60)
    
    # Initialize the tester
    print("\n1️⃣ Initializing multi-variant testing framework...")
    tester = MultiVariantTester(
        significance_level=0.05,
        min_sample_size=100,
        auto_stop_enabled=True
    )
    
    # Create an email subject line test
    print("\n2️⃣ Creating a new multi-variant test for email subject lines...")
    subject_line_test = tester.create_test(
        test_name="Holiday Season Email Subject Line Test",
        variants=[
            {
                "name": "Control",
                "content": "Holiday Season Sale: 25% off all products"
            },
            {
                "name": "Emoji Variant",
                "content": "Holiday Season Sale: 25% off all products 🎄❄️"
            },
            {
                "name": "Urgency Variant",
                "content": "Last Chance: Holiday deals end in 24 hours!"
            },
            {
                "name": "Personalized Variant",
                "content": "[First Name], your holiday wishlist is on sale!"
            },
            {
                "name": "Question Variant",
                "content": "Ready for the holidays? Get 25% off everything!"
            }
        ],
        metrics=["open_rate", "click_rate", "conversion_rate"],
        primary_metric="conversion_rate",
        segment_dimensions=["device_type", "customer_tier", "age_group"]
    )
    
    test_id = subject_line_test["test_id"]
    print(f"   Created test with ID: {test_id}")
    print(f"   Testing {len(subject_line_test['variants'])} variants")
    print(f"   Primary metric: {subject_line_test['primary_metric']}")
    
    # Start the test
    print("\n3️⃣ Starting the test...")
    tester.start_test(test_id)
    
    # Simulate data collection
    print("\n4️⃣ Simulating data collection (1000 users)...")
    
    # Sample distribution data for realistic simulation
    variant_performance = {
        "control": {"open_rate": 0.20, "click_rate": 0.15, "conversion_rate": 0.10},
        "variant_1": {"open_rate": 0.25, "click_rate": 0.14, "conversion_rate": 0.09},  # Emoji variant
        "variant_2": {"open_rate": 0.23, "click_rate": 0.18, "conversion_rate": 0.12},  # Urgency variant
        "variant_3": {"open_rate": 0.28, "click_rate": 0.20, "conversion_rate": 0.15},  # Personalized variant
        "variant_4": {"open_rate": 0.22, "click_rate": 0.17, "conversion_rate": 0.11}   # Question variant
    }
    
    # Segment-specific adjustments
    segment_modifiers = {
        "device_type": {
            "mobile": {"open_rate": -0.02, "click_rate": 0.0, "conversion_rate": -0.03},
            "desktop": {"open_rate": 0.05, "click_rate": 0.03, "conversion_rate": 0.05},
            "tablet": {"open_rate": 0.02, "click_rate": 0.02, "conversion_rate": 0.02}
        },
        "customer_tier": {
            "free": {"open_rate": -0.05, "click_rate": -0.05, "conversion_rate": -0.03},
            "standard": {"open_rate": 0.0, "click_rate": 0.0, "conversion_rate": 0.0},
            "premium": {"open_rate": 0.10, "click_rate": 0.08, "conversion_rate": 0.07}
        },
        "age_group": {
            "18-24": {"open_rate": -0.03, "click_rate": -0.02, "conversion_rate": -0.02},
            "25-34": {"open_rate": 0.03, "click_rate": 0.05, "conversion_rate": 0.04},
            "35-44": {"open_rate": 0.05, "click_rate": 0.03, "conversion_rate": 0.03},
            "45+": {"open_rate": -0.02, "click_rate": -0.03, "conversion_rate": -0.04}
        }
    }
    
    # Simulate collecting data from 1000 users
    for i in range(1000):
        user_id = f"user_{i}"
        
        # Assign user to a variant
        variant = tester.get_variant_for_user(
            test_id=test_id,
            user_id=user_id,
            user_attributes=None  # In real usage, you'd pass user attributes for targeting
        )
        variant_id = variant["variant_id"]
        
        # Generate segment data for this user
        segment_data = {
            "device_type": random.choice(["mobile", "desktop", "tablet"]),
            "customer_tier": random.choice(["free", "standard", "premium"]),
            "age_group": random.choice(["18-24", "25-34", "35-44", "45+"])
        }
        
        # Get base performance for this variant
        base_performance = variant_performance[variant_id]
        
        # Apply segment modifiers
        modifiers = {
            metric: sum(segment_modifiers[dim][segment_data[dim]][metric] 
                       for dim in segment_data.keys())
            for metric in ["open_rate", "click_rate", "conversion_rate"]
        }
        
        # Calculate actual performance with some randomness
        open_rate_prob = max(0, min(1, base_performance["open_rate"] + modifiers["open_rate"]))
        open_rate = 1 if random.random() < open_rate_prob else 0
        
        # Only calculate click and conversion if the email was opened
        if open_rate:
            click_rate_prob = max(0, min(1, base_performance["click_rate"] + modifiers["click_rate"]))
            click_rate = 1 if random.random() < click_rate_prob else 0
            
            if click_rate:
                conversion_rate_prob = max(0, min(1, base_performance["conversion_rate"] + modifiers["conversion_rate"]))
                conversion_rate = 1 if random.random() < conversion_rate_prob else 0
            else:
                conversion_rate = 0
        else:
            click_rate = 0
            conversion_rate = 0
        
        # Add data to the test
        metrics_data = {
            "open_rate": open_rate,
            "click_rate": click_rate,
            "conversion_rate": conversion_rate
        }
        
        tester.add_test_data(
            test_id=test_id,
            variant_id=variant_id,
            metrics_data=metrics_data,
            segment_data=segment_data,
            user_id=user_id
        )
        
        # Display progress
        if (i + 1) % 200 == 0:
            print(f"   Processed data for {i + 1} users...")
    
    print("   Data collection complete!")
    
    # Calculate results
    print("\n5️⃣ Analyzing test results...")
    results = tester._calculate_test_results(test_id)
    
    # Display results
    print("\n   Overall Test Results:")
    for metric in subject_line_test["metrics"]:
        print(f"\n   {metric.replace('_', ' ').title()}:")
        
        for variant in subject_line_test["variants"]:
            variant_id = variant["variant_id"]
            variant_name = variant.get("name", variant_id)
            
            if variant_id in results["metrics"][metric]:
                value = results["metrics"][metric][variant_id]["mean"]
                print(f"     {variant_name}: {value:.2%}")
                
                # Print comparison if not control
                if variant_id != "control" and variant_id in results["comparison"][metric]:
                    comparison = results["comparison"][metric][variant_id]
                    
                    if comparison.get("significant", False):
                        relative_improvement = comparison.get("relative_improvement", 0) * 100
                        print(f"       {relative_improvement:+.1f}% vs control (significant)")
                    else:
                        relative_improvement = comparison.get("relative_improvement", 0) * 100
                        print(f"       {relative_improvement:+.1f}% vs control (not significant)")
    
    # Print winning variant
    if subject_line_test["winning_variant"]:
        winning_variant = next(v for v in subject_line_test["variants"] 
                              if v["variant_id"] == subject_line_test["winning_variant"])
        
        print(f"\n   Winning Variant: {winning_variant.get('name', winning_variant['variant_id'])}")
        print(f"   Confidence Level: {subject_line_test['confidence_level']:.2%}")
        print(f"   Recommended action: Roll out the winning variant to all users")
    else:
        print("\n   No statistically significant winner found.")
        print(f"   Recommended action: Continue the test to gather more data")
    
    # Visualize results
    print("\n6️⃣ Generating visualizations...")
    visualization = tester.visualize_results(
        test_id=test_id,
        output_path="multivariant_test_results.png",
        include_segment_charts=True
    )
    
    print(f"   Overall results chart saved to multivariant_test_results.png")
    
    # Show segment insights
    print("\n7️⃣ Analyzing segment performance...")
    
    segment_insights = tester.analyze_segments(test_id=test_id)
    print("\n   Top Segments by Performance:")
    
    for i, insight in enumerate(segment_insights[:3], 1):
        segment_desc = ", ".join([f"{k}={v}" for k, v in insight["segment_data"].items()])
        print(f"   {i}. {segment_desc}")
        print(f"      Best variant: {insight['best_variant_name']}")
        print(f"      {insight['metric']}: {insight['value']:.2%} ({insight['improvement']:+.1f}% vs. overall)")
    
    # Export results
    print("\n8️⃣ Exporting test results...")
    
    csv_path = tester.export_results(
        test_id=test_id,
        output_format="csv",
        output_path="multivariant_test_results.csv",
        include_segments=True
    )
    
    json_path = tester.export_results(
        test_id=test_id, 
        output_format="json",
        output_path="multivariant_test_results.json",
        include_segments=True
    )
    
    print(f"   Results exported to {csv_path} and {json_path}")
    
    # Netcore integration
    print("\n9️⃣ Integrating with Netcore...")
    
    # Simulate integration with Netcore API
    print("   Uploading test results to Netcore Cloud...")
    print("   Configuring automated campaign with the winning variant...")
    
    print("\n✅ Multi-variant test complete!")
    print("=" * 60)
    print("Benefits of multi-variant testing:")
    print("- Test multiple ideas simultaneously")
    print("- Discover segment-specific preferences")
    print("- Make data-driven decisions with statistical confidence")
    print("- Improve marketing effectiveness by 15-30%")
    
if __name__ == "__main__":
    main() 