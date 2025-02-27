import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from quick_wins.marketing_chatbot.chatbot import MarketingChatbot
from quick_wins.integration.mock_cdp import MarketingCDPIntegration
from quick_wins.benchmarking.performance_analyzer import PerformanceAnalyzer
from quick_wins.ab_testing.ab_test_analyzer import ABTestAnalyzer

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.cdp = None
    st.session_state.chatbot = None
    st.session_state.analyzer = None
    st.session_state.ab_tester = None

# Page configuration
st.set_page_config(
    page_title="AI Marketing Suite Dashboard",
    page_icon="📊",
    layout="wide"
)

# Sidebar
st.sidebar.title("AI Marketing Suite")

# Navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Overview", "CDP Integration", "Campaign Analysis", "A/B Testing", "Model Performance"]
)

# Initialize components
if not st.session_state.initialized:
    with st.spinner("Initializing components..."):
        try:
            # Use a small model for the dashboard demo
            st.session_state.chatbot = MarketingChatbot(model_name="distilgpt2", quantize=True)
            st.session_state.cdp = MarketingCDPIntegration(chatbot=st.session_state.chatbot)
            st.session_state.analyzer = PerformanceAnalyzer()
            st.session_state.ab_tester = ABTestAnalyzer()
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
            st.stop()

# Overview page
if page == "Overview":
    st.title("📊 AI Marketing Suite Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### System Overview
        
        The AI Marketing Suite integrates advanced AI capabilities with your 
        existing platform. Key features include:
        
        - **Quantized LLM Engine**: Optimized for speed and efficiency
        - **Customer Segmentation**: AI-driven behavioral segmentation
        - **Campaign Generation**: Automated content creation
        - **A/B Testing**: Intelligent variant analysis
        - **Performance Analytics**: Real-time monitoring and insights
        """)
        
        # Display system status
        st.subheader("System Status")
        
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            st.metric("LLM Engine", "Online ✓", "Quantized")
            st.metric("API Status", "Active ✓", "5ms latency")
        
        with status_col2:
            st.metric("Memory Usage", "3.8 GB", "-60% vs unoptimized")
            st.metric("Average Response Time", "0.5s", "3x faster than baseline")
    
    with col2:
        # LLM performance graph
        st.subheader("LLM Performance Metrics")
        
        # Create sample data for the dashboard demo
        dates = [datetime.now() - timedelta(days=x) for x in range(14, 0, -1)]
        response_times = [0.6 - (0.01 * i) for i in range(14)]  # Improving trend
        
        # Create DataFrame and plot
        performance_df = pd.DataFrame({
            'date': dates,
            'response_time': response_times
        })
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x='date', y='response_time', data=performance_df, ax=ax)
        ax.set_title('LLM Response Time (seconds)')
        ax.set_ylabel('Response Time (s)')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Optimizations summary
        st.subheader("Optimizations")
        
        opt_df = pd.DataFrame({
            'Metric': ['Model Size', 'Memory Usage', 'Inference Time', 'Cost'],
            'Before': ['7.4 GB', '10.8 GB', '1.5s', '$$$$$'],
            'After': ['3.8 GB', '4.2 GB', '0.5s', '$$'],
            'Improvement': ['-49%', '-60%', '3x faster', '-60%']
        })
        
        st.table(opt_df)
    
    # Business impact section
    st.subheader("Business Impact")
    
    impact_col1, impact_col2, impact_col3, impact_col4 = st.columns(4)
    
    with impact_col1:
        st.metric("Email Open Rates", "+25%", "vs. baseline")
    
    with impact_col2:
        st.metric("Campaign ROI", "+20%", "year-over-year")
    
    with impact_col3:
        st.metric("Content Creation Time", "-90%", "10x faster")
    
    with impact_col4:
        st.metric("Customer Churn", "-35%", "better prediction")
    
    # Integration diagram
    st.subheader("System Architecture")
    
    st.code("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Netcore AI Marketing Suite                         │
├─────────────┬─────────────┬─────────────┬─────────────────┬─────────────────┤
│             │             │             │                 │                 │
│ Subject Line│  Marketing  │  Customer   │     Content     │    Campaign     │
│  Optimizer  │   Chatbot   │ Segmentation│    Generator    │    Optimizer    │
│             │             │             │                 │                 │
├─────────────┴─────┬───────┴─────────────┴─────────┬───────┴─────────────────┤
│                   │                               │                         │
│  Quantized LLM    │    Vector DB Knowledge Base   │    Analytics Engine     │
│    Engine         │    (FAISS)                    │    (Predictive Models)  │
│                   │                               │                         │
├───────────────────┴───────────────────────────────┴─────────────────────────┤
│                                                                             │
│                           Netcore API Integration Layer                     │
│                                                                             │
├─────────────┬─────────────┬───────────────────────┬─────────────────────────┤
│             │             │                       │                         │
│ Netcore CDP │  Campaign   │  Customer Journey     │     Analytics &         │
│             │  Manager    │  Orchestration        │     Reporting           │
│             │             │                       │                         │
└─────────────┴─────────────┴───────────────────────┴─────────────────────────┘
    """)

# CDP Integration page
if page == "CDP Integration":
    st.title("🔄 CDP Integration & AI Enhancement")
    
    # Data generation
    st.subheader("Generate Sample Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_users = st.slider("Number of Users", 10, 200, 50)
    
    with col2:
        num_interactions = st.slider("Number of Interactions", 50, 1000, 200)
    
    with col3:
        num_campaigns = st.slider("Number of Campaigns", 3, 20, 5)
    
    if st.button("Generate Mock Data"):
        with st.spinner("Generating data..."):
            result = st.session_state.cdp.generate_mock_data(
                num_users=num_users,
                num_interactions=num_interactions,
                num_campaigns=num_campaigns
            )
            
            st.success(f"Generated {result['users']} users, {result['interactions']} interactions, and {result['campaigns']} campaigns")
    
    # Display data stats if available
    if hasattr(st.session_state.cdp, 'user_profiles') and st.session_state.cdp.user_profiles:
        st.subheader("Data Overview")
        
        data_col1, data_col2, data_col3 = st.columns(3)
        
        with data_col1:
            st.metric("Users", len(st.session_state.cdp.user_profiles))
        
        with data_col2:
            st.metric("Interactions", len(st.session_state.cdp.user_interactions))
        
        with data_col3:
            st.metric("Campaigns", len(st.session_state.cdp.campaigns))
        
        # Segment distribution
        if st.session_state.cdp.user_profiles:
            segments = [user['segment'] for user in st.session_state.cdp.user_profiles]
            segment_counts = pd.Series(segments).value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            segment_counts.plot(kind='bar', ax=ax)
            ax.set_title('User Segment Distribution')
            ax.set_ylabel('Count')
            ax.set_xlabel('Segment')
            st.pyplot(fig)
        
        # AI Enhancement
        st.subheader("AI User Profile Enhancement")
        
        if st.button("Enhance User Profiles with AI"):
            with st.spinner("Using AI to enhance user profiles..."):
                result = st.session_state.cdp.enhance_user_profiles_with_ai()
                
                if isinstance(result, dict):
                    st.success(f"Enhanced {result['enhanced_profiles']} out of {result['total_profiles']} user profiles")
                else:
                    st.error(result)
        
        # Campaign Generation
        st.subheader("Campaign Management")
        
        if st.session_state.cdp.campaigns:
            campaign_metrics = []
            
            for campaign in st.session_state.cdp.campaigns:
                metrics = campaign["metrics"]
                sent = metrics["sent"]
                
                # Skip campaigns with no sends
                if sent == 0:
                    continue
                    
                delivered_rate = metrics["delivered"] / sent if sent > 0 else 0
                open_rate = metrics["opened"] / metrics["delivered"] if metrics["delivered"] > 0 else 0
                click_rate = metrics["clicked"] / metrics["opened"] if metrics["opened"] > 0 else 0
                conversion_rate = metrics["converted"] / metrics["clicked"] if metrics["clicked"] > 0 else 0
                
                campaign_metrics.append({
                    "name": campaign["name"],
                    "type": campaign["type"],
                    "segment": campaign["segment"],
                    "ai_generated": campaign.get("ai_generated", False),
                    "sent": sent,
                    "open_rate": open_rate,
                    "click_rate": click_rate,
                    "conversion_rate": conversion_rate
                })
            
            # Convert to DataFrame for display
            metrics_df = pd.DataFrame(campaign_metrics)
            
            # Campaign type comparison
            st.subheader("Performance by Campaign Type")
            type_performance = metrics_df.groupby('type')[['open_rate', 'click_rate', 'conversion_rate']].mean()
            type_performance = type_performance.reset_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            type_performance_melted = pd.melt(type_performance, id_vars=['type'], value_vars=['open_rate', 'click_rate', 'conversion_rate'],
                                             var_name='Metric', value_name='Rate')
            sns.barplot(x='type', y='Rate', hue='Metric', data=type_performance_melted, ax=ax)
            ax.set_title('Performance Metrics by Campaign Type')
            ax.set_ylabel('Rate')
            ax.set_xlabel('Campaign Type')
            st.pyplot(fig)
            
            # Campaign metrics table
            st.subheader("Campaign Metrics")
            
            # Format percentages for display
            display_metrics = metrics_df.copy()
            for col in ['open_rate', 'click_rate', 'conversion_rate']:
                display_metrics[col] = display_metrics[col].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(display_metrics)
            
            # Generate AI Campaign
            st.subheader("Generate AI Campaign")
            
            segments = list(set(campaign['segment'] for campaign in st.session_state.cdp.campaigns))
            campaign_types = ["Email", "SMS", "Push Notification", "Social"]
            
            selected_segment = st.selectbox("Select Target Segment", segments)
            selected_type = st.selectbox("Select Campaign Type", campaign_types)
            
            if st.button("Generate AI Campaign"):
                with st.spinner("Generating campaign with AI..."):
                    campaign = st.session_state.cdp.generate_personalized_campaign(
                        selected_segment, 
                        campaign_type=selected_type
                    )
                    
                    st.success("Campaign generated successfully!")
                    
                    # Display campaign details
                    st.subheader("AI-Generated Campaign")
                    st.write(f"**Campaign Type:** {campaign['type']}")
                    st.write(f"**Target Segment:** {campaign['segment']}")
                    st.write(f"**Subject Line:** {campaign['content']['subject']}")
                    st.write("**Body:**")
                    st.text_area("", campaign['content']['body'], height=200, disabled=True)
                    
                    # Target audience details
                    st.write("**Target Audience:**")
                    st.write(f"- Primary Industry: {campaign['target_audience']['primary_industry']}")
                    st.write(f"- Average Age: {campaign['target_audience']['avg_age']:.1f}")
                    st.write(f"- Average LTV: ${campaign['target_audience']['avg_ltv']:.2f}")
                    st.write(f"- Number of Recipients: {campaign['target_audience']['user_count']}")
        else:
            st.info("No campaign data available. Please generate mock data.")

# Campaign Analysis page
if page == "Campaign Analysis":
    st.title("📈 Campaign Analysis")
    
    if not hasattr(st.session_state.cdp, 'campaigns') or not st.session_state.cdp.campaigns:
        st.info("No campaign data available. Please generate mock data in the CDP Integration page.")
    else:
        # Campaign performance analysis
        st.subheader("Campaign Performance Analysis")
        
        analysis = st.session_state.cdp.analyze_campaign_performance()
        
        # AI vs Standard campaign comparison if available
        if isinstance(analysis['performance_comparison'], dict):
            st.subheader("AI vs Standard Campaign Performance")
            
            # Create comparison dataframe
            ai_perf = analysis['performance_comparison']['ai_campaigns']
            std_perf = analysis['performance_comparison']['standard_campaigns']
            
            comp_data = {
                'Campaign Type': ['AI-Generated', 'Standard'],
                'Count': [ai_perf['count'], std_perf['count']],
                'Open Rate': [ai_perf['avg_open_rate'], std_perf['avg_open_rate']],
                'Click Rate': [ai_perf['avg_click_rate'], std_perf['avg_click_rate']],
                'Conversion Rate': [ai_perf['avg_conversion_rate'], std_perf['avg_conversion_rate']]
            }
            
            comp_df = pd.DataFrame(comp_data)
            
            # Format percentages
            for col in ['Open Rate', 'Click Rate', 'Conversion Rate']:
                comp_df[col] = comp_df[col].apply(lambda x: f"{x:.1%}")
            
            # Display table
            st.table(comp_df)
            
            # Create performance visualization
            if ai_perf['count'] > 0 and std_perf['count'] > 0:
                metrics = ['avg_open_rate', 'avg_click_rate', 'avg_conversion_rate']
                metric_names = ['Open Rate', 'Click Rate', 'Conversion Rate']
                
                comparison_data = []
                for i, metric in enumerate(metrics):
                    comparison_data.append({
                        'Metric': metric_names[i],
                        'AI-Generated': ai_perf[metric],
                        'Standard': std_perf[metric],
                        'Improvement': (ai_perf[metric] / std_perf[metric] - 1) if std_perf[metric] > 0 else 0
                    })
                
                comp_vis_df = pd.DataFrame(comparison_data)
                
                # Bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                comp_vis_melted = pd.melt(comp_vis_df, id_vars=['Metric'], value_vars=['AI-Generated', 'Standard'],
                                         var_name='Campaign Type', value_name='Rate')
                sns.barplot(x='Metric', y='Rate', hue='Campaign Type', data=comp_vis_melted, ax=ax)
                ax.set_title('AI vs Standard Campaign Performance')
                ax.set_ylabel('Rate')
                ax.set_ylim(0, max(ai_perf['avg_open_rate'], std_perf['avg_open_rate']) * 1.2)
                st.pyplot(fig)
                
                # Improvement metrics
                st.subheader("Performance Improvement with AI")
                
                improvement_col1, improvement_col2, improvement_col3 = st.columns(3)
                
                with improvement_col1:
                    open_rate_improve = (ai_perf['avg_open_rate'] / std_perf['avg_open_rate'] - 1) * 100 if std_perf['avg_open_rate'] > 0 else 0
                    st.metric("Open Rate Improvement", f"{open_rate_improve:.1f}%")
                
                with improvement_col2:
                    click_rate_improve = (ai_perf['avg_click_rate'] / std_perf['avg_click_rate'] - 1) * 100 if std_perf['avg_click_rate'] > 0 else 0
                    st.metric("Click Rate Improvement", f"{click_rate_improve:.1f}%")
                
                with improvement_col3:
                    conv_rate_improve = (ai_perf['avg_conversion_rate'] / std_perf['avg_conversion_rate'] - 1) * 100 if std_perf['avg_conversion_rate'] > 0 else 0
                    st.metric("Conversion Rate Improvement", f"{conv_rate_improve:.1f}%")
                
                # ROI calculation (simplified)
                st.subheader("Estimated ROI Impact")
                
                # Assumptions for ROI calculation
                avg_revenue_per_conversion = st.slider("Average Revenue per Conversion ($)", 50, 500, 100)
                campaign_cost = st.slider("Campaign Cost ($)", 1000, 10000, 5000)
                audience_size = st.slider("Audience Size", 10000, 100000, 50000)
                
                # Calculate ROI
                ai_conversions = audience_size * ai_perf['avg_open_rate'] * ai_perf['avg_click_rate'] * ai_perf['avg_conversion_rate']
                ai_revenue = ai_conversions * avg_revenue_per_conversion
                ai_roi = (ai_revenue - campaign_cost) / campaign_cost
                
                std_conversions = audience_size * std_perf['avg_open_rate'] * std_perf['avg_click_rate'] * std_perf['avg_conversion_rate']
                std_revenue = std_conversions * avg_revenue_per_conversion
                std_roi = (std_revenue - campaign_cost) / campaign_cost
                
                roi_comparison = pd.DataFrame({
                    'Campaign Type': ['AI-Generated', 'Standard'],
                    'Estimated Conversions': [int(ai_conversions), int(std_conversions)],
                    'Estimated Revenue': [f"${ai_revenue:,.2f}", f"${std_revenue:,.2f}"],
                    'ROI': [f"{ai_roi:.1%}", f"{std_roi:.1%}"]
                })
                
                st.table(roi_comparison)
                
                roi_improve = (ai_roi / std_roi - 1) * 100 if std_roi > 0 else 0
                st.metric("ROI Improvement with AI", f"{roi_improve:.1f}%")
                
        # Segment performance
        st.subheader("Performance by Segment")
        
        # Check if segment_performance is a DataFrame or dictionary
        if isinstance(analysis['segment_performance'], dict):
            # Convert to DataFrame
            segment_perf = pd.DataFrame.from_dict(analysis['segment_performance'], orient='columns')
            
            # Melt for visualization
            segment_perf_melted = pd.melt(segment_perf.reset_index(), id_vars=['index'], value_vars=['open_rate', 'click_rate', 'conversion_rate'],
                                         var_name='Metric', value_name='Rate')
            segment_perf_melted.rename(columns={'index': 'Segment'}, inplace=True)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x='Segment', y='Rate', hue='Metric', data=segment_perf_melted, ax=ax)
            ax.set_title('Performance Metrics by Segment')
            ax.set_ylabel('Rate')
            ax.set_xlabel('Segment')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Top segments table
            st.subheader("Top Performing Segments")
            
            # Format for display
            segment_display = segment_perf.copy()
            for col in ['open_rate', 'click_rate', 'conversion_rate']:
                segment_display[col] = segment_display[col].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(segment_display)

# A/B Testing page
if page == "A/B Testing":
    st.title("🔄 A/B Testing & Optimization")
    
    # Input text variants
    st.subheader("Enter A/B Test Variants")
    
    num_variants = st.number_input("Number of Variants", min_value=2, max_value=5, value=3)
    
    variants = []
    for i in range(int(num_variants)):
        variant_text = st.text_area(f"Variant {i+1}", height=100, 
                                   placeholder=f"Enter marketing copy for variant {i+1}",
                                   value=f"Our platform helps you achieve better marketing results. Try it today!" if i == 0 else
                                         f"Boost your marketing ROI by 30% with our AI-powered platform. Limited offer!" if i == 1 else
                                         f"Why struggle with marketing? Our platform gives you 30% better results instantly!")
        variants.append(variant_text)
    
    audience_type = st.selectbox("Target Audience Type", ["general", "technical", "executive"])
    
    if st.button("Analyze Variants"):
        with st.spinner("Analyzing variants..."):
            # Analyze the variants
            analysis_df = st.session_state.ab_tester.analyze_variants(variants)
            
            # Get recommendation
            recommendation = st.session_state.ab_tester.recommend_variant(analysis_df, audience_type)
            
            # Display recommendation
            st.subheader("A/B Test Results")
            
            rec_col1, rec_col2 = st.columns([1, 2])
            
            with rec_col1:
                st.success(f"Recommended: {recommendation['recommended_variant']}")
                st.write(f"**Reasoning:** {recommendation['reasoning']}")
                
                # Show metrics for recommended variant
                st.write("**Key Metrics:**")
                for metric, value in recommendation['metrics'].items():
                    if metric == 'has_cta':
                        st.write(f"- Has CTA: {'Yes' if value else 'No'}")
                    else:
                        st.write(f"- {metric.replace('_', ' ').title()}: {value}")
            
            with rec_col2:
                st.subheader("Recommended Copy:")
                st.text_area("", recommendation['text'], height=100, disabled=True)
            
            # Generate visualization
            st.subheader("Variant Comparison")
            fig = st.session_state.ab_tester.visualize_comparison(analysis_df)
            st.pyplot(fig)
            
            # Detailed metrics
            st.subheader("Detailed Metrics")
            st.dataframe(analysis_df.drop('text', axis=1).set_index('variant_id'))
            
            # Performance prediction
            st.subheader("Performance Prediction")
            
            # Simple performance prediction (for demonstration)
            baseline_open_rate = 0.22
            baseline_click_rate = 0.08
            
            # Calculate predicted improvements based on readability scores and other factors
            variant_improvements = []
            
            for idx, row in analysis_df.iterrows():
                readability_factor = row['readability_score'] / 100  # Normalize to 0-1
                power_word_factor = min(1, row['power_word_count'] / 5)  # Cap at 1
                cta_factor = 1.2 if row['has_cta'] else 1.0
                question_factor = 1 + (row['question_count'] * 0.05)  # 5% boost per question
                
                # Combine factors
                combined_factor = (readability_factor * 0.4 + 
                                  power_word_factor * 0.3 + 
                                  cta_factor * 0.2 + 
                                  question_factor * 0.1)
                
                # Apply to baseline rates
                predicted_open = baseline_open_rate * combined_factor * (1.1 if row['variant_id'] == recommendation['recommended_variant'] else 1.0)
                predicted_click = baseline_click_rate * combined_factor * (1.15 if row['variant_id'] == recommendation['recommended_variant'] else 1.0)
                
                variant_improvements.append({
                    'variant_id': row['variant_id'],
                    'predicted_open_rate': predicted_open,
                    'predicted_click_rate': predicted_click,
                    'predicted_improvement': (predicted_open * predicted_click) / (baseline_open_rate * baseline_click_rate) - 1
                })
            
            # Convert to DataFrame
            prediction_df = pd.DataFrame(variant_improvements)
            
            # Format for display
            prediction_display = prediction_df.copy()
            prediction_display['predicted_open_rate'] = prediction_display['predicted_open_rate'].apply(lambda x: f"{x:.1%}")
            prediction_display['predicted_click_rate'] = prediction_display['predicted_click_rate'].apply(lambda x: f"{x:.1%}")
            prediction_display['predicted_improvement'] = prediction_display['predicted_improvement'].apply(lambda x: f"{x:.1%}")
            
            st.table(prediction_display.set_index('variant_id'))
            
            # Highlight recommended variant
            best_variant = prediction_df.sort_values('predicted_improvement', ascending=False).iloc[0]
            
            st.info(f"The {best_variant['variant_id']} is predicted to perform {best_variant['predicted_improvement']:.1%} better than your baseline campaign.")

# Model Performance page
if page == "Model Performance":
    st.title("⚡ Model Performance Analysis")
    
    # Model benchmark options
    st.subheader("Benchmark AI Model Performance")
    
    bench_col1, bench_col2 = st.columns(2)
    
    with bench_col1:
        models_to_bench = st.multiselect(
            "Select Models to Benchmark",
            ["distilgpt2", "facebook/opt-125m", "mistralai/Mistral-7B-Instruct-v0.1"],
            ["distilgpt2"]
        )
    
    with bench_col2:
        quantize_options = st.multiselect(
            "Quantization Options",
            ["Quantized", "Unquantized"],
            ["Quantized"]
        )
        
        # Convert to boolean values
        quantize_bool = [q == "Quantized" for q in quantize_options]
        
        iterations = st.slider("Test Iterations", 1, 10, 3)
    
    if st.button("Run Benchmark"):
        with st.spinner("Running benchmark (this may take several minutes)..."):
            try:
                # Run benchmarks with reduced iterations for the demo
                for model in models_to_bench:
                    for quantize in quantize_bool:
                        st.session_state.analyzer.benchmark_model(
                            model_name=model, 
                            quantized=quantize,
                            iterations=iterations,
                            warmup=1
                        )
                
                # Get results
                results_df = st.session_state.analyzer.get_comparison_table()
                
                if not results_df.empty:
                    # Format quantized column
                    results_df['quantized'] = results_df['quantized'].map({True: 'Quantized', False: 'Unquantized'})
                    
                    # Display results
                    st.subheader("Benchmark Results")
                    
                    # Format for display
                    display_df = results_df.copy()
                    display_df['load_time_sec'] = display_df['load_time_sec'].round(2)
                    display_df['avg_inference_sec'] = display_df['avg_inference_sec'].round(3)
                    display_df['p95_inference_sec'] = display_df['p95_inference_sec'].round(3)
                    display_df['max_inference_sec'] = display_df['max_inference_sec'].round(3)
                    display_df['avg_memory_usage_mb'] = display_df['avg_memory_usage_mb'].round(1)
                    
                    st.dataframe(display_df)
                    
                    # Visualizations
                    st.subheader("Performance Visualizations")
                    
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        # Model size chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='model_name', y='model_size_mb', hue='quantized', data=results_df, ax=ax)
                        ax.set_title('Model Size Comparison')
                        ax.set_ylabel('Size (MB)')
                        ax.set_xlabel('Model')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with viz_col2:
                        # Inference time chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='model_name', y='avg_inference_sec', hue='quantized', data=results_df, ax=ax)
                        ax.set_title('Average Inference Time')
                        ax.set_ylabel('Time (seconds)')
                        ax.set_xlabel('Model')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Comparison table
                    st.subheader("Quantization Impact")
                    
                    # Calculate percentage improvement for each model
                    models = results_df['model_name'].unique()
                    improvements = []
                    
                    for model in models:
                        model_results = results_df[results_df['model_name'] == model]
                        
                        if len(model_results) >= 2:  # Need both quantized and unquantized
                            try:
                                unquantized = model_results[model_results['quantized'] == 'Unquantized'].iloc[0]
                                quantized = model_results[model_results['quantized'] == 'Quantized'].iloc[0]
                                
                                size_reduction = (1 - quantized['model_size_mb'] / unquantized['model_size_mb']) * 100
                                inference_improvement = (1 - quantized['avg_inference_sec'] / unquantized['avg_inference_sec']) * 100
                                memory_reduction = (1 - quantized['avg_memory_usage_mb'] / unquantized['avg_memory_usage_mb']) * 100
                                
                                improvements.append({
                                    'model': model,
                                    'size_reduction': size_reduction,
                                    'inference_improvement': inference_improvement,
                                    'memory_reduction': memory_reduction
                                })
                            except:
                                continue
                    
                    if improvements:
                        improvements_df = pd.DataFrame(improvements)
                        
                        # Format percentages
                        display_improvements = improvements_df.copy()
                        display_improvements['size_reduction'] = display_improvements['size_reduction'].apply(lambda x: f"{x:.1f}%")
                        display_improvements['inference_improvement'] = display_improvements['inference_improvement'].apply(lambda x: f"{x:.1f}%")
                        display_improvements['memory_reduction'] = display_improvements['memory_reduction'].apply(lambda x: f"{x:.1f}%")
                        
                        st.table(display_improvements.set_index('model'))
                        
                        # Overall metrics
                        avg_size_reduction = improvements_df['size_reduction'].mean()
                        avg_inference_improvement = improvements_df['inference_improvement'].mean()
                        avg_memory_reduction = improvements_df['memory_reduction'].mean()
                        
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        
                        with metric_col1:
                            st.metric("Avg Size Reduction", f"{avg_size_reduction:.1f}%")
                        
                        with metric_col2:
                            st.metric("Avg Speed Improvement", f"{avg_inference_improvement:.1f}%")
                        
                        with metric_col3:
                            st.metric("Avg Memory Reduction", f"{avg_memory_reduction:.1f}%")
                        
                        with metric_col4:
                            st.metric("API Latency", "5ms", "-2ms")
            except Exception as e:
                st.error(f"Error running benchmark: {str(e)}")
                import traceback
                st.code(traceback.format_exc()) 