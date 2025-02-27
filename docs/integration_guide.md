# Netcore Integration Guide

This guide explains how to integrate the Netcore AI Marketing Suite with your existing Netcore Cloud implementation.

## Prerequisites

- An active Netcore Cloud account
- API key with appropriate permissions
- The Netcore AI Marketing Suite installed (see [Installation Guide](installation.md))

## Setting Up the Integration

### 1. Obtain Netcore API Credentials

1. Log in to your Netcore Cloud dashboard
2. Navigate to Settings > API Management
3. Generate a new API key with the following permissions:
   - Campaign data (read/write)
   - Customer data (read/write)
   - Content management (read/write)
4. Copy the generated API key

### 2. Configure Environment Variables

Add your Netcore API key to the `.env` file: 

```
NETCORE_API_KEY=your_netcore_api_key_here
```

### 3. Test the Connection

Run the connection test script to verify your integration:

```python
from utils.netcore_integration import NetcoreIntegration

netcore = NetcoreIntegration()
if netcore.test_connection():
    print("✅ Connection successful!")
else:
    print("❌ Connection failed. Check your API key and try again.")
```

## Data Flow

The Netcore AI Marketing Suite interacts with Netcore Cloud in the following ways:

### Inbound Data (Netcore → Suite)

- **Campaign data**: Historical performance metrics used for optimization
- **Customer data**: User profiles and behavior for segmentation and personalization
- **Email interaction data**: Opens, clicks, and conversions for subject line optimization

### Outbound Data (Suite → Netcore)

- **Optimized content**: AI-generated email subjects, ad copy, and marketing content
- **Customer segments**: ML-derived customer segments for targeted campaigns
- **Campaign recommendations**: Optimized workflow configurations
- **Churn predictions**: High-risk customers for retention campaigns

## Integration Scenarios

### 1. Email Subject Line Optimization

```python
from quick_wins.subject_line_optimizer.main import SubjectLineOptimizer
from utils.netcore_integration import NetcoreIntegration

# Setup components
netcore = NetcoreIntegration()
optimizer = SubjectLineOptimizer()

# Get historical data from Netcore
campaign_data = netcore.fetch_campaign_data(
    start_date="2023-01-01", 
    end_date="2023-06-30"
)

# Train optimizer with real Netcore data
optimizer.load_historical_data(data=campaign_data)

# Generate subject lines
subject_lines = optimizer.generate_subject_lines(
    product_info={"name": "Product Name", "features": ["Feature 1", "Feature 2"]},
    target_audience="Target Audience",
    campaign_type="promotional"
)

# Upload suggestions to Netcore
netcore.upload_content_suggestions(
    content_type="email_subject", 
    content_data=[{
        "subject_lines": [s["subject_line"] for s in subject_lines],
        "predicted_metrics": [{"open_rate": s["predicted_open_rate"]} for s in subject_lines]
    }]
)
```

### 2. Customer Segmentation and Targeting

```python
from quick_wins.customer_segmentation.segment_analyzer import CustomerSegmentAnalyzer
from utils.netcore_integration import NetcoreIntegration

# Setup components
netcore = NetcoreIntegration()
segmenter = CustomerSegmentAnalyzer()

# Fetch customer data from Netcore
customer_data = netcore.fetch_customer_data(limit=10000)

# Create segments
segmenter.load_data(data=customer_data)
segmenter.preprocess()
segmenter.create_segments()

# Upload segments to Netcore
segments_df = segmenter.segments[['customer_id', 'segment_name']]
netcore.upload_customer_segments(segments_df)
```

### 3. Churn Prevention Campaign

```python
from advanced_features.predictive_analytics.churn_predictor import CustomerChurnPredictor
from utils.netcore_integration import NetcoreIntegration

# Setup components
netcore = NetcoreIntegration()
predictor = CustomerChurnPredictor()

# Fetch customer data from Netcore
customer_data = netcore.fetch_customer_data(limit=5000)

# Train model and predict churn
predictor.load_customer_data(data=customer_data)
predictor.train_model()

# Export high-risk customers
high_risk = predictor.export_high_risk_customers(
    risk_level='medium',
    include_recommendations=True
)

# Upload churn predictions to Netcore
netcore.upload_churn_predictions(
    predictions=high_risk,
    include_recommendations=True
)
```

## API Endpoints

When integrating with Netcore, you'll be interacting with the following endpoints:

| Endpoint | Purpose |
|----------|---------|
| `/campaigns/performance` | Fetch campaign performance data |
| `/customers` | Fetch customer data |
| `/segments/batch` | Upload customer segments |
| `/content/suggestions` | Upload AI-generated content |
| `/campaigns/recommendations` | Upload campaign workflow recommendations |
| `/predictive/churn` | Upload churn predictions |

## Best Practices

1. **Rate Limiting**: Respect Netcore API rate limits by implementing retry mechanisms
2. **Data Privacy**: Only fetch and process the data you need
3. **Incremental Processing**: For large datasets, process data in batches
4. **Error Handling**: Implement robust error handling to manage API failures
5. **Logging**: Maintain detailed logs of all API interactions

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Authentication errors | Verify API key is correct and has required permissions |
| Timeout errors | Reduce batch size or implement pagination |
| Data format errors | Ensure data conforms to Netcore's expected format |
| Rate limit exceeded | Implement exponential backoff and retry logic |

For additional support, contact Netcore Cloud support or consult the [API documentation](https://docs.netcorecloud.com/api/).
