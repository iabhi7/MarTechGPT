# API Documentation

The Netcore AI Marketing Suite provides several API endpoints for integrating the AI capabilities into your applications.

## Authentication

All API requests require authentication using an API key:

```
Authorization: Bearer YOUR_API_KEY
```

## Base URL

```
https://your-deployment-url.com/api/v1
```

## Endpoints

### Subject Line Generator

#### Generate Subject Lines

```
POST /subject-lines/generate
```

Generates optimized email subject lines based on input parameters.

**Request Body:**

```json
{
  "product_info": {
    "name": "Smart Fitness Watch",
    "category": "Wearable Tech",
    "features": ["Heart rate monitoring", "Sleep tracking", "7-day battery"]
  },
  "target_audience": "Fitness enthusiasts aged 25-45",
  "campaign_type": "promotional",
  "num_suggestions": 5
}
```

**Response:**

```json
{
  "status": "success",
  "subject_lines": [
    {
      "subject_line": "Track Your Sleep, Optimize Your Fitness: Smart Watch Sale 🔥",
      "predicted_open_rate": 0.32,
      "predicted_click_rate": 0.14,
      "score": 0.85
    },
    {
      "subject_line": "JUST LAUNCHED: The Watch That Monitors Your Heart 24/7",
      "predicted_open_rate": 0.29,
      "predicted_click_rate": 0.13,
      "score": 0.79
    }
  ]
}
```

### Customer Segmentation

#### Create Segments

```
POST /segments/create
```

Creates customer segments based on behavioral data.

**Request Body:**

```json
{
  "customers": [
    {
      "customer_id": "CUST12345",
      "recency": 5,
      "frequency": 12,
      "monetary": 950.00,
      "email_engagement": 0.68,
      "product_categories": ["electronics", "accessories"]
    }
  ],
  "n_segments": 4
}
```

**Response:**

```json
{
  "status": "success",
  "segments": [
    {
      "segment_id": 0,
      "segment_name": "high_value",
      "segment_size": 0.12,
      "customer_ids": ["CUST12345", "CUST67890"]
    }
  ]
}
```

## Error Responses

All endpoints return standard error responses in this format:

```json
{
  "status": "error",
  "error_code": "invalid_input",
  "message": "Required field 'product_info' is missing"
}
```

## Rate Limits

- 100 requests per minute per API key
- 5,000 requests per day per API key 