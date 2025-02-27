from fastapi import FastAPI, HTTPException, Depends, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import os
import sys
import datetime
import uuid

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from quick_wins.marketing_chatbot.chatbot import MarketingChatbot
from quick_wins.ab_testing.ab_test_analyzer import ABTestAnalyzer

# Models
class ChatRequest(BaseModel):
    message: str = Field(..., example="How can I improve my email open rates?")
    context: Dict[str, Any] = Field(default={}, example={"user_id": "12345", "previous_campaigns": ["Spring Sale", "Summer Launch"]})

class ChatResponse(BaseModel):
    response: str
    status: str = "success"
    request_id: str
    timestamp: str

class AdCopyRequest(BaseModel):
    product_name: str = Field(..., example="Cloud Storage Service")
    target_audience: str = Field(..., example="Small business owners")
    key_benefits: List[str] = Field(..., example=["Secure", "Affordable", "Easy to use"])
    tone: Optional[str] = Field(default="professional", example="friendly")
    max_length: Optional[int] = Field(default=300, example=200)

class AdCopyResponse(BaseModel):
    ad_copy: str
    status: str = "success"
    request_id: str
    metrics: Dict[str, Any]
    timestamp: str

class ABTestRequest(BaseModel):
    product_name: str = Field(..., example="Marketing Analytics Platform")
    target_audience: str = Field(..., example="Digital marketers and marketing directors")
    key_message: str = Field(..., example="Increase conversion rates with AI-powered insights")
    num_variants: Optional[int] = Field(default=3, example=3, ge=2, le=5)
    audience_type: Optional[str] = Field(default="general", example="executive")

class ABTestResponse(BaseModel):
    variants: List[str]
    recommended_variant: int
    analysis: Dict[str, Any]
    status: str = "success"
    request_id: str
    timestamp: str

class ErrorResponse(BaseModel):
    error: str
    status: str = "error"
    request_id: str
    timestamp: str

# Initialize FastAPI
app = FastAPI(
    title="AI Marketing Suite API",
    description="API for AI-powered marketing tools",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
chatbot = None
ab_tester = None

def get_chatbot():
    """Get or initialize the chatbot"""
    global chatbot
    if chatbot is None:
        chatbot = MarketingChatbot(model_name=os.environ.get("MODEL_NAME", "distilgpt2"), 
                                    quantize=os.environ.get("QUANTIZE", "True").lower() == "true")
    return chatbot

def get_ab_tester():
    """Get or initialize the A/B test analyzer"""
    global ab_tester
    if ab_tester is None:
        ab_tester = ABTestAnalyzer()
    return ab_tester

# Routes
@app.get("/health", summary="Health check endpoint")
async def health_check():
    """Check if the API is running correctly"""
    return {
        "status": "healthy",
        "service": "marketing-api",
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.post("/chat", 
         response_model=ChatResponse,
         responses={
             200: {"description": "Successful response"},
             400: {"model": ErrorResponse, "description": "Bad request"},
             500: {"model": ErrorResponse, "description": "Server error"}
         },
         summary="Chat with the marketing assistant",
         tags=["Marketing Chatbot"])
async def chat(request: ChatRequest):
    """
    Chat with the AI marketing assistant.
    
    - **message**: The user's message or question
    - **context**: Optional context about the user or conversation
    """
    try:
        bot = get_chatbot()
        response = bot.get_response(request.message, context=request.context)
        
        return {
            "response": response,
            "status": "success",
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "status": "error",
                "request_id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now().isoformat()
            }
        )

@app.post("/generate_ad", 
         response_model=AdCopyResponse,
         responses={
             200: {"description": "Successful response"},
             400: {"model": ErrorResponse, "description": "Bad request"},
             500: {"model": ErrorResponse, "description": "Server error"}
         },
         summary="Generate marketing ad copy",
         tags=["Ad Generation"])
async def generate_ad(request: AdCopyRequest):
    """
    Generate marketing ad copy for a product or service.
    
    - **product_name**: Name of the product or service
    - **target_audience**: Description of the target audience
    - **key_benefits**: List of key benefits or selling points
    - **tone**: Optional tone for the ad (professional, friendly, formal, etc.)
    - **max_length**: Optional maximum length in characters
    """
    try:
        bot = get_chatbot()
        ad_copy = bot.generate_ad_copy(
            product_name=request.product_name,
            target_audience=request.target_audience,
            key_benefits=request.key_benefits,
            tone=request.tone,
            max_length=request.max_length
        )
        
        # Calculate basic metrics
        metrics = {
            "character_count": len(ad_copy),
            "word_count": len(ad_copy.split()),
            "reading_time_sec": len(ad_copy.split()) / 3,  # Approx. reading time
            "has_key_benefits": all(benefit.lower() in ad_copy.lower() for benefit in request.key_benefits),
        }
        
        return {
            "ad_copy": ad_copy,
            "status": "success",
            "request_id": str(uuid.uuid4()),
            "metrics": metrics,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "status": "error",
                "request_id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now().isoformat()
            }
        )

@app.post("/ab_test", 
         response_model=ABTestResponse,
         responses={
             200: {"description": "Successful response"},
             400: {"model": ErrorResponse, "description": "Bad request"},
             500: {"model": ErrorResponse, "description": "Server error"}
         },
         summary="Generate and analyze A/B test variants",
         tags=["A/B Testing"])
async def generate_ab_test(request: ABTestRequest):
    """
    Generate multiple variants for A/B testing and analyze which one is likely to perform best.
    
    - **product_name**: Name of the product or service
    - **target_audience**: Description of the target audience
    - **key_message**: Main message or value proposition
    - **num_variants**: Number of variants to generate (2-5)
    - **audience_type**: Type of audience for optimization (general, technical, executive)
    """
    try:
        bot = get_chatbot()
        ab_tester = get_ab_tester()
        
        # Generate variants
        variants = bot.generate_ab_test_variants(
            product_name=request.product_name,
            target_audience=request.target_audience,
            key_message=request.key_message,
            num_variants=request.num_variants
        )
        
        # Analyze variants
        analysis_df = ab_tester.analyze_variants(variants)
        
        # Get recommendation
        recommendation = ab_tester.recommend_variant(analysis_df, request.audience_type)
        
        # Format for response
        # Get the index of the recommended variant
        recommended_idx = int(recommendation['recommended_variant'].split()[-1]) - 1
        
        # Convert analysis to serializable format
        analysis_dict = analysis_df.to_dict(orient='records')
        
        return {
            "variants": variants,
            "recommended_variant": recommended_idx,
            "analysis": {
                "recommendation": recommendation,
                "metrics": analysis_dict
            },
            "status": "success",
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "status": "error",
                "request_id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now().isoformat()
            }
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 