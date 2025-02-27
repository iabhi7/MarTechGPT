import json
import pandas as pd
import numpy as np
import uuid
import datetime
import random
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from quick_wins.marketing_chatbot.chatbot import MarketingChatbot

class NetcoreCDPIntegration:
    """
    Mock integration with Netcore CDP (Customer Data Platform)
    
    This class simulates how the AI Marketing Suite would integrate with
    Netcore's existing customer data platform, enhancing it with AI capabilities.
    """
    
    def __init__(self, data_dir="netcore_data", chatbot=None):
        """Initialize the CDP integration"""
        self.data_dir = data_dir
        Path(data_dir).mkdir(exist_ok=True)
        
        # Initialize or load user data
        self.user_data_file = f"{data_dir}/user_profiles.json"
        self.interaction_data_file = f"{data_dir}/user_interactions.json"
        self.campaign_data_file = f"{data_dir}/campaigns.json"
        
        self.user_profiles = self._load_or_create(self.user_data_file)
        self.user_interactions = self._load_or_create(self.interaction_data_file)
        self.campaigns = self._load_or_create(self.campaign_data_file)
        
        # Initialize or load the chatbot
        if chatbot:
            self.chatbot = chatbot
        else:
            try:
                self.chatbot = MarketingChatbot(model_name="distilgpt2", quantize=True)
            except:
                self.chatbot = None
                print("Warning: Could not initialize chatbot. Some AI functions will be disabled.")
    
    def _load_or_create(self, file_path):
        """Load data from file or create empty dataset"""
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return []
    
    def _save_data(self, data, file_path):
        """Save data to file"""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def generate_mock_data(self, num_users=100, num_interactions=500, num_campaigns=10):
        """Generate mock data for demonstration purposes"""
        # Clear existing data
        self.user_profiles = []
        self.user_interactions = []
        self.campaigns = []
        
        # Generate user profiles
        industries = ["Retail", "Technology", "Finance", "Healthcare", "Education", "Manufacturing"]
        segments = ["High Value", "Mid Value", "Low Value", "New Customer", "Churned", "At Risk"]
        
        for i in range(num_users):
            user = {
                "user_id": str(uuid.uuid4()),
                "email": f"user{i}@example.com",
                "name": f"User {i}",
                "age": random.randint(25, 65),
                "industry": random.choice(industries),
                "segment": random.choice(segments),
                "acquisition_date": (datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 365))).isoformat(),
                "ltv": round(random.uniform(50, 5000), 2),
                "engagement_score": round(random.uniform(0, 100), 1)
            }
            self.user_profiles.append(user)
        
        # Generate campaigns
        campaign_types = ["Email", "SMS", "Push Notification", "In-app", "Social"]
        for i in range(num_campaigns):
            start_date = datetime.datetime.now() - datetime.timedelta(days=random.randint(10, 100))
            campaign = {
                "campaign_id": str(uuid.uuid4()),
                "name": f"Campaign {i}",
                "type": random.choice(campaign_types),
                "segment": random.choice(segments),
                "start_date": start_date.isoformat(),
                "end_date": (start_date + datetime.timedelta(days=random.randint(7, 30))).isoformat(),
                "content": {
                    "subject": f"Subject line for campaign {i}",
                    "body": f"This is the body content for campaign {i}. It contains marketing messages."
                },
                "metrics": {
                    "sent": random.randint(1000, 10000),
                    "delivered": 0,
                    "opened": 0,
                    "clicked": 0,
                    "converted": 0
                }
            }
            # Calculate realistic metrics
            campaign["metrics"]["delivered"] = int(campaign["metrics"]["sent"] * random.uniform(0.92, 0.99))
            campaign["metrics"]["opened"] = int(campaign["metrics"]["delivered"] * random.uniform(0.15, 0.35))
            campaign["metrics"]["clicked"] = int(campaign["metrics"]["opened"] * random.uniform(0.08, 0.25))
            campaign["metrics"]["converted"] = int(campaign["metrics"]["clicked"] * random.uniform(0.05, 0.20))
            
            self.campaigns.append(campaign)
        
        # Generate user interactions
        interaction_types = ["Email Open", "Email Click", "Website Visit", "Product View", "Cart Add", "Purchase", "Support Request"]
        for i in range(num_interactions):
            # Select a random user
            user = random.choice(self.user_profiles)
            
            # Generate interaction
            interaction_date = datetime.datetime.now() - datetime.timedelta(days=random.randint(0, 90), 
                                                                       hours=random.randint(0, 23),
                                                                       minutes=random.randint(0, 59))
            interaction_type = random.choice(interaction_types)
            
            # Create details based on interaction type
            details = {}
            if interaction_type == "Email Open" or interaction_type == "Email Click":
                campaign = random.choice(self.campaigns)
                details = {
                    "campaign_id": campaign["campaign_id"],
                    "campaign_name": campaign["name"],
                    "email_subject": campaign["content"]["subject"]
                }
            elif interaction_type == "Website Visit":
                pages = ["home", "products", "pricing", "blog", "about", "contact"]
                details = {
                    "page": random.choice(pages),
                    "referrer": random.choice(["google", "facebook", "direct", "email", "twitter"]),
                    "time_on_page": random.randint(5, 300)
                }
            elif interaction_type == "Product View" or interaction_type == "Cart Add":
                products = ["Product A", "Product B", "Product C", "Product D"]
                details = {
                    "product": random.choice(products),
                    "category": random.choice(["category1", "category2", "category3"]),
                    "price": round(random.uniform(9.99, 499.99), 2)
                }
            elif interaction_type == "Purchase":
                products = ["Product A", "Product B", "Product C", "Product D"]
                details = {
                    "order_id": str(uuid.uuid4()),
                    "products": random.sample(products, random.randint(1, 3)),
                    "total": round(random.uniform(9.99, 999.99), 2),
                    "payment_method": random.choice(["credit_card", "paypal", "bank_transfer"])
                }
            
            interaction = {
                "interaction_id": str(uuid.uuid4()),
                "user_id": user["user_id"],
                "type": interaction_type,
                "timestamp": interaction_date.isoformat(),
                "details": details
            }
            
            self.user_interactions.append(interaction)
        
        # Save all data
        self._save_data(self.user_profiles, self.user_data_file)
        self._save_data(self.user_interactions, self.interaction_data_file)
        self._save_data(self.campaigns, self.campaign_data_file)
        
        return {
            "users": len(self.user_profiles),
            "interactions": len(self.user_interactions),
            "campaigns": len(self.campaigns)
        }
    
    def enhance_user_profiles_with_ai(self):
        """Use AI to enhance user profiles with predicted interests and behaviors"""
        if not self.chatbot:
            return "Error: Chatbot not initialized"
        
        for user in self.user_profiles:
            # Get user interactions
            user_interactions = [i for i in self.user_interactions if i["user_id"] == user["user_id"]]
            
            # Skip if not enough interactions
            if len(user_interactions) < 3:
                continue
            
            # Extract data for AI analysis
            interaction_types = [i["type"] for i in user_interactions]
            interaction_details = [i["details"] for i in user_interactions]
            
            # Create a summary for the AI
            summary = f"Customer profile: {user['industry']} industry, age {user['age']}, {user['segment']} segment. "
            summary += f"Recent activities: {', '.join(interaction_types[:5])}. "
            
            # Use AI to predict interests and next best action
            prompt = f"Based on this customer data, predict top 3 interests and best marketing approach: {summary}"
            ai_analysis = self.chatbot.get_response(prompt)
            
            # Extract insights (in real CDP integration, we'd use more structured approach)
            user["ai_enhanced"] = {
                "predicted_interests": ai_analysis.split("interests")[1].split(".")[0] if "interests" in ai_analysis else "Unknown",
                "marketing_recommendation": ai_analysis.split("approach")[1].split(".")[0] if "approach" in ai_analysis else "Unknown",
                "analysis_date": datetime.datetime.now().isoformat()
            }
        
        # Save enhanced profiles
        self._save_data(self.user_profiles, self.user_data_file)
        
        return {
            "enhanced_profiles": sum(1 for user in self.user_profiles if "ai_enhanced" in user),
            "total_profiles": len(self.user_profiles)
        }
    
    def generate_personalized_campaign(self, segment, campaign_type="Email"):
        """Generate AI-personalized campaign for a specific segment"""
        if not self.chatbot:
            return "Error: Chatbot not initialized"
        
        # Get users in segment
        segment_users = [user for user in self.user_profiles if user["segment"] == segment]
        
        if not segment_users:
            return f"No users found in segment: {segment}"
        
        # Analyze segment characteristics
        industries = [user["industry"] for user in segment_users]
        industry_counts = pd.Series(industries).value_counts()
        primary_industry = industry_counts.index[0] if len(industry_counts) > 0 else "Various"
        
        avg_age = sum(user["age"] for user in segment_users) / len(segment_users)
        avg_ltv = sum(user["ltv"] for user in segment_users) / len(segment_users)
        
        # Create prompt for campaign generation
        prompt = f"Create an {campaign_type} campaign for customers in the {segment} segment. "
        prompt += f"They are primarily in the {primary_industry} industry with average age {avg_age:.1f} "
        prompt += f"and average lifetime value ${avg_ltv:.2f}. "
        prompt += f"Generate a subject line and brief email body."
        
        # Generate campaign content
        campaign_content = self.chatbot.get_response(prompt)
        
        # Extract subject line and body (simple parsing - would be more robust in production)
        subject_line = ""
        body = ""
        
        if "subject:" in campaign_content.lower():
            parts = campaign_content.split("Subject:", 1)
            parts = parts[1].split("\n", 1)
            subject_line = parts[0].strip()
            body = parts[1].strip() if len(parts) > 1 else ""
        else:
            # Fallback parsing
            lines = campaign_content.split("\n")
            subject_line = lines[0]
            body = "\n".join(lines[1:])
        
        # Create campaign
        campaign_id = str(uuid.uuid4())
        campaign = {
            "campaign_id": campaign_id,
            "name": f"AI Campaign for {segment}",
            "type": campaign_type,
            "segment": segment,
            "start_date": datetime.datetime.now().isoformat(),
            "end_date": (datetime.datetime.now() + datetime.timedelta(days=14)).isoformat(),
            "content": {
                "subject": subject_line,
                "body": body
            },
            "ai_generated": True,
            "target_audience": {
                "segment": segment,
                "primary_industry": primary_industry,
                "avg_age": avg_age,
                "avg_ltv": avg_ltv,
                "user_count": len(segment_users)
            },
            "metrics": {
                "sent": 0,
                "delivered": 0,
                "opened": 0,
                "clicked": 0,
                "converted": 0
            }
        }
        
        # Save to campaigns
        self.campaigns.append(campaign)
        self._save_data(self.campaigns, self.campaign_data_file)
        
        return campaign
    
    def analyze_campaign_performance(self):
        """Analyze campaign performance with AI insights"""
        if not self.campaigns:
            return "No campaigns to analyze"
        
        # Calculate performance metrics
        campaign_metrics = []
        for campaign in self.campaigns:
            metrics = campaign["metrics"]
            
            # Calculate rates
            sent = metrics["sent"]
            if sent == 0:
                continue  # Skip campaigns with no sends
                
            delivered_rate = metrics["delivered"] / sent if sent > 0 else 0
            open_rate = metrics["opened"] / metrics["delivered"] if metrics["delivered"] > 0 else 0
            click_rate = metrics["clicked"] / metrics["opened"] if metrics["opened"] > 0 else 0
            conversion_rate = metrics["converted"] / metrics["clicked"] if metrics["clicked"] > 0 else 0
            
            campaign_metrics.append({
                "campaign_id": campaign["campaign_id"],
                "name": campaign["name"],
                "type": campaign["type"],
                "segment": campaign["segment"],
                "ai_generated": campaign.get("ai_generated", False),
                "sent": sent,
                "delivered_rate": delivered_rate,
                "open_rate": open_rate,
                "click_rate": click_rate,
                "conversion_rate": conversion_rate
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(campaign_metrics)
        
        # Compare AI-generated vs manual campaigns
        if "ai_generated" in df.columns and df["ai_generated"].any():
            ai_performance = df[df["ai_generated"]].mean()
            non_ai_performance = df[~df["ai_generated"]].mean()
            
            performance_comparison = {
                "ai_campaigns": {
                    "count": df[df["ai_generated"]].shape[0],
                    "avg_open_rate": ai_performance["open_rate"],
                    "avg_click_rate": ai_performance["click_rate"],
                    "avg_conversion_rate": ai_performance["conversion_rate"]
                },
                "standard_campaigns": {
                    "count": df[~df["ai_generated"]].shape[0],
                    "avg_open_rate": non_ai_performance["open_rate"],
                    "avg_click_rate": non_ai_performance["click_rate"],
                    "avg_conversion_rate": non_ai_performance["conversion_rate"]
                }
            }
        else:
            performance_comparison = "No AI-generated campaigns to compare"
        
        # Top performing campaigns
        if not df.empty:
            top_by_open = df.sort_values("open_rate", ascending=False).head(3)[["name", "segment", "open_rate"]]
            top_by_conversion = df.sort_values("conversion_rate", ascending=False).head(3)[["name", "segment", "conversion_rate"]]
        else:
            top_by_open = "No campaign data"
            top_by_conversion = "No campaign data"
        
        # Segment performance analysis
        segment_performance = df.groupby("segment")[["open_rate", "click_rate", "conversion_rate"]].mean() if not df.empty else "No campaign data"
        
        results = {
            "campaign_count": len(campaign_metrics),
            "performance_comparison": performance_comparison,
            "top_by_open_rate": top_by_open.to_dict() if isinstance(top_by_open, pd.DataFrame) else top_by_open,
            "top_by_conversion": top_by_conversion.to_dict() if isinstance(top_by_conversion, pd.DataFrame) else top_by_conversion,
            "segment_performance": segment_performance.to_dict() if isinstance(segment_performance, pd.DataFrame) else segment_performance
        }
        
        return results
    
    def run_sample_netcore_integration(self):
        """Run a complete sample workflow to demonstrate integration"""
        results = []
        
        # 1. Generate mock data
        results.append({
            "step": "Generate Mock Data",
            "result": self.generate_mock_data(num_users=50, num_interactions=200, num_campaigns=5)
        })
        
        # 2. Enhance user profiles with AI
        results.append({
            "step": "Enhance User Profiles with AI",
            "result": self.enhance_user_profiles_with_ai()
        })
        
        # 3. Generate AI campaign for a segment
        segment = "High Value"
        results.append({
            "step": f"Generate AI Campaign for {segment} Segment",
            "result": {
                "campaign_generated": bool(self.generate_personalized_campaign(segment)),
                "segment": segment
            }
        })
        
        # 4. Analyze campaign performance
        results.append({
            "step": "Analyze Campaign Performance",
            "result": self.analyze_campaign_performance()
        })
        
        return results

# Example usage
if __name__ == "__main__":
    integration = NetcoreCDPIntegration()
    workflow_results = integration.run_sample_netcore_integration()
    
    for step in workflow_results:
        print(f"\n=== {step['step']} ===")
        print(json.dumps(step['result'], indent=2)) 