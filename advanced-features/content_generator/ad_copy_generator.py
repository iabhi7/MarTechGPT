import os
import re
from typing import List, Dict, Any, Optional, Union
import json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AdCopyGenerator:
    """
    AI-powered ad copy generator.
    Uses machine learning to generate and optimize advertising copy
    across different channels and formats.
    """

    def __init__(self, 
                 model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", 
                 api_key: Optional[str] = None,
                 device: str = "auto"):
        """
        Initialize the Ad Copy Generator.
        
        Args:
            model_name: Name of the pretrained model to use
            api_key: API key for Netcore integration
            device: Device to use for inference ("cpu", "cuda", "auto")
        """
        self.model_name = model_name
        self.api_key = api_key
        
        print(f"Loading model {model_name}...")
        self._initialize_model(model_name, device)
        
        # Load brand voice examples if provided
        self.brand_voice_examples = []
        
        # Performance tracking
        self.generation_history = []
        
        print("Ad Copy Generator initialized successfully!")
        
    def _initialize_model(self, model_name: str, device: str):
        """
        Initialize the language model for text generation.
        
        Args:
            model_name: Name of the pretrained model
            device: Device to use for inference
        """
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Initialize model with efficient settings for inference
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device if device != "auto" else "auto",
                torch_dtype=torch.float16,
                load_in_8bit=True  # For memory efficiency
            )
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15
            )
        except Exception as e:
            print(f"Error initializing model: {e}")
            print("Falling back to smaller model...")
            self.generator = pipeline(
                "text-generation",
                model="gpt2"
            )
            
    def set_brand_voice(self, examples: List[str]):
        """
        Set brand voice examples to guide the tone and style of generated content.
        
        Args:
            examples: List of example ad copies in the brand's voice
        """
        self.brand_voice_examples = examples
        print(f"Set {len(examples)} brand voice examples")
        
    def generate_email_copy(self, 
                           product_info: Dict[str, Any],
                           campaign_type: str,
                           target_audience: str,
                           key_message: str,
                           email_sections: List[str] = ["subject_line", "headline", "body", "cta"],
                           tone: str = "friendly",
                           max_length: int = 400) -> Dict[str, str]:
        """
        Generate email marketing copy.
        
        Args:
            product_info: Dictionary with product details
            campaign_type: Type of email campaign (promotional, newsletter, etc.)
            target_audience: Description of the target audience
            key_message: Main message to convey
            email_sections: List of sections to generate
            tone: Tone of the copy (friendly, professional, urgent, etc.)
            max_length: Maximum length of the generated body text
            
        Returns:
            Dictionary with generated email copy sections
        """
        # Extract key information
        product_name = product_info.get('name', 'our product')
        product_features = product_info.get('features', [])
        product_category = product_info.get('category', '')
        
        # Construct prompt for each section
        result = {}
        
        # Build brand voice context
        brand_context = ""
        if self.brand_voice_examples:
            brand_context = "Use these examples to match our brand voice:\n"
            for i, example in enumerate(self.brand_voice_examples[:3]):
                brand_context += f"Example {i+1}: {example}\n"
        
        for section in email_sections:
            if section == "subject_line":
                prompt = f"""
                Generate 3 engaging email subject lines for a {campaign_type} campaign for {product_name}.
                Target audience: {target_audience}
                Key message: {key_message}
                Tone: {tone}
                {brand_context}
                
                Keep them under 50 characters, compelling, and likely to drive high open rates.
                Format: 1. [subject line 1], 2. [subject line 2], 3. [subject line 3]
                """
                
            elif section == "headline":
                prompt = f"""
                Write a compelling headline for an email about {product_name} in a {campaign_type} campaign.
                Target audience: {target_audience}
                Key message: {key_message}
                Tone: {tone}
                {brand_context}
                
                Make it attention-grabbing and clear. Keep it under 80 characters.
                """
                
            elif section == "body":
                features_text = ""
                if product_features:
                    features_text = "Key features:\n" + "\n".join([f"- {feature}" for feature in product_features])
                
                prompt = f"""
                Write a persuasive email body for a {campaign_type} campaign featuring {product_name}.
                Target audience: {target_audience}
                Key message: {key_message}
                {features_text}
                Tone: {tone}
                {brand_context}
                
                Focus on benefits rather than features. Include personalization elements.
                Keep it concise and engaging, under {max_length} characters.
                """
                
            elif section == "cta":
                prompt = f"""
                Create a clear, compelling call-to-action button text for an email about {product_name}.
                Campaign type: {campaign_type}
                Key message: {key_message}
                Tone: {tone}
                
                Make it action-oriented, specific, and create urgency. Keep it under 30 characters.
                """
                
            # Generate the content
            generated = self.generator(prompt, max_new_tokens=512)
            response_text = generated[0]['generated_text'] if isinstance(generated, list) else generated['generated_text']
            
            # Extract the generated content from the response
            if section == "subject_line":
                # Extract numbered list items
                subject_lines = re.findall(r'\d+\.\s+\[?(.*?)\]?(?=\d+\.|$)', response_text)
                if not subject_lines:
                    subject_lines = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|$)', response_text)
                
                # Clean up and limit to 3
                subject_lines = [line.strip().strip('[]') for line in subject_lines if line.strip()][:3]
                result[section] = subject_lines
            else:
                # Extract the content after the prompt
                content = response_text.split(prompt)[-1].strip()
                
                # Remove any extra instructions or formatting
                content = re.sub(r'^[:\-\s]+', '', content)
                content = re.sub(r'^\[|\]$', '', content)
                
                result[section] = content
                
        # Track this generation for performance analysis
        self.generation_history.append({
            'product': product_name,
            'campaign_type': campaign_type,
            'target_audience': target_audience,
            'sections_generated': email_sections,
            'timestamp': pd.Timestamp.now()
        })
                
        return result
    
    def generate_social_media_posts(self,
                                  product_info: Dict[str, Any],
                                  platform: str,
                                  campaign_objective: str,
                                  num_posts: int = 3,
                                  include_hashtags: bool = True) -> List[Dict[str, str]]:
        """
        Generate social media post content.
        
        Args:
            product_info: Dictionary with product details
            platform: Social media platform (instagram, facebook, twitter, linkedin)
            campaign_objective: Objective (awareness, engagement, conversion)
            num_posts: Number of post variations to generate
            include_hashtags: Whether to include hashtags
            
        Returns:
            List of dictionaries with post content and metadata
        """
        # Extract key information
        product_name = product_info.get('name', 'our product')
        product_features = product_info.get('features', [])
        product_category = product_info.get('category', '')
        
        # Define platform-specific constraints
        platform_constraints = {
            'twitter': {'max_length': 280, 'format': 'concise', 'hashtags': 2},
            'instagram': {'max_length': 500, 'format': 'visual-focused', 'hashtags': 5},
            'facebook': {'max_length': 500, 'format': 'conversational', 'hashtags': 3},
            'linkedin': {'max_length': 700, 'format': 'professional', 'hashtags': 3}
        }
        
        # Get constraints for the selected platform (or use defaults)
        constraints = platform_constraints.get(platform.lower(), {'max_length': 500, 'format': 'general', 'hashtags': 3})
        
        # Build brand voice context
        brand_context = ""
        if self.brand_voice_examples:
            brand_context = "Use these examples to match our brand voice:\n"
            for i, example in enumerate(self.brand_voice_examples[:2]):
                brand_context += f"Example {i+1}: {example}\n"
        
        # Construct prompt
        features_text = ""
        if product_features:
            features_text = "Key features:\n" + "\n".join([f"- {feature}" for feature in product_features[:3]])
            
        prompt = f"""
        Generate {num_posts} engaging {platform} posts for {product_name} ({product_category}).
        Campaign objective: {campaign_objective}
        {features_text}
        {brand_context}
        
        Format: {constraints['format']}
        Maximum length: {constraints['max_length']} characters
        
        Write {num_posts} different posts, numbered 1-{num_posts}.
        {f"Include relevant hashtags (max {constraints['hashtags']})." if include_hashtags else "Do not include hashtags."}
        
        Each post should have a clear message, be engaging, and aligned with {platform}'s best practices.
        """
        
        # Generate
        generated = self.generator(prompt, max_new_tokens=1024)
        response_text = generated[0]['generated_text'] if isinstance(generated, list) else generated['generated_text']
        
        # Extract the generated posts
        # Find the numbered posts in the response
        posts_text = response_text.split(prompt)[-1].strip()
        
        # Try to extract numbered posts
        posts = []
        post_matches = re.findall(r'(?:^|\n)\s*(\d+)[\.:\)]\s*(.*?)(?=(?:\n\s*\d+[\.:\)])|$)', posts_text, re.DOTALL)
        
        for _, content in post_matches:
            post_content = content.strip()
            
            # Extract hashtags if present
            hashtags = []
            if include_hashtags:
                hashtag_matches = re.findall(r'(#\w+)', post_content)
                hashtags = hashtag_matches
                
            posts.append({
                'content': post_content,
                'platform': platform,
                'hashtags': hashtags,
                'objective': campaign_objective,
                'character_count': len(post_content)
            })
        
        # If no posts were extracted, split by newlines and try to get the posts
        if not posts:
            lines = posts_text.split('\n')
            current_post = ""
            for line in lines:
                if re.match(r'^\s*\d+[\.:\)]', line):  # New post starts
                    if current_post:  # Save previous post if exists
                        posts.append({
                            'content': current_post.strip(),
                            'platform': platform,
                            'hashtags': re.findall(r'(#\w+)', current_post),
                            'objective': campaign_objective,
                            'character_count': len(current_post.strip())
                        })
                    current_post = re.sub(r'^\s*\d+[\.:\)]\s*', '', line)
                else:
                    current_post += " " + line.strip()
            
            # Add the last post
            if current_post:
                posts.append({
                    'content': current_post.strip(),
                    'platform': platform,
                    'hashtags': re.findall(r'(#\w+)', current_post),
                    'objective': campaign_objective,
                    'character_count': len(current_post.strip())
                })
        
        # Limit to requested number of posts
        posts = posts[:num_posts]
        
        # Track this generation
        self.generation_history.append({
            'product': product_name,
            'platform': platform,
            'objective': campaign_objective,
            'posts_generated': len(posts),
            'timestamp': pd.Timestamp.now()
        })
        
        return posts
    
    def generate_ad_copy(self,
                        product_info: Dict[str, Any],
                        ad_platform: str,
                        ad_objective: str,
                        target_audience: str,
                        num_variations: int = 3,
                        max_headline_length: int = 30,
                        max_description_length: int = 90) -> List[Dict[str, str]]:
        """
        Generate ad copy for digital advertising platforms.
        
        Args:
            product_info: Dictionary with product details
            ad_platform: Advertising platform (google_ads, facebook_ads, etc.)
            ad_objective: Campaign objective (conversions, awareness, etc.)
            target_audience: Description of the target audience
            num_variations: Number of ad variations to generate
            max_headline_length: Maximum headline length
            max_description_length: Maximum description length
            
        Returns:
            List of dictionaries with ad copy variations
        """
        # Extract key information
        product_name = product_info.get('name', 'our product')
        product_features = product_info.get('features', [])
        product_category = product_info.get('category', '')
        
        # Adjust constraints based on platform
        if ad_platform.lower() == 'google_ads':
            max_headline_length = min(max_headline_length, 30)
            max_description_length = min(max_description_length, 90)
        elif ad_platform.lower() == 'facebook_ads':
            max_headline_length = min(max_headline_length, 40)
            max_description_length = min(max_description_length, 125)
        
        # Build brand voice context
        brand_context = ""
        if self.brand_voice_examples:
            brand_context = "Use these examples to match our brand voice:\n"
            for i, example in enumerate(self.brand_voice_examples[:2]):
                brand_context += f"Example {i+1}: {example}\n"
        
        # Construct prompt
        features_text = ""
        if product_features:
            features_text = "Key features:\n" + "\n".join([f"- {feature}" for feature in product_features[:3]])
            
        prompt = f"""
        Generate {num_variations} engaging ad copy variations for {product_name} on {ad_platform}.
        Campaign objective: {ad_objective}
        Target audience: {target_audience}
        {features_text}
        {brand_context}
        
        Each ad should include:
        1. Headline (max {max_headline_length} characters)
        2. Description (max {max_description_length} characters)
        3. Call to action
        
        Format your response as:
        
        Variation 1:
        Headline: [headline text]
        Description: [description text]
        CTA: [call to action text]
        
        Variation 2:
        ...
        
        Focus on benefits over features, create a sense of urgency or exclusivity,
        and make sure the copy is aligned with the campaign objective.
        """
        
        # Generate
        generated = self.generator(prompt, max_new_tokens=1024)
        response_text = generated[0]['generated_text'] if isinstance(generated, list) else generated['generated_text']
        
        # Extract the generated ad variations
        variations_text = response_text.split(prompt)[-1].strip()
        
        # Extract variations
        variations = []
        variation_blocks = re.split(r'\n+\s*Variation \d+:\s*\n+', variations_text)
        
        # Process each variation block
        for block in variation_blocks:
            if not block.strip():
                continue
                
            # Extract components
            headline_match = re.search(r'Headline:\s*(.*?)(?:\n|$)', block)
            description_match = re.search(r'Description:\s*(.*?)(?:\n|$)', block)
            cta_match = re.search(r'CTA:\s*(.*?)(?:\n|$)', block)
            
            headline = headline_match.group(1).strip() if headline_match else ""
            description = description_match.group(1).strip() if description_match else ""
            cta = cta_match.group(1).strip() if cta_match else ""
            
            if headline or description:  # At least one component should be present
                variations.append({
                    'headline': headline,
                    'description': description,
                    'cta': cta,
                    'platform': ad_platform,
                    'objective': ad_objective
                })
        
        # If no variations were extracted using the pattern, try a simpler approach
        if not variations:
            lines = variations_text.split('\n')
            current_variation = {}
            
            for line in lines:
                if "Headline:" in line:
                    # Save previous variation if it exists
                    if current_variation and ('headline' in current_variation or 'description' in current_variation):
                        variations.append(current_variation.copy())
                        current_variation = {}
                    
                    current_variation['headline'] = line.split("Headline:")[-1].strip()
                elif "Description:" in line:
                    current_variation['description'] = line.split("Description:")[-1].strip()
                elif "CTA:" in line:
                    current_variation['cta'] = line.split("CTA:")[-1].strip()
            
            # Don't forget to add the last variation
            if current_variation and ('headline' in current_variation or 'description' in current_variation):
                variations.append(current_variation.copy())
        
        # Add platform and objective to any variations that don't have them
        for var in variations:
            var.setdefault('platform', ad_platform)
            var.setdefault('objective', ad_objective)
        
        # Limit to requested number of variations
        variations = variations[:num_variations]
        
        # Track this generation
        self.generation_history.append({
            'product': product_name,
            'platform': ad_platform,
            'objective': ad_objective,
            'variations_generated': len(variations),
            'timestamp': pd.Timestamp.now()
        })
        
        return variations
    
    def generate_content_report(self) -> Dict[str, Any]:
        """
        Generate a report on content creation activity.
        
        Returns:
            Dictionary with content generation statistics
        """
        if not self.generation_history:
            return {"message": "No content has been generated yet."}
        
        # Convert history to DataFrame for analysis
        history_df = pd.DataFrame(self.generation_history)
        
        # Calculate statistics
        total_generations = len(history_df)
        generations_by_type = {}
        
        if 'sections_generated' in history_df.columns:
            email_generations = history_df[history_df['sections_generated'].notnull()].shape[0]
            generations_by_type['email'] = email_generations
            
        if 'posts_generated' in history_df.columns:
            social_generations = history_df[history_df['posts_generated'].notnull()].shape[0]
            generations_by_type['social'] = social_generations
            
        if 'variations_generated' in history_df.columns:
            ad_generations = history_df[history_df['variations_generated'].notnull()].shape[0]
            generations_by_type['ad'] = ad_generations
        
        # Create report
        report = {
            'total_generations': total_generations,
            'generations_by_type': generations_by_type,
            'recent_generations': history_df.tail(5).to_dict('records')
        }
        
        # Add platform-specific stats if available
        if 'platform' in history_df.columns:
            platform_counts = history_df['platform'].value_counts().to_dict()
            report['platform_distribution'] = platform_counts
            
        # Add objective stats if available
        if 'objective' in history_df.columns:
            objective_counts = history_df['objective'].value_counts().to_dict()
            report['objective_distribution'] = objective_counts
            
        return report

# Example usage
if __name__ == "__main__":
    # Initialize the generator
    generator = AdCopyGenerator()
    
    # Set brand voice examples
    brand_voice_examples = [
        "Our AI-powered platform helps marketers create personalized campaigns at scale, driving better engagement and conversions.",
        "Transform your customer experience with intelligent automation that thinks like a marketer but works at the speed of AI."
    ]
    generator.set_brand_voice(brand_voice_examples)
    
    # Generate email copy
    product_info = {
        "name": "Netcore Genware",
        "category": "AI Marketing Platform",
        "features": [
            "AI content generation",
            "Personalized email campaigns",
            "Customer segmentation",
            "Real-time analytics"
        ]
    }
    
    email_copy = generator.generate_email_copy(
        product_info=product_info,
        campaign_type="product launch",
        target_audience="Marketing directors at enterprise companies",
        key_message="Launch AI-powered campaigns in minutes, not weeks",
        tone="professional"
    )
    
    print("Generated Email Copy:")
    print(f"Subject Line Options: {email_copy['subject_line']}")
    print(f"Headline: {email_copy['headline']}")
    print(f"Body: {email_copy['body'][:100]}...")
    print(f"CTA: {email_copy['cta']}")
    
    # Generate social media posts
    social_posts = generator.generate_social_media_posts(
        product_info=product_info,
        platform="linkedin",
        campaign_objective="lead_generation",
        num_posts=2
    )
    
    print("\nGenerated LinkedIn Posts:")
    for i, post in enumerate(social_posts, 1):
        print(f"Post {i}: {post['content'][:100]}...")
        print(f"Hashtags: {', '.join(post['hashtags'])}")
        print()
    
    # Generate ad copy
    ad_variations = generator.generate_ad_copy(
        product_info=product_info,
        ad_platform="google_ads",
        ad_objective="conversions",
        target_audience="Marketing professionals",
        num_variations=2
    )
    
    print("\nGenerated Ad Variations:")
    for i, variation in enumerate(ad_variations, 1):
        print(f"Variation {i}:")
        print(f"Headline: {variation['headline']}")
        print(f"Description: {variation['description']}")
        print(f"CTA: {variation['cta']}")
        print() 