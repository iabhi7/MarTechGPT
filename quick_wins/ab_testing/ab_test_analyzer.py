import pandas as pd
import numpy as np
from textstat import flesch_reading_ease, flesch_kincaid_grade
import re
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class ABTestAnalyzer:
    """Analyze and compare A/B test variants for marketing content"""
    
    def __init__(self, nlp_model="en_core_web_sm"):
        """Initialize the A/B test analyzer"""
        # Load SpaCy model for NLP analysis
        try:
            self.nlp = spacy.load(nlp_model)
        except:
            # If model not found, download a minimal English model
            print(f"Downloading spaCy model: {nlp_model}")
            spacy.cli.download(nlp_model)
            self.nlp = spacy.load(nlp_model)
    
    def analyze_variants(self, variants):
        """
        Analyze multiple variants of marketing copy
        
        Args:
            variants: List of text variants to analyze
            
        Returns:
            DataFrame with analysis metrics for each variant
        """
        results = []
        
        for i, variant in enumerate(variants, 1):
            analysis = self.analyze_text(variant)
            analysis['variant_id'] = f"Variant {i}"
            analysis['text'] = variant
            results.append(analysis)
        
        return pd.DataFrame(results)
    
    def analyze_text(self, text):
        """
        Analyze a single piece of marketing copy
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of analysis metrics
        """
        # Process with SpaCy
        doc = self.nlp(text)
        
        # Basic text statistics
        word_count = len([token for token in doc if not token.is_punct and not token.is_space])
        sentence_count = len(list(doc.sents))
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Readability scores
        try:
            f_ease = flesch_reading_ease(text)
            fk_grade = flesch_kincaid_grade(text)
        except:
            f_ease = 0
            fk_grade = 0
        
        # Word usage analysis
        word_freq = Counter([token.text.lower() for token in doc if not token.is_punct and not token.is_space and not token.is_stop])
        top_words = [word for word, count in word_freq.most_common(5)]
        
        # Sentiment analysis (basic)
        sentiment = sum([token.sentiment for token in doc if hasattr(token, 'sentiment')])
        
        # Call-to-action detection
        cta_phrases = ["buy now", "sign up", "learn more", "get started", "try free", "contact us", "subscribe"]
        cta_count = sum([1 for phrase in cta_phrases if phrase in text.lower()])
        has_cta = cta_count > 0
        
        # Question detection
        question_count = len([sent for sent in doc.sents if sent.text.strip().endswith('?')])
        
        # Power words detection
        power_words = ["exclusive", "guaranteed", "limited", "free", "instant", "new", "proven", 
                      "save", "best", "easy", "discover", "amazing", "secret", "revolutionary"]
        power_word_count = sum([1 for word in power_words if re.search(r'\b' + word + r'\b', text.lower())])
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'readability_score': f_ease,
            'grade_level': fk_grade,
            'top_words': ', '.join(top_words),
            'sentiment_score': sentiment,
            'has_cta': has_cta,
            'cta_count': cta_count,
            'question_count': question_count,
            'power_word_count': power_word_count
        }
    
    def recommend_variant(self, analysis_df, primary_metric='readability_score', audience_type='general'):
        """
        Recommend the best variant based on analysis and audience type
        
        Args:
            analysis_df: DataFrame from analyze_variants
            primary_metric: Metric to prioritize in recommendation
            audience_type: Type of audience (general, technical, executive)
            
        Returns:
            Dictionary with recommendation and reasoning
        """
        # Create a scoring system based on audience type
        scores = pd.DataFrame()
        scores['variant_id'] = analysis_df['variant_id']
        
        # Different scoring models for different audiences
        if audience_type == 'technical':
            # Technical audiences can handle more complex content
            scores['score'] = (
                analysis_df['readability_score'] * 0.3 +  # Lower weight on simplicity
                analysis_df['power_word_count'] * 5 +     # Some power words
                analysis_df['has_cta'].astype(int) * 10   # CTAs important
            )
            ideal_metrics = "higher grade level, relevant technical terms, clear CTA"
            
        elif audience_type == 'executive':
            # Executives prefer concise, benefit-focused content
            scores['score'] = (
                (100 - analysis_df['word_count']) * 0.2 +  # Brevity is valued
                analysis_df['power_word_count'] * 8 +      # Impact words
                analysis_df['has_cta'].astype(int) * 15    # Strong CTA
            )
            ideal_metrics = "concise messaging, benefit-focused, strong CTA"
            
        else:  # 'general'
            # General audience prefers readable, engaging content
            scores['score'] = (
                analysis_df['readability_score'] * 0.5 +   # Readability important
                analysis_df['power_word_count'] * 6 +      # Engaging power words
                analysis_df['has_cta'].astype(int) * 10 +  # Clear CTA
                analysis_df['question_count'] * 5          # Questions engage readers
            )
            ideal_metrics = "high readability, engaging language, clear CTA"
        
        # Get the best variant
        best_variant_idx = scores['score'].idxmax()
        best_variant_id = scores.loc[best_variant_idx, 'variant_id']
        best_variant_text = analysis_df.loc[best_variant_idx, 'text']
        best_variant_metrics = {col: analysis_df.loc[best_variant_idx, col] for col in 
                              ['word_count', 'readability_score', 'power_word_count', 'has_cta']}
        
        recommendation = {
            'recommended_variant': best_variant_id,
            'text': best_variant_text,
            'audience_type': audience_type,
            'metrics': best_variant_metrics,
            'reasoning': f"This variant scores best for {audience_type} audiences, with {ideal_metrics}.",
            'all_scores': scores.to_dict(orient='records')
        }
        
        return recommendation
    
    def visualize_comparison(self, analysis_df, output_file=None):
        """
        Create visualization comparing variants
        
        Args:
            analysis_df: DataFrame from analyze_variants
            output_file: Optional file path to save visualization
            
        Returns:
            Matplotlib figure or saves to file
        """
        # Create a multi-panel figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('A/B Test Variant Comparison', fontsize=16)
        
        # Readability comparison
        sns.barplot(x='variant_id', y='readability_score', data=analysis_df, ax=axes[0, 0], palette='viridis')
        axes[0, 0].set_title('Readability Score (higher is better)')
        axes[0, 0].set_ylim(0, 100)
        
        # Word count comparison
        sns.barplot(x='variant_id', y='word_count', data=analysis_df, ax=axes[0, 1], palette='viridis')
        axes[0, 1].set_title('Word Count')
        
        # Power words
        sns.barplot(x='variant_id', y='power_word_count', data=analysis_df, ax=axes[1, 0], palette='viridis')
        axes[1, 0].set_title('Power Word Count')
        
        # Sentence length
        sns.barplot(x='variant_id', y='avg_sentence_length', data=analysis_df, ax=axes[1, 1], palette='viridis')
        axes[1, 1].set_title('Average Sentence Length')
        
        plt.tight_layout()
        
        # Save if output file specified
        if output_file:
            plt.savefig(output_file)
            return f"Visualization saved to {output_file}"
        
        return fig

# Example usage
if __name__ == "__main__":
    analyzer = ABTestAnalyzer()
    
    # Example variants for testing
    test_variants = [
        "Transform your marketing with our AI platform. Get started today and see 30% better results!",
        "Why struggle with outdated marketing tools? Our AI platform delivers 30% better results. Try it free.",
        "EXCLUSIVE OFFER: Revolutionary AI marketing platform guarantees 30% improvement. Limited time free trial!"
    ]
    
    # Analyze variants
    analysis = analyzer.analyze_variants(test_variants)
    print(analysis[['variant_id', 'word_count', 'readability_score', 'has_cta']])
    
    # Get recommendation for different audience types
    for audience in ['general', 'technical', 'executive']:
        recommendation = analyzer.recommend_variant(analysis, audience_type=audience)
        print(f"\nFor {audience} audience:")
        print(f"Recommended: {recommendation['recommended_variant']}")
        print(f"Reasoning: {recommendation['reasoning']}")
    
    # Visualize
    analyzer.visualize_comparison(analysis, "ab_test_comparison.png") 