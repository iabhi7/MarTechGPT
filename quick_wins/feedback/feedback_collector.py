import json
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class FeedbackCollector:
    """Collect, store, and analyze user feedback on AI-generated content"""
    
    def __init__(self, feedback_dir="feedback_data"):
        """Initialize the feedback collector"""
        self.feedback_dir = feedback_dir
        Path(feedback_dir).mkdir(exist_ok=True)
        self.feedback_file = os.path.join(feedback_dir, "feedback_log.jsonl")
        
    def log_feedback(self, content_type, content, rating, comments=None, metadata=None):
        """
        Log user feedback about generated content
        
        Args:
            content_type: Type of content (e.g., 'ad_copy', 'chat_response', 'subject_line')
            content: The actual content that was rated
            rating: User rating (1-5 stars or thumbs up/down)
            comments: Optional user comments
            metadata: Additional metadata (e.g., user info, context)
        """
        feedback_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "content_type": content_type,
            "content": content,
            "rating": rating,
            "comments": comments,
            "metadata": metadata or {}
        }
        
        # Append to JSONL file
        with open(self.feedback_file, "a") as f:
            f.write(json.dumps(feedback_entry) + "\n")
        
        return True
    
    def get_feedback_data(self):
        """Load all feedback data into a pandas DataFrame"""
        if not os.path.exists(self.feedback_file):
            return pd.DataFrame()
        
        # Read JSONL file
        data = []
        with open(self.feedback_file, "r") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
        
        return pd.DataFrame(data)
    
    def generate_analytics_report(self, output_dir=None):
        """Generate analytics report on feedback data"""
        df = self.get_feedback_data()
        
        if df.empty:
            return "No feedback data available"
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Create output directory
        if output_dir:
            Path(output_dir).mkdir(exist_ok=True)
        
        # Basic statistics
        stats = {
            "total_feedback": len(df),
            "average_rating": df["rating"].mean(),
            "feedback_by_type": df["content_type"].value_counts().to_dict(),
            "rating_distribution": df["rating"].value_counts().sort_index().to_dict(),
            "feedback_over_time": df.groupby(df["timestamp"].dt.date)["rating"].mean().to_dict()
        }
        
        # Generate visualizations
        if output_dir:
            # Rating distribution
            plt.figure(figsize=(10, 6))
            sns.countplot(x="rating", data=df, palette="viridis")
            plt.title("Distribution of Ratings")
            plt.xlabel("Rating")
            plt.ylabel("Count")
            plt.savefig(os.path.join(output_dir, "rating_distribution.png"))
            
            # Average rating by content type
            plt.figure(figsize=(10, 6))
            sns.barplot(x="content_type", y="rating", data=df, palette="viridis")
            plt.title("Average Rating by Content Type")
            plt.xlabel("Content Type")
            plt.ylabel("Average Rating")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "rating_by_content_type.png"))
            
            # Ratings over time
            plt.figure(figsize=(12, 6))
            df_time = df.groupby(df["timestamp"].dt.date)["rating"].mean().reset_index()
            plt.plot(df_time["timestamp"], df_time["rating"], marker='o')
            plt.title("Average Rating Over Time")
            plt.xlabel("Date")
            plt.ylabel("Average Rating")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "rating_over_time.png"))
        
        # Save statistics to JSON
        if output_dir:
            with open(os.path.join(output_dir, "feedback_stats.json"), "w") as f:
                json.dump(stats, f, indent=2, default=str)
        
        return stats
    
    def get_improvement_suggestions(self):
        """Analyze feedback to suggest potential improvements"""
        df = self.get_feedback_data()
        
        if df.empty:
            return "No feedback data available"
        
        # Focus on low-rated content
        low_rated = df[df["rating"] <= 3]
        
        if low_rated.empty:
            return "No low-rated content to analyze"
        
        suggestions = {
            "content_types_to_improve": low_rated["content_type"].value_counts().to_dict(),
            "example_low_rated": low_rated.sort_values("rating").head(5)[["content_type", "content", "rating", "comments"]].to_dict(orient="records")
        }
        
        return suggestions

# Example usage
if __name__ == "__main__":
    collector = FeedbackCollector()
    
    # Log some example feedback
    collector.log_feedback(
        content_type="ad_copy",
        content="Transform your business with our revolutionary platform!",
        rating=4,
        comments="Good, but could be more specific",
        metadata={"industry": "tech", "target_audience": "SMBs"}
    )
    
    collector.log_feedback(
        content_type="chat_response",
        content="To improve email open rates, focus on compelling subject lines and proper list segmentation.",
        rating=5,
        comments="Very helpful and practical advice",
        metadata={"query": "How to improve email open rates"}
    )
    
    # Generate analytics
    stats = collector.generate_analytics_report("feedback_analytics")
    print(stats)
    
    # Get improvement suggestions
    suggestions = collector.get_improvement_suggestions()
    print(suggestions) 