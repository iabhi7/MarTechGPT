from flask import Flask, request, jsonify
from chatbot import MarketingChatbot
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize chatbot
chatbot = MarketingChatbot()

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint for health checks"""
    return jsonify({"status": "healthy", "service": "marketing-chatbot-api"})

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint for chatbot interactions"""
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400
        
        user_message = data['message']
        context = data.get('context', {})
        
        # Log incoming requests (excluding sensitive data)
        logger.info(f"Received chat request. Length: {len(user_message)}")
        
        # Get response from chatbot
        response = chatbot.get_response(user_message, context)
        
        return jsonify({
            "response": response,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/generate_ad', methods=['POST'])
def generate_ad():
    """Endpoint for generating ad copy"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Missing request body"}), 400
        
        product_name = data.get('product_name', '')
        target_audience = data.get('target_audience', '')
        key_benefits = data.get('key_benefits', [])
        
        # Generate ad copy
        ad_copy = chatbot.generate_ad_copy(product_name, target_audience, key_benefits)
        
        return jsonify({
            "ad_copy": ad_copy,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error generating ad: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) 