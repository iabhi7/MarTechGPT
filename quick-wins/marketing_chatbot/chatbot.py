import os
from typing import Dict, List, Optional, Any
import json
import pandas as pd
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, CSVLoader
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketingChatbot:
    def __init__(self, 
                knowledge_base_path: Optional[str] = None,
                model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
                netcore_api_key: Optional[str] = None,
                quantize: bool = True):
        """
        Initialize the Marketing Chatbot.
        
        Args:
            knowledge_base_path: Path to knowledge base files (CSV or TXT)
            model_name: HuggingFace model to use
            netcore_api_key: API key for Netcore integration
            quantize: Whether to apply quantization to reduce model size
        """
        self.model_name = model_name
        self.netcore_api_key = netcore_api_key
        self.knowledge_base_path = knowledge_base_path
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        print("Loading the language model...")
        self._setup_llm(quantize)
        
        print("Preparing the knowledge base...")
        self._setup_knowledge_base()
        
        print("Setting up the conversational chain...")
        self._setup_chain()
        
        print("Marketing chatbot is ready!")
        
    def _setup_llm(self, quantize: bool):
        """Set up the language model for text generation"""
        start_time = time.time()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model with quantization if enabled
        if quantize:
            logger.info(f"Loading model {self.model_name} with 8-bit quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                load_in_8bit=True,  # Enable 8-bit quantization
                torch_dtype=torch.float16  # Use half precision
            )
        else:
            logger.info(f"Loading model {self.model_name} without quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        
        # Calculate and log memory usage
        memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
        logger.info(f"Memory usage: {memory_used:.2f} GB")
        
        self.model_size = self._calculate_model_size()
        logger.info(f"Model size: {self.model_size:.2f} MB")
        
        # Create text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        # Create LangChain wrapper for the pipeline
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
    def _setup_knowledge_base(self):
        """Set up the knowledge base for retrieval"""
        if self.knowledge_base_path and os.path.exists(self.knowledge_base_path):
            # Load documents from knowledge base
            if self.knowledge_base_path.endswith('.csv'):
                loader = CSVLoader(self.knowledge_base_path)
            else:
                loader = TextLoader(self.knowledge_base_path)
                
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Create vector store
            self.vector_store = FAISS.from_documents(chunks, embeddings)
            
        else:
            # Create a simple sample knowledge base
            sample_data = """
            Q: What is Netcore Cloud?
            A: Netcore Cloud is a global MarTech product company that helps B2C brands create amazing digital experiences with a range of products that help in acquisition, engagement, retention, and analytics.
            
            Q: What is email deliverability?
            A: Email deliverability refers to the ability to deliver emails to subscribers' inboxes. It's affected by factors like sender reputation, email content, and technical setup.
            
            Q: How does Netcore improve email open rates?
            A: Netcore improves email open rates through AI-powered send time optimization, subject line recommendations, content personalization, and maintaining high deliverability standards.
            
            Q: What is a Customer Data Platform (CDP)?
            A: A Customer Data Platform (CDP) is a unified customer database that collects, organizes, and activates customer data from multiple sources to create a single customer view for personalized marketing.
            
            Q: How does AI help in customer segmentation?
            A: AI helps in customer segmentation by analyzing large datasets to identify patterns and behaviors, enabling more precise targeting based on predicted actions rather than just demographic information.
            """
            
            # Split into documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = text_splitter.split_text(sample_data)
            
            # Create embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Create vector store
            self.vector_store = FAISS.from_texts(chunks, embeddings)
            
    def _setup_chain(self):
        """Set up the conversational chain for chatbot interactions"""
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            return_source_documents=True
        )
        
    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user message and return a response.
        
        Args:
            user_message: The user's message or question
            
        Returns:
            Dictionary containing response and metadata
        """
        # Enhance the user message with marketing context
        enhanced_message = f"""
        As a marketing AI assistant for Netcore Cloud, please help with: {user_message}
        
        Provide information about marketing best practices, campaign optimization, 
        or Netcore's specific features if relevant. If unsure, suggest relevant 
        resources or next steps.
        """
        
        # Get response from the conversational chain
        response = self.chain({"question": enhanced_message})
        
        # Format the output
        result = {
            "user_message": user_message,
            "response": response["answer"],
            "source_documents": [doc.page_content for doc in response.get("source_documents", [])]
        }
        
        # Optional: If integrated with Netcore, log the conversation
        if self.netcore_api_key:
            self._log_to_netcore(user_message, response["answer"])
            
        return result
    
    def _log_to_netcore(self, user_message: str, bot_response: str):
        """
        Log conversation to Netcore for analytics (example integration).
        
        Args:
            user_message: User's question
            bot_response: Bot's response
        """
        # This is a placeholder for Netcore API integration
        # In a real implementation, you would use Netcore's API to log the interaction
        print(f"Logging conversation to Netcore: Q: {user_message[:30]}... A: {bot_response[:30]}...")
        
    def clear_history(self):
        """Reset the conversation history"""
        self.memory.clear()
        print("Conversation history has been cleared.")

    def _calculate_model_size(self):
        """Calculate the model size in MB"""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
    
    def get_response(self, user_message, context=None):
        """
        Generate a response to the user's message
        
        Args:
            user_message (str): The user's input message
            context (dict, optional): Additional context for personalization
        
        Returns:
            str: The chatbot's response
        """
        # Add context to the prompt if available
        prompt = user_message
        if context:
            user_name = context.get('user_name', '')
            interests = context.get('interests', [])
            if user_name or interests:
                context_prefix = f"[Context: User: {user_name}, Interests: {', '.join(interests)}]\n"
                prompt = context_prefix + prompt
        
        # Record inference start time
        start_time = time.time()
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # Decode and return response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the response (remove the input prompt)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        # Log inference time
        inference_time = time.time() - start_time
        logger.info(f"Generated response in {inference_time:.2f} seconds")
        
        return response
    
    def generate_ad_copy(self, product_name, target_audience, key_benefits):
        """
        Generate ad copy for a specific product and target audience
        
        Args:
            product_name (str): Name of the product
            target_audience (str): Description of the target audience
            key_benefits (list): List of key benefits of the product
        
        Returns:
            str: Generated ad copy
        """
        benefits_text = "\n".join([f"- {benefit}" for benefit in key_benefits])
        prompt = f"""Generate compelling ad copy for:
Product: {product_name}
Target Audience: {target_audience}
Key Benefits:
{benefits_text}

Create a short, engaging ad with a strong call-to-action:"""
        
        return self.get_response(prompt)

    def generate_ab_test_variants(self, product_name, target_audience, key_message, num_variants=2):
        """
        Generate multiple ad variants for A/B testing
        
        Args:
            product_name (str): Name of the product
            target_audience (str): Description of the target audience
            key_message (str): Main message to convey
            num_variants (int): Number of variants to generate
        
        Returns:
            list: List of ad variants
        """
        prompt = f"""Generate {num_variants} different ad copy variants for A/B testing:
Product: {product_name}
Target Audience: {target_audience}
Key Message: {key_message}

Create {num_variants} distinct ad variants with different approaches but the same core message:"""
        
        response = self.get_response(prompt)
        
        # Simple parsing to extract variants (actual implementation might need more robust parsing)
        variants = []
        for line in response.split('\n'):
            if line.strip().startswith(('Variant', 'Option', '#')):
                variants.append(line)
        
        return variants if variants else [response]
    
    def get_performance_metrics(self):
        """Return performance metrics of the model"""
        return {
            "model_name": self.model_name,
            "model_size_mb": self.model_size,
            "quantized": hasattr(self.model, "is_quantized") and self.model.is_quantized,
            "device": str(next(self.model.parameters()).device)
        }

# Example usage
if __name__ == "__main__":
    # Initialize the chatbot
    chatbot = MarketingChatbot()
    
    # Example interaction
    queries = [
        "How can I improve my email open rates?",
        "Tell me about customer segmentation",
        "What's the best way to create personalized campaigns?",
        "How does Netcore's AI technology work?",
    ]
    
    for query in queries:
        print(f"\nUser: {query}")
        response = chatbot.chat(query)
        print(f"Bot: {response['response']}") 