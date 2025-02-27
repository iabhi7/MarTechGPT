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

class MarketingChatbot:
    def __init__(self, 
                knowledge_base_path: Optional[str] = None,
                model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
                netcore_api_key: Optional[str] = None):
        """
        Initialize the Marketing Chatbot.
        
        Args:
            knowledge_base_path: Path to knowledge base files (CSV or TXT)
            model_name: HuggingFace model to use
            netcore_api_key: API key for Netcore integration
        """
        self.model_name = model_name
        self.netcore_api_key = netcore_api_key
        self.knowledge_base_path = knowledge_base_path
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        print("Loading the language model...")
        self._setup_llm()
        
        print("Preparing the knowledge base...")
        self._setup_knowledge_base()
        
        print("Setting up the conversational chain...")
        self._setup_chain()
        
        print("Marketing chatbot is ready!")
        
    def _setup_llm(self):
        """Set up the language model for text generation"""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True,  # For memory efficiency
        )
        
        # Create text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
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