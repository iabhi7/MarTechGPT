import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import matplotlib.pyplot as plt
import networkx as nx
import json
import os
from datetime import datetime, timedelta
import logging
import uuid
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('journey_orchestrator')

class JourneyOrchestrator:
    """
    AI-powered customer journey orchestration engine.
    Optimizes multi-channel marketing campaigns using machine learning.
    """
    
    def __init__(self, 
                journey_config_path: Optional[str] = None,
                use_predictive_optimization: bool = True,
                api_key: Optional[str] = None):
        """
        Initialize the Journey Orchestrator.
        
        Args:
            journey_config_path: Optional path to journey configuration file
            use_predictive_optimization: Whether to use ML for journey optimization
            api_key: Optional API key for services
        """
        self.journeys = {}
        self.active_customer_journeys = {}
        self.journey_analytics = {}
        self.channel_configs = self._default_channel_configs()
        self.use_predictive_optimization = use_predictive_optimization
        self.conversion_models = {}
        self.api_key = api_key
        
        # Load journey config if provided
        if journey_config_path and os.path.exists(journey_config_path):
            self.load_journeys(journey_config_path)
            
        logger.info(f"Journey Orchestrator initialized (predictive optimization: {use_predictive_optimization})")
    
    def _default_channel_configs(self) -> Dict[str, Dict[str, Any]]:
        """Create default channel configurations."""
        return {
            "email": {
                "send_delay": 0,  # immediate
                "throttle_period": 24,  # hours between messages
                "delivery_time_optimization": True,
                "max_frequency": 2,  # max emails per week
                "api_endpoint": "https://api.your_company.com/email/send"
            },
            "push": {
                "send_delay": 0,
                "throttle_period": 48,
                "delivery_time_optimization": True,
                "max_frequency": 3,
                "api_endpoint": "https://api.your_company.com/push/send"
            },
            "sms": {
                "send_delay": 0,
                "throttle_period": 72,
                "delivery_time_optimization": True,
                "max_frequency": 1,
                "api_endpoint": "https://api.your_company.com/sms/send"
            },
            "in_app": {
                "send_delay": 0,
                "throttle_period": 24,
                "delivery_time_optimization": False,
                "max_frequency": 5,
                "api_endpoint": "https://api.your_company.com/inapp/display"
            },
            "whatsapp": {
                "send_delay": 0,
                "throttle_period": 48,
                "delivery_time_optimization": True,
                "max_frequency": 2,
                "api_endpoint": "https://api.your_company.com/whatsapp/send"
            }
        }
    
    def create_journey(self, journey_name: str,
                      entry_condition: Dict[str, Any],
                      steps: List[Dict[str, Any]],
                      exit_condition: Optional[Dict[str, Any]] = None,
                      journey_goal: Optional[str] = "conversion",
                      audience_segment: Optional[str] = None,
                      description: Optional[str] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      max_duration_days: Optional[int] = 30) -> Dict[str, Any]:
        """
        Create a new customer journey.
        
        Args:
            journey_name: Name of the journey
            entry_condition: Condition for customers to enter the journey
            steps: List of journey steps with content, conditions, and channels
            exit_condition: Condition for customers to exit the journey
            journey_goal: Primary goal of the journey (conversion, engagement, etc.)
            audience_segment: Target audience for the journey
            description: Description of the journey
            start_date: When the journey should start
            end_date: When the journey should end
            max_duration_days: Maximum duration for an individual to be in the journey
            
        Returns:
            Dictionary with journey configuration
        """
        # Generate journey ID
        journey_id = str(uuid.uuid4())
        
        # Validate steps
        self._validate_journey_steps(steps)
        
        # Prepare step IDs and build the journey graph
        journey_graph = self._build_journey_graph(steps)
        
        # Create journey configuration
        journey_config = {
            'journey_id': journey_id,
            'journey_name': journey_name,
            'entry_condition': entry_condition,
            'exit_condition': exit_condition,
            'steps': steps,
            'journey_graph': journey_graph,
            'journey_goal': journey_goal,
            'audience_segment': audience_segment,
            'description': description,
            'status': 'created',
            'creation_date': datetime.now().isoformat(),
            'update_date': datetime.now().isoformat(),
            'start_date': start_date,
            'end_date': end_date,
            'max_duration_days': max_duration_days,
            'active_customers': 0,
            'completed_customers': 0,
            'conversion_rate': 0.0,
            'step_performance': {step['step_id']: {
                'impressions': 0,
                'conversions': 0,
                'conversion_rate': 0.0
            } for step in steps if 'step_id' in step}
        }
        
        # Store journey configuration
        self.journeys[journey_id] = journey_config
        
        # Initialize journey analytics
        self.journey_analytics[journey_id] = {
            'customer_progress': {},
            'step_performance': {},
            'conversion_paths': [],
            'entry_dates': {},
            'exit_dates': {},
            'goal_completion_dates': {}
        }
        
        logger.info(f"Created journey '{journey_name}' with ID {journey_id} and {len(steps)} steps")
        return journey_config
    
    def _validate_journey_steps(self, steps: List[Dict[str, Any]]) -> None:
        """Validate journey step configurations."""
        for i, step in enumerate(steps):
            # Ensure each step has an ID
            if 'step_id' not in step:
                step['step_id'] = f"step_{i}"
                
            # Ensure each step has a name
            if 'name' not in step:
                step['name'] = f"Step {i + 1}"
                
            # Validate channel
            if 'channel' in step and step['channel'] not in self.channel_configs:
                raise ValueError(f"Invalid channel '{step['channel']}' in step '{step['name']}'")
                
            # Ensure wait_time has proper format
            if 'wait_time' in step and not isinstance(step['wait_time'], (int, float)):
                raise ValueError(f"Wait time in step '{step['name']}' must be a number (hours)")
                
            # Ensure each conditional step has a condition
            if step.get('type') == 'conditional' and 'condition' not in step:
                raise ValueError(f"Conditional step '{step['name']}' missing condition")
                
            # Ensure each split step has branches
            if step.get('type') == 'split' and ('branches' not in step or len(step['branches']) < 2):
                raise ValueError(f"Split step '{step['name']}' must have at least 2 branches")
    
    def _build_journey_graph(self, steps: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Build a graph of step connections from the journey definition.
        
        Args:
            steps: List of journey step configurations
            
        Returns:
            Dictionary mapping step IDs to lists of next step IDs
        """
        graph = {}
        step_map = {step['step_id']: step for step in steps}
        
        # First pass: direct connections
        for step in steps:
            step_id = step['step_id']
            graph[step_id] = []
            
            # If the step has a next_step_id, add it to the graph
            if 'next_step_id' in step:
                next_step_id = step['next_step_id']
                
                if next_step_id in step_map:
                    graph[step_id].append(next_step_id)
                    
        # Second pass: handle branches and conditions
        for step in steps:
            step_id = step['step_id']
            
            # Handle branch steps
            if step.get('type') == 'branch':
                branches = step.get('branches', [])
                
                for branch in branches:
                    if 'next_step_id' in branch and branch['next_step_id'] in step_map:
                        if branch['next_step_id'] not in graph[step_id]:
                            graph[step_id].append(branch['next_step_id'])
                            
            # Handle condition steps
            elif step.get('type') == 'condition':
                for path in ['success_step_id', 'failure_step_id']:
                    if path in step and step[path] in step_map:
                        if step[path] not in graph[step_id]:
                            graph[step_id].append(step[path])
                            
        return graph
    
    def start_journey(self, journey_id: str) -> Dict[str, Any]:
        """
        Start a journey.
        
        Args:
            journey_id: ID of the journey to start
            
        Returns:
            Updated journey configuration
        """
        if journey_id not in self.journeys:
            raise ValueError(f"Journey ID {journey_id} not found")
            
        journey = self.journeys[journey_id]
        
        if journey['status'] != 'created':
            logger.warning(f"Journey '{journey['journey_name']}' is already {journey['status']}")
            return journey
            
        journey['status'] = 'active'
        journey['start_date'] = journey.get('start_date') or datetime.now().isoformat()
        journey['update_date'] = datetime.now().isoformat()
        
        logger.info(f"Started journey '{journey['journey_name']}' with ID {journey_id}")
        return journey
    
    def stop_journey(self, journey_id: str) -> Dict[str, Any]:
        """
        Stop a journey.
        
        Args:
            journey_id: ID of the journey to stop
            
        Returns:
            Updated journey configuration
        """
        if journey_id not in self.journeys:
            raise ValueError(f"Journey ID {journey_id} not found")
            
        journey = self.journeys[journey_id]
        
        if journey['status'] in ['completed', 'stopped']:
            logger.warning(f"Journey '{journey['journey_name']}' is already {journey['status']}")
            return journey
            
        journey['status'] = 'stopped'
        journey['update_date'] = datetime.now().isoformat()
        
        logger.info(f"Stopped journey '{journey['journey_name']}' with ID {journey_id}")
        return journey
    
    def add_customer_to_journey(self, journey_id: str, 
                              customer_id: str,
                              customer_data: Dict[str, Any],
                              start_step_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a customer to a journey.
        
        Args:
            journey_id: ID of the journey
            customer_id: ID of the customer
            customer_data: Customer data for personalization and targeting
            start_step_id: Optional specific step to start from
            
        Returns:
            Customer journey state
        """
        if journey_id not in self.journeys:
            raise ValueError(f"Journey ID {journey_id} not found")
            
        journey = self.journeys[journey_id]
        
        # Check if journey is active
        if journey.get('status') != 'active':
            logger.warning(f"Adding customer to inactive journey '{journey['journey_name']}'")
            
        # Check if journey has an entry condition and evaluate it
        entry_condition = journey.get('entry_condition')
        
        if entry_condition and not self._evaluate_condition(entry_condition, customer_data):
            logger.info(f"Customer {customer_id} does not meet entry condition for journey {journey_id}")
            return None
            
        # Get first step if start_step_id not provided
        if not start_step_id:
            # Find the first step
            steps = journey.get('steps', [])
            
            if not steps:
                logger.warning(f"Journey {journey_id} has no steps")
                return None
                
            start_step_id = steps[0].get('step_id')
            
        # Check if start step exists
        if not any(step.get('step_id') == start_step_id for step in journey.get('steps', [])):
            raise ValueError(f"Start step ID {start_step_id} not found in journey {journey_id}")
            
        # Create customer journey state
        customer_key = f"{journey_id}_{customer_id}"
        
        customer_journey = {
            'journey_id': journey_id,
            'customer_id': customer_id,
            'entry_date': datetime.now().isoformat(),
            'current_step_id': start_step_id,
            'next_step_id': None,
            'next_action_date': None,
            'steps_completed': [],
            'active': True,
            'completed': False,
            'goal_achieved': False,
            'customer_data': customer_data,
            'events': [],
            'messages_sent': {}  # Track message history by channel
        }
        
        # Store customer journey
        self.active_customer_journeys[customer_key] = customer_journey
        
        # Update journey metrics
        journey['active_customers'] += 1
        
        # Store entry date in analytics
        self.journey_analytics[journey_id]['entry_dates'][customer_id] = customer_journey['entry_date']
        
        logger.info(f"Added customer {customer_id} to journey '{journey['journey_name']}'")
        
        # Process the first step
        self._process_customer_step(journey_id, customer_id)
        
        return customer_journey
    
    def _find_first_step(self, steps: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the first step in a journey."""
        # Look for step with is_first_step flag
        for step in steps:
            if step.get('is_first_step', False):
                return step
                
        # If no explicit first step, use the first in the list
        if steps:
            return steps[0]
            
        return None
    
    def _evaluate_condition(self, condition: Dict[str, Any], 
                          customer_data: Dict[str, Any]) -> bool:
        """
        Evaluate a condition against customer data.
        
        Args:
            condition: Condition specification
            customer_data: Customer data to evaluate against
            
        Returns:
            True if condition is met, False otherwise
        """
        condition_type = condition.get('type', 'simple')
        
        if condition_type == 'simple':
            field = condition.get('field')
            operator = condition.get('operator', 'equals')
            value = condition.get('value')
            
            if field not in customer_data:
                return False
                
            customer_value = customer_data[field]
            
            if operator == 'equals':
                return customer_value == value
            elif operator == 'not_equals':
                return customer_value != value
            elif operator == 'contains':
                return value in customer_value
            elif operator == 'not_contains':
                return value not in customer_value
            elif operator == 'greater_than':
                return customer_value > value
            elif operator == 'less_than':
                return customer_value < value
            elif operator == 'in':
                return customer_value in value
            elif operator == 'not_in':
                return customer_value not in value
            else:
                logger.warning(f"Unknown operator '{operator}'")
                return False
                
        elif condition_type == 'and':
            subconditions = condition.get('conditions', [])
            return all(self._evaluate_condition(subcond, customer_data) for subcond in subconditions)
            
        elif condition_type == 'or':
            subconditions = condition.get('conditions', [])
            return any(self._evaluate_condition(subcond, customer_data) for subcond in subconditions)
            
        elif condition_type == 'not':
            subcondition = condition.get('condition', {})
            return not self._evaluate_condition(subcondition, customer_data)
            
        elif condition_type == 'predictive':
            # Use ML model to predict probability of meeting condition
            model_name = condition.get('model', 'conversion')
            threshold = condition.get('threshold', 0.5)
            
            if model_name not in self.conversion_models:
                logger.warning(f"Predictive model '{model_name}' not found")
                return False
                
            try:
                features = self._extract_model_features(customer_data)
                probability = self._predict_probability(model_name, features)
                
                return probability >= threshold
                
            except Exception as e:
                logger.error(f"Error evaluating predictive condition: {str(e)}")
                return False
                
        else:
            logger.warning(f"Unknown condition type '{condition_type}'")
            return False
    
    def _process_customer_step(self, journey_id: str, customer_id: str) -> None:
        """
        Process the current step for a customer in a journey.
        
        Args:
            journey_id: ID of the journey
            customer_id: ID of the customer
        """
        if journey_id not in self.journeys:
            logger.warning(f"Journey ID {journey_id} not found")
            return
            
        customer_key = f"{journey_id}_{customer_id}"
        
        if customer_key not in self.active_customer_journeys:
            logger.warning(f"Customer {customer_id} not found in journey {journey_id}")
            return
            
        customer_journey = self.active_customer_journeys[customer_key]
        
        if not customer_journey['active']:
            logger.info(f"Customer {customer_id} is no longer active in journey {journey_id}")
            return
            
        # Get the current step
        current_step_id = customer_journey['current_step_id']
        
        if not current_step_id:
            logger.warning(f"No current step defined for customer {customer_id} in journey {journey_id}")
            return
            
        # Find the step in the journey
        journey = self.journeys[journey_id]
        step = None
        
        for s in journey['steps']:
            if s['step_id'] == current_step_id:
                step = s
                break
                
        if not step:
            logger.warning(f"Step {current_step_id} not found in journey {journey_id}")
            return
            
        # Process step based on type
        step_type = step.get('type', 'message')
        
        if step_type == 'message':
            self._process_message_step(journey_id, customer_id, step)
            
        elif step_type == 'delay':
            self._process_delay_step(journey_id, customer_id, step)
            
        elif step_type == 'condition':
            self._process_condition_step(journey_id, customer_id, step)
            
        elif step_type == 'split':
            self._process_split_step(journey_id, customer_id, step)
            
        elif step_type == 'goal':
            self._process_goal_step(journey_id, customer_id, step)
            
        elif step_type == 'exit':
            self._process_exit_step(journey_id, customer_id, step)
            
        else:
            logger.warning(f"Unknown step type '{step_type}' for step {current_step_id}")
            
        # Mark step as completed
        if current_step_id not in customer_journey['steps_completed']:
            customer_journey['steps_completed'].append(current_step_id)
            
        # Update journey step analytics
        if current_step_id in journey['step_performance']:
            journey['step_performance'][current_step_id]['impressions'] += 1
        else:
            journey['step_performance'][current_step_id] = {
                'impressions': 1,
                'conversions': 0,
                'conversion_rate': 0.0
            }

    def _extract_model_features(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features for predictive models from customer data.
        
        Args:
            customer_data: Customer data dictionary
            
        Returns:
            Dictionary of features for model training/prediction
        """
        features = {}
        
        # Copy basic customer attributes
        for key in ['account_age_days', 'average_session_time', 'browse_count']:
            if key in customer_data:
                features[key] = customer_data[key]
            else:
                features[key] = 0
                
        # Extract device type
        if 'device_type' in customer_data:
            device_type = customer_data['device_type']
            features['device_mobile'] = 1 if device_type == 'mobile' else 0
            features['device_desktop'] = 1 if device_type == 'desktop' else 0
            features['device_tablet'] = 1 if device_type == 'tablet' else 0
        else:
            features['device_mobile'] = 0
            features['device_desktop'] = 0
            features['device_tablet'] = 0
            
        # Extract acquisition source
        if 'acquisition_source' in customer_data:
            source = customer_data['acquisition_source']
            features['source_app'] = 1 if 'app' in source else 0
            features['source_web'] = 1 if 'web' in source else 0
            features['source_social'] = 1 if 'social' in source else 0
            features['source_email'] = 1 if 'email' in source else 0
            features['source_search'] = 1 if 'search' in source else 0
        else:
            features['source_app'] = 0
            features['source_web'] = 0
            features['source_social'] = 0
            features['source_email'] = 0
            features['source_search'] = 0
            
        return features

    def _process_message_step(self, journey_id: str, customer_id: str, 
                           step: Dict[str, Any]) -> None:
        """Process a message step for a customer."""
        customer_key = f"{journey_id}_{customer_id}"
        customer_journey = self.active_customer_journeys[customer_key]
        
        # Get channel and content
        channel = step.get('channel')
        content = step.get('content', {})
        
        if not channel:
            logger.warning(f"No channel specified for message step {step['step_id']}")
            return
            
        # Personalize content
        personalized_content = self._personalize_content(content, customer_journey['customer_data'])
        
        # Send to channel
        response = self._send_to_channel(
            journey_id=journey_id,
            customer_id=customer_id,
            channel=channel,
            content=personalized_content,
            step_id=step['step_id']
        )
        
        # Record message sent
        if channel not in customer_journey['messages_sent']:
            customer_journey['messages_sent'][channel] = []
            
        message_record = {
            'step_id': step['step_id'],
            'timestamp': datetime.now().isoformat(),
            'content': personalized_content,
            'response': response
        }
        
        customer_journey['messages_sent'][channel].append(message_record)
        
        # Add event
        customer_journey['events'].append({
            'type': 'message_sent',
            'channel': channel,
            'step_id': step['step_id'],
            'timestamp': datetime.now().isoformat()
        })
        
        # Continue to next step if defined
        next_step_id = self._get_next_step_id(self.journeys[journey_id], step, customer_journey)
        
        if next_step_id:
            customer_journey['next_step_id'] = next_step_id

    def _process_delay_step(self, journey_id: str, customer_id: str, 
                         step: Dict[str, Any]) -> None:
        """Process a delay step for a customer."""
        customer_key = f"{journey_id}_{customer_id}"
        customer_journey = self.active_customer_journeys[customer_key]
        
        # Get wait time in hours
        wait_time = step.get('wait_time', 24)  # default to 24 hours
        
        # Calculate next action date
        next_action = datetime.now() + timedelta(hours=wait_time)
        customer_journey['next_action_date'] = next_action.isoformat()
        
        # Add event
        customer_journey['events'].append({
            'type': 'delay_started',
            'wait_time': wait_time,
            'step_id': step['step_id'],
            'timestamp': datetime.now().isoformat(),
            'next_action_date': customer_journey['next_action_date']
        })
        
        # Continue to next step if defined
        next_step_id = self._get_next_step_id(self.journeys[journey_id], step, customer_journey)
        
        if next_step_id:
            customer_journey['next_step_id'] = next_step_id
            
        logger.info(f"Customer {customer_id} waiting for {wait_time} hours in journey {journey_id}")

    def _process_condition_step(self, journey_id: str, customer_id: str, 
                             step: Dict[str, Any]) -> None:
        """Process a condition step for a customer."""
        customer_key = f"{journey_id}_{customer_id}"
        customer_journey = self.active_customer_journeys[customer_key]
        
        # Get condition
        condition = step.get('condition')
        
        if not condition:
            logger.warning(f"No condition specified for condition step {step['step_id']}")
            return
            
        # Evaluate condition
        success = self._evaluate_condition(condition, customer_journey['customer_data'])
        
        # Set next step based on condition
        if success:
            next_step_id = step.get('success_step_id')
            result = 'success'
        else:
            next_step_id = step.get('failure_step_id')
            result = 'failure'
            
        if next_step_id:
            customer_journey['next_step_id'] = next_step_id
            
        # Add event
        customer_journey['events'].append({
            'type': 'condition_evaluated',
            'result': result,
            'step_id': step['step_id'],
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Condition {result} for customer {customer_id} in step {step['step_id']}")

    def _process_split_step(self, journey_id: str, customer_id: str, 
                         step: Dict[str, Any]) -> None:
        """Process a split step for a customer."""
        customer_key = f"{journey_id}_{customer_id}"
        customer_journey = self.active_customer_journeys[customer_key]
        
        # Get branches
        branches = step.get('branches', [])
        
        if not branches:
            logger.warning(f"No branches specified for split step {step['step_id']}")
            return
            
        # Determine branch assignment method
        assignment_method = step.get('assignment_method', 'random')
        
        if assignment_method == 'random':
            # Random assignment with probability weights
            weights = [branch.get('probability', 1) for branch in branches]
            total_weight = sum(weights)
            
            if total_weight == 0:
                weights = [1] * len(branches)
                
            normalized_weights = [w / sum(weights) for w in weights]
            
            # Randomly select a branch
            selected_branch = random.choices(branches, weights=normalized_weights, k=1)[0]
            
        elif assignment_method == 'predictive':
            # Use predictive model to assign branch
            if not self.use_predictive_optimization:
                # Fall back to random if predictive is not enabled
                selected_branch = random.choice(branches)
            else:
                # Get prediction model
                model_name = step.get('model', 'conversion')
                
                # Extract features
                features = self._extract_model_features(customer_journey['customer_data'])
                
                # Get probabilities for each branch
                branch_scores = []
                
                for branch in branches:
                    branch_model = f"{model_name}_{branch['branch_id']}"
                    probability = self._predict_probability(branch_model, features)
                    branch_scores.append((branch, probability))
                    
                # Select branch with highest probability
                if branch_scores:
                    selected_branch = max(branch_scores, key=lambda x: x[1])[0]
                else:
                    selected_branch = random.choice(branches)
                    
        else:
            # Unknown assignment method, use random
            selected_branch = random.choice(branches)
            
        # Set next step based on selected branch
        next_step_id = selected_branch.get('next_step_id')
        
        if next_step_id:
            customer_journey['next_step_id'] = next_step_id
            
        # Add event
        customer_journey['events'].append({
            'type': 'branch_selected',
            'branch_id': selected_branch.get('branch_id', 'unknown'),
            'step_id': step['step_id'],
            'timestamp': datetime.now().isoformat()
        })

    def _process_goal_step(self, journey_id: str, customer_id: str, 
                         step: Dict[str, Any]) -> None:
        """Process a goal step for a customer."""
        customer_key = f"{journey_id}_{customer_id}"
        customer_journey = self.active_customer_journeys[customer_key]
        
        # Mark goal as achieved
        customer_journey['goal_achieved'] = True
        
        # Update journey analytics
        self.journey_analytics[journey_id]['goal_completion_dates'][customer_id] = datetime.now().isoformat()
        
        # Track conversion for journey metrics
        journey = self.journeys[journey_id]
        step_id = step['step_id']
        
        if step_id in journey['step_performance']:
            journey['step_performance'][step_id]['conversions'] += 1
            impressions = journey['step_performance'][step_id]['impressions']
            
            if impressions > 0:
                journey['step_performance'][step_id]['conversion_rate'] = (
                    journey['step_performance'][step_id]['conversions'] / impressions
                )
        
        # Update overall journey conversion metrics
        journey['completed_customers'] += 1
        total_customers = journey['active_customers'] + journey['completed_customers']
        
        if total_customers > 0:
            journey['conversion_rate'] = journey['completed_customers'] / total_customers
            
        logger.info(f"Customer {customer_id} achieved journey goal in step {step_id}")
        
        # Continue to next step if defined
        next_step_id = self._get_next_step_id(journey, step, customer_journey)
        
        if next_step_id:
            customer_journey['next_step_id'] = next_step_id
    
    def _process_exit_step(self, journey_id: str, customer_id: str, 
                         step: Dict[str, Any]) -> None:
        """Process an exit step for a customer."""
        customer_key = f"{journey_id}_{customer_id}"
        customer_journey = self.active_customer_journeys[customer_key]
        
        # Mark as inactive and completed
        customer_journey['active'] = False
        customer_journey['completed'] = True
        
        # Update journey analytics
        self.journey_analytics[journey_id]['exit_dates'][customer_id] = datetime.now().isoformat()
        
        # Update journey metrics
        journey = self.journeys[journey_id]
        journey['active_customers'] -= 1
        
        logger.info(f"Customer {customer_id} exited journey at step {step['step_id']}")
    
    def _get_next_step_id(self, journey: Dict[str, Any], 
                        current_step: Dict[str, Any],
                        customer_journey: Dict[str, Any]) -> Optional[str]:
        """Get the ID of the next step in the journey."""
        # If next step already defined in the customer journey, use that
        if customer_journey.get('next_step_id'):
            return customer_journey['next_step_id']
            
        # If next step defined in the current step, use that
        if 'next_step_id' in current_step:
            return current_step['next_step_id']
            
        # Check journey graph
        journey_graph = journey.get('journey_graph', {})
        current_step_id = current_step['step_id']
        
        if current_step_id in journey_graph and journey_graph[current_step_id]:
            return journey_graph[current_step_id][0]
            
        return None
    
    def advance_customer_journey(self, journey_id: str, 
                               customer_id: str,
                               event_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Advance a customer to the next step in their journey.
        
        Args:
            journey_id: ID of the journey
            customer_id: ID of the customer
            event_data: Optional event data to update customer data
            
        Returns:
            Updated customer journey state
        """
        customer_key = f"{journey_id}_{customer_id}"
        
        if customer_key not in self.active_customer_journeys:
            logger.warning(f"Customer {customer_id} not found in journey {journey_id}")
            return None
            
        customer_journey = self.active_customer_journeys[customer_key]
        
        if not customer_journey['active']:
            logger.info(f"Customer {customer_id} is no longer active in journey {journey_id}")
            return customer_journey
            
        # Check if we have a next action date and it's in the future
        if customer_journey.get('next_action_date'):
            next_action = datetime.fromisoformat(customer_journey['next_action_date'])
            
            if next_action > datetime.now():
                logger.info(f"Customer {customer_id} waiting until {next_action}")
                return customer_journey
                
        # Clear next action date
        customer_journey['next_action_date'] = None
        
        # Update customer data with event data
        if event_data:
            customer_journey['customer_data'].update(event_data)
            
        # Check if we have a next step
        next_step_id = customer_journey.get('next_step_id')
        
        if not next_step_id:
            logger.warning(f"No next step defined for customer {customer_id} in journey {journey_id}")
            return customer_journey
            
        # Move to next step
        customer_journey['current_step_id'] = next_step_id
        customer_journey['next_step_id'] = None
        
        # Process the new step
        self._process_customer_step(journey_id, customer_id)
        
        return customer_journey
    
    def visualize_journey(self, journey_id: str, 
                        output_path: Optional[str] = None) -> plt.Figure:
        """
        Create a visualization of a journey.
        
        Args:
            journey_id: ID of the journey
            output_path: Optional path to save the visualization
            
        Returns:
            Matplotlib figure with the visualization
        """
        if journey_id not in self.journeys:
            raise ValueError(f"Journey ID {journey_id} not found")
            
        journey = self.journeys[journey_id]
        steps = journey.get('steps', [])
        journey_graph = journey.get('journey_graph', {})
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes for each step
        for step in steps:
            step_id = step['step_id']
            step_name = step.get('name', step_id)
            step_type = step.get('type', 'unknown')
            
            G.add_node(step_id, name=step_name, type=step_type)
            
        # Add edges from the journey graph
        for source, targets in journey_graph.items():
            for target in targets:
                G.add_edge(source, target)
                
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define colors for different step types
        color_map = {
            'message': 'lightblue',
            'delay': 'lightgray',
            'condition': 'lightyellow',
            'branch': 'lightgreen',
            'goal': 'gold',
            'exit': 'lightcoral',
            'unknown': 'white'
        }
        
        # Get node positions using layout algorithm
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes with different colors based on type
        for node_type in color_map:
            nodes = [node for node in G.nodes if G.nodes[node].get('type') == node_type]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color_map[node_type], 
                                 node_size=2000, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrowsize=20, ax=ax)
        
        # Draw labels
        labels = {node: G.nodes[node].get('name', node) for node in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, ax=ax)
        
        # Add conversion rates to nodes if available
        for node in G.nodes:
            if node in journey.get('step_performance', {}):
                performance = journey['step_performance'][node]
                conversion_rate = performance.get('conversion_rate', 0)
                
                if conversion_rate > 0:
                    x, y = pos[node]
                    ax.text(x, y-0.1, f"{conversion_rate:.1%}", 
                          ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', 
                                                            facecolor='white', alpha=0.7))
        
        # Set plot title and remove axes
        plt.title(f"Journey: {journey.get('journey_name', journey_id)}")
        plt.axis('off')
        
        # Save figure if output path provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            logger.info(f"Journey visualization saved to {output_path}")
            
        return fig

    def _send_to_channel(self, journey_id: str, customer_id: str, 
                       channel: str, content: Dict[str, Any],
                       step_id: str) -> Dict[str, Any]:
        """Send a message to a channel using the messaging API."""
        try:
            # Get channel config
            channel_config = self.channel_configs.get(channel, {})
            api_endpoint = channel_config.get('api_endpoint')
            
            if not api_endpoint:
                logger.warning(f"No API endpoint defined for channel '{channel}'")
                return {"status": "error", "message": f"No API endpoint for channel '{channel}'"}
                
            # In a real implementation, this would make an API call to the messaging service
            # For demonstration, we'll simulate the API call
            
            # Prepare payload
            payload = {
                "journey_id": journey_id,
                "step_id": step_id,
                "customer_id": customer_id,
                "channel": channel,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            
            # Log the simulated API call
            logger.info(f"Sending to {channel} API: {api_endpoint}")
            
            # Simulate successful response
            response = {
                "status": "success",
                "message": f"Message sent to {customer_id} via {channel}",
                "message_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error sending to {channel}: {str(e)}")
            return {"status": "error", "message": str(e)}

    def train_conversion_model(self, journey_id: str, 
                             model_name: str = 'conversion',
                             model_type: str = 'random_forest',
                             training_data: Optional[pd.DataFrame] = None,
                             features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train a predictive model for optimizing journey conversion.
        
        Args:
            journey_id: ID of the journey
            model_name: Name for the trained model
            model_type: Type of model to train ('random_forest', etc.)
            training_data: Optional custom training data
            features: List of features to use
            
        Returns:
            Dictionary with model training results
        """
        if journey_id not in self.journeys:
            raise ValueError(f"Journey ID {journey_id} not found")
            
        journey = self.journeys[journey_id]
        journey_analytics = self.journey_analytics.get(journey_id, {})
        
        # Use provided training data or extract from journey analytics
        if training_data is None:
            # Build training data from journey analytics
            training_data = self._extract_training_data(journey_id)
            
            if len(training_data) < 100:
                logger.warning(f"Limited training data ({len(training_data)} samples) for journey {journey_id}")
                
        # Get label column (whether the customer converted)
        label_column = 'converted'
        
        if label_column not in training_data.columns:
            raise ValueError(f"Training data must include '{label_column}' column")
            
        # Split features and target
        X = training_data.drop(columns=[label_column])
        y = training_data[label_column]
        
        # Use provided features or all available features
        if features is not None:
            X = X[features]
            
        # Handle missing values
        X = X.fillna(0)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Handle categorical features
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )
        
        # Choose and configure model
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            logger.warning(f"Unknown model type '{model_type}', using random forest instead")
            
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Evaluate model
        train_accuracy = pipeline.score(X_train, y_train)
        test_accuracy = pipeline.score(X_test, y_test)
        
        # Get feature importances if available
        feature_importances = {}
        
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            all_features = numeric_features + categorical_features
            
            # Need to account for one-hot encoding expansion
            if categorical_features:
                # This is an approximation; exact mapping depends on the one-hot encoder
                importances = pipeline.named_steps['model'].feature_importances_
                idx = 0
                
                for feature in all_features:
                    if feature in numeric_features:
                        feature_importances[feature] = importances[idx]
                        idx += 1
                    else:
                        # Approximation for categorical features (average of one-hot columns)
                        # The exact mapping would require knowledge of the one-hot encoding
                        feature_importances[feature] = importances[idx]
                        idx += 1
            else:
                # Simple case: only numeric features
                for feature, importance in zip(all_features, pipeline.named_steps['model'].feature_importances_):
                    feature_importances[feature] = importance
        
        # Store model
        model_key = f"{journey_id}_{model_name}"
        self.conversion_models[model_key] = {
            'pipeline': pipeline,
            'features': X.columns.tolist(),
            'training_date': datetime.now().isoformat(),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'feature_importances': feature_importances,
            'sample_size': len(X_train) + len(X_test)
        }
        
        logger.info(f"Trained conversion model for journey {journey_id} with {test_accuracy:.2f} accuracy")
        
        # Return results
        return {
            'journey_id': journey_id,
            'model_name': model_name,
            'model_type': model_type,
            'training_date': self.conversion_models[model_key]['training_date'],
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'feature_importances': feature_importances,
            'sample_size': len(X_train) + len(X_test)
        }
    
    def _extract_training_data(self, journey_id: str) -> pd.DataFrame:
        """Extract training data from journey analytics."""
        if journey_id not in self.journeys:
            raise ValueError(f"Journey ID {journey_id} not found")
            
        analytics = self.journey_analytics.get(journey_id, {})
        
        if not analytics:
            logger.warning(f"No analytics found for journey {journey_id}")
            return pd.DataFrame()
            
        # Get customer data
        customer_data = []
        
        for customer_id in analytics.get('entry_dates', {}):
            # Get customer journey
            customer_key = f"{journey_id}_{customer_id}"
            
            if customer_key not in self.active_customer_journeys:
                continue
                
            journey_data = self.active_customer_journeys[customer_key]
            
            # Extract features
            features = self._extract_model_features(journey_data.get('customer_data', {}))
            
            # Add whether the customer converted
            features['converted'] = 1 if journey_data.get('goal_achieved', False) else 0
            
            # Add time to conversion if available
            if journey_data.get('goal_achieved', False) and 'goal_completion_dates' in analytics:
                entry_date = datetime.fromisoformat(journey_data.get('entry_date'))
                completion_date = datetime.fromisoformat(analytics['goal_completion_dates'].get(customer_id))
                
                conversion_time = (completion_date - entry_date).total_seconds() / 3600  # hours
                features['conversion_time'] = conversion_time
            else:
                features['conversion_time'] = None
                
            customer_data.append(features)
            
        # Create DataFrame
        df = pd.DataFrame(customer_data)
        
        return df
    
    def _predict_probability(self, model_name: str, features: Dict[str, Any]) -> float:
        """
        Predict conversion probability using a trained model.
        
        Args:
            model_name: Name of the model to use
            features: Customer features for prediction
            
        Returns:
            Predicted conversion probability
        """
        # Find the model
        model_info = None
        
        for model_key, info in self.conversion_models.items():
            if model_key.endswith(f"_{model_name}"):
                model_info = info
                break
                
        if model_info is None:
            logger.warning(f"Model '{model_name}' not found")
            return 0.5  # default probability
            
        # Get pipeline
        pipeline = model_info.get('pipeline')
        expected_features = model_info.get('features', [])
        
        # Prepare feature vector
        feature_vector = {}
        
        for feature in expected_features:
            if feature in features:
                feature_vector[feature] = features[feature]
            else:
                feature_vector[feature] = 0  # default value
                
        # Convert to DataFrame
        df = pd.DataFrame([feature_vector])
        
        # Predict probability
        try:
            proba = pipeline.predict_proba(df)[0][1]  # probability of positive class
            return proba
        except Exception as e:
            logger.error(f"Error predicting probability: {str(e)}")
            return 0.5  # default probability
    
    def analyze_journey_performance(self, journey_id: str) -> Dict[str, Any]:
        """
        Analyze the performance of a journey.
        
        Args:
            journey_id: ID of the journey
            
        Returns:
            Dictionary with performance metrics
        """
        if journey_id not in self.journeys:
            raise ValueError(f"Journey ID {journey_id} not found")
            
        journey = self.journeys[journey_id]
        analytics = self.journey_analytics.get(journey_id, {})
        
        # Get conversion rate
        active_customers = journey.get('active_customers', 0)
        completed_customers = journey.get('completed_customers', 0)
        total_customers = active_customers + completed_customers
        
        conversion_rate = 0.0
        if total_customers > 0:
            conversion_rate = completed_customers / total_customers
            
        # Get average time to conversion
        conversion_times = []
        
        if 'entry_dates' in analytics and 'goal_completion_dates' in analytics:
            for customer_id in analytics['goal_completion_dates']:
                if customer_id in analytics['entry_dates']:
                    entry_date = datetime.fromisoformat(analytics['entry_dates'][customer_id])
                    completion_date = datetime.fromisoformat(analytics['goal_completion_dates'][customer_id])
                    
                    conversion_time = (completion_date - entry_date).total_seconds() / 3600  # hours
                    conversion_times.append(conversion_time)
        
        avg_conversion_time = None
        if conversion_times:
            avg_conversion_time = sum(conversion_times) / len(conversion_times)
            
        # Get step performance
        step_performance = journey.get('step_performance', {})
        
        # Get channel performance
        channel_metrics = {}
        
        for customer_key, customer_journey in self.active_customer_journeys.items():
            if not customer_key.startswith(f"{journey_id}_"):
                continue
                
            for channel, messages in customer_journey.get('messages_sent', {}).items():
                if channel not in channel_metrics:
                    channel_metrics[channel] = {'count': 0, 'conversions': 0}
                    
                channel_metrics[channel]['count'] += len(messages)
                
                if customer_journey.get('goal_achieved', False):
                    channel_metrics[channel]['conversions'] += 1
        
        # Calculate channel conversion rates
        for channel in channel_metrics:
            if channel_metrics[channel]['count'] > 0:
                channel_metrics[channel]['conversion_rate'] = (
                    channel_metrics[channel]['conversions'] / channel_metrics[channel]['count']
                )
            else:
                channel_metrics[channel]['conversion_rate'] = 0.0
        
        # Compile results
        return {
            'journey_id': journey_id,
            'journey_name': journey.get('journey_name'),
            'status': journey.get('status'),
            'total_customers': total_customers,
            'active_customers': active_customers,
            'completed_customers': completed_customers,
            'conversion_rate': conversion_rate,
            'avg_conversion_time': avg_conversion_time,
            'step_performance': step_performance,
            'channel_performance': channel_metrics,
            'analysis_date': datetime.now().isoformat()
        }
    
    def export_journey_analytics(self, journey_id: str, 
                               output_format: str = 'json',
                               output_path: Optional[str] = None) -> str:
        """
        Export journey analytics data.
        
        Args:
            journey_id: ID of the journey
            output_format: Format to export (json, csv)
            output_path: Path to save the exported data
            
        Returns:
            Path to the exported data
        """
        if journey_id not in self.journeys:
            raise ValueError(f"Journey ID {journey_id} not found")
            
        # Get performance analysis
        performance = self.analyze_journey_performance(journey_id)
        
        # Get journey config
        journey = self.journeys[journey_id]
        
        # Combine data
        export_data = {
            'journey_config': {k: v for k, v in journey.items() if k not in ['steps', 'journey_graph']},
            'performance': performance,
            'export_date': datetime.now().isoformat()
        }
        
        # Generate output path if not provided
        if output_path is None:
            output_path = f"journey_{journey_id}_analytics.{output_format}"
            
        # Export based on format
        if output_format == 'json':
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
                
        elif output_format == 'csv':
            # Convert to DataFrame for CSV export
            journey_df = pd.DataFrame([export_data['journey_config']])
            performance_df = pd.DataFrame([export_data['performance']])
            
            # Export to CSV
            journey_df.to_csv(output_path.replace('.csv', '_config.csv'), index=False)
            performance_df.to_csv(output_path, index=False)
            
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
            
        logger.info(f"Exported journey analytics to {output_path}")
        
        return output_path