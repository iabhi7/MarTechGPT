import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os
from datetime import datetime, timedelta
import logging
import uuid
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('multivariant_testing')

class MultiVariantTester:
    """
    Advanced multi-variant testing framework for optimizing marketing campaigns
    beyond simple A/B testing. Supports testing multiple variants simultaneously
    with sophisticated statistical analysis.
    """
    
    def __init__(self, significance_level: float = 0.05, 
                min_sample_size: int = 100,
                auto_stop_enabled: bool = True):
        """
        Initialize the MultiVariantTester.
        
        Args:
            significance_level: Statistical significance level (default: 0.05)
            min_sample_size: Minimum sample size required for each variant
            auto_stop_enabled: Whether to automatically stop tests when significance is reached
        """
        self.significance_level = significance_level
        self.min_sample_size = min_sample_size
        self.auto_stop_enabled = auto_stop_enabled
        self.tests = {}
        self.segments = {}
        self.variant_recommendations = {}
        logger.info("MultiVariantTester initialized")

    def create_test(self, test_name: str, 
                  variants: List[Dict[str, Any]], 
                  control_index: int = 0,
                  metrics: List[str] = None,
                  primary_metric: str = 'conversion_rate',
                  segment_dimensions: List[str] = None,
                  traffic_allocation: Union[Dict[str, float], float] = None,
                  description: str = None) -> Dict[str, Any]:
        """
        Create a new multi-variant test.
        
        Args:
            test_name: Name of the test
            variants: List of variant configurations
            control_index: Index of the control variant (default: 0)
            metrics: List of metrics to track (default: ['open_rate', 'click_rate', 'conversion_rate'])
            primary_metric: Primary metric for decision making (default: 'conversion_rate')
            segment_dimensions: List of dimensions for segmented analysis (e.g., ['age_group', 'device'])
            traffic_allocation: Percentage of traffic to allocate to this test or dict of variant allocations
            description: Description of the test
            
        Returns:
            Dictionary with test configuration
        """
        # Generate test ID
        test_id = str(uuid.uuid4())
        
        # Set default metrics if not provided
        if metrics is None:
            metrics = ['open_rate', 'click_rate', 'conversion_rate']
            
        # Validate variants
        if len(variants) < 2:
            raise ValueError("At least two variants (including control) are required")
            
        if control_index >= len(variants):
            raise ValueError(f"Control index {control_index} is out of range for {len(variants)} variants")
            
        # Set traffic allocation
        if traffic_allocation is None:
            # Equal allocation to all variants
            variant_allocation = {i: 1.0 / len(variants) for i in range(len(variants))}
        elif isinstance(traffic_allocation, dict):
            # Custom allocation per variant
            variant_allocation = traffic_allocation
        else:
            # Equal allocation within the specified test percentage
            variant_allocation = {i: traffic_allocation / len(variants) for i in range(len(variants))}
            
        # Create variant IDs and prepare variant data
        for i, variant in enumerate(variants):
            variant_id = f"variant_{i}" if i != control_index else "control"
            variant['variant_id'] = variant_id
            variant['index'] = i
            variant['traffic_allocation'] = variant_allocation.get(i, 1.0 / len(variants))
            
        # Create test configuration
        test_config = {
            'test_id': test_id,
            'test_name': test_name,
            'variants': variants,
            'control_index': control_index,
            'metrics': metrics,
            'primary_metric': primary_metric,
            'segment_dimensions': segment_dimensions,
            'status': 'created',
            'creation_date': datetime.now().isoformat(),
            'update_date': None,
            'description': description,
            'results': None,
            'segment_results': {},
            'winning_variant': None,
            'confidence_level': None,
            'sample_sizes': {variant['variant_id']: 0 for variant in variants},
            'metrics_data': {metric: {variant['variant_id']: [] for variant in variants} for metric in metrics}
        }
        
        # Store test configuration
        self.tests[test_id] = test_config
        
        logger.info(f"Created multi-variant test '{test_name}' with ID {test_id} and {len(variants)} variants")
        return test_config
    
    def get_test(self, test_id: str) -> Dict[str, Any]:
        """Get test configuration by ID."""
        if test_id not in self.tests:
            raise ValueError(f"Test ID {test_id} not found")
        return self.tests[test_id]
    
    def get_test_by_name(self, test_name: str) -> Dict[str, Any]:
        """Get test configuration by name."""
        for test_id, test in self.tests.items():
            if test['test_name'] == test_name:
                return test
        raise ValueError(f"Test with name '{test_name}' not found")
    
    def start_test(self, test_id: str) -> Dict[str, Any]:
        """
        Start a test.
        
        Args:
            test_id: ID of the test to start
            
        Returns:
            Updated test configuration
        """
        if test_id not in self.tests:
            raise ValueError(f"Test ID {test_id} not found")
            
        test = self.tests[test_id]
        test['status'] = 'running'
        test['start_date'] = datetime.now().isoformat()
        test['update_date'] = datetime.now().isoformat()
        
        logger.info(f"Started test '{test['test_name']}' with ID {test_id}")
        return test
    
    def stop_test(self, test_id: str) -> Dict[str, Any]:
        """
        Stop a test.
        
        Args:
            test_id: ID of the test to stop
            
        Returns:
            Updated test configuration with final results
        """
        if test_id not in self.tests:
            raise ValueError(f"Test ID {test_id} not found")
            
        test = self.tests[test_id]
        
        if test['status'] == 'completed':
            logger.info(f"Test '{test['test_name']}' is already completed")
            return test
            
        test['status'] = 'completed'
        test['end_date'] = datetime.now().isoformat()
        test['update_date'] = datetime.now().isoformat()
        
        # Calculate final results
        self._calculate_test_results(test_id)
        
        logger.info(f"Stopped test '{test['test_name']}' with ID {test_id}")
        return test 

    def add_test_data(self, test_id: str, 
                      variant_id: str, 
                      metrics_data: Dict[str, float],
                      segment_data: Optional[Dict[str, str]] = None,
                      user_id: Optional[str] = None,
                      timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Add metrics data for a variant in a test.
        
        Args:
            test_id: ID of the test
            variant_id: ID of the variant
            metrics_data: Dictionary of metric values (e.g., {'open_rate': 1, 'click_rate': 0})
            segment_data: Dictionary of segment values (e.g., {'device': 'mobile', 'age_group': '25-34'})
            user_id: Optional user ID for tracking
            timestamp: Optional timestamp for the data point
            
        Returns:
            Updated test configuration
        """
        if test_id not in self.tests:
            raise ValueError(f"Test ID {test_id} not found")
            
        test = self.tests[test_id]
        
        if test['status'] != 'running':
            logger.warning(f"Test '{test['test_name']}' is not running. Current status: {test['status']}")
            
        # Validate variant ID
        valid_variant_ids = [v['variant_id'] for v in test['variants']]
        if variant_id not in valid_variant_ids:
            raise ValueError(f"Variant ID {variant_id} not found in test '{test['test_name']}'")
            
        # Validate metrics
        for metric in metrics_data:
            if metric not in test['metrics']:
                logger.warning(f"Metric '{metric}' not defined in test '{test['test_name']}'. Ignoring.")
                continue
                
            # Add the metric value to the test data
            value = metrics_data[metric]
            test['metrics_data'][metric][variant_id].append(value)
            
        # Increment sample size
        test['sample_sizes'][variant_id] += 1
        
        # Handle segmented data if provided
        if segment_data and test['segment_dimensions']:
            # Create a segment key based on segment dimensions
            relevant_dimensions = {k: v for k, v in segment_data.items() 
                                if k in test['segment_dimensions']}
                                
            if relevant_dimensions:
                segment_key = json.dumps(relevant_dimensions, sort_keys=True)
                
                # Initialize segment if it doesn't exist
                if segment_key not in test['segment_results']:
                    test['segment_results'][segment_key] = {
                        'segment_data': relevant_dimensions,
                        'metrics_data': {metric: {v_id: [] for v_id in valid_variant_ids} 
                                        for metric in test['metrics']},
                        'sample_sizes': {v_id: 0 for v_id in valid_variant_ids},
                        'results': None
                    }
                    
                # Add metrics to segment
                for metric in metrics_data:
                    if metric in test['metrics']:
                        value = metrics_data[metric]
                        test['segment_results'][segment_key]['metrics_data'][metric][variant_id].append(value)
                        
                # Increment segment sample size
                test['segment_results'][segment_key]['sample_sizes'][variant_id] += 1
        
        # Store user ID if provided (for consistent variant assignment)
        if user_id:
            self.variant_recommendations[user_id] = {
                'test_id': test_id,
                'variant_id': variant_id,
                'timestamp': timestamp or datetime.now().isoformat()
            }
            
        # Check if we should automatically calculate results
        if self.auto_stop_enabled:
            # Check if we have enough samples for all variants
            if all(size >= self.min_sample_size for size in test['sample_sizes'].values()):
                # Calculate results
                self._calculate_test_results(test_id)
                
                # Check if we have a statistically significant winner
                if (test['results'] and 
                    test['winning_variant'] and 
                    test['confidence_level'] >= 1 - self.significance_level):
                    # Stop the test
                    logger.info(f"Auto-stopping test '{test['test_name']}' due to significant results")
                    self.stop_test(test_id)
        
        return test
        
    def _calculate_test_results(self, test_id: str) -> Dict[str, Any]:
        """
        Calculate statistical results for a test.
        
        Args:
            test_id: ID of the test
            
        Returns:
            Dictionary with test results
        """
        if test_id not in self.tests:
            raise ValueError(f"Test ID {test_id} not found")
            
        test = self.tests[test_id]
        metrics = test['metrics']
        primary_metric = test['primary_metric']
        variants = test['variants']
        control_id = variants[test['control_index']]['variant_id']
        
        # Initialize results
        results = {
            'metrics': {},
            'comparison': {},
            'winning_variant': None,
            'confidence_level': None,
            'sample_sizes': test['sample_sizes'].copy()
        }
        
        # Calculate metrics for each variant
        for metric in metrics:
            results['metrics'][metric] = {}
            
            for variant in variants:
                variant_id = variant['variant_id']
                values = test['metrics_data'][metric][variant_id]
                
                if not values:
                    results['metrics'][metric][variant_id] = {
                        'mean': None,
                        'std': None,
                        'count': 0,
                        'sum': None
                    }
                    continue
                    
                # Calculate statistics
                values_array = np.array(values)
                mean = np.mean(values_array)
                std = np.std(values_array) if len(values_array) > 1 else 0
                count = len(values_array)
                sum_value = np.sum(values_array)
                
                results['metrics'][metric][variant_id] = {
                    'mean': float(mean),
                    'std': float(std),
                    'count': count,
                    'sum': float(sum_value)
                }
        
        # Make comparisons against control
        for metric in metrics:
            results['comparison'][metric] = {}
            
            # Get control metrics
            control_metrics = results['metrics'][metric].get(control_id)
            
            if not control_metrics or control_metrics['count'] < self.min_sample_size:
                # Skip comparison if control has insufficient data
                continue
                
            control_mean = control_metrics['mean']
            control_std = control_metrics['std']
            control_count = control_metrics['count']
            
            # Compare each variant to control
            for variant in variants:
                variant_id = variant['variant_id']
                
                if variant_id == control_id:
                    # Skip comparison with self
                    continue
                    
                variant_metrics = results['metrics'][metric].get(variant_id)
                
                if not variant_metrics or variant_metrics['count'] < self.min_sample_size:
                    # Skip comparison if variant has insufficient data
                    continue
                    
                variant_mean = variant_metrics['mean']
                variant_std = variant_metrics['std']
                variant_count = variant_metrics['count']
                
                # Calculate p-value using t-test
                try:
                    t_stat, p_value = stats.ttest_ind_from_stats(
                        mean1=variant_mean, std1=variant_std, nobs1=variant_count,
                        mean2=control_mean, std2=control_std, nobs2=control_count,
                        equal_var=False  # Welch's t-test
                    )
                    
                    # Calculate effect size (relative improvement)
                    if control_mean != 0:
                        relative_improvement = (variant_mean - control_mean) / control_mean
                    else:
                        relative_improvement = float('inf') if variant_mean > 0 else 0
                        
                    # Calculate absolute difference
                    absolute_difference = variant_mean - control_mean
                    
                    # Calculate confidence interval (95%)
                    # Using Welch-Satterthwaite equation for degrees of freedom
                    std_err_diff = np.sqrt((variant_std**2 / variant_count) + 
                                         (control_std**2 / control_count))
                    
                    dof = ((std_err_diff**2)**2) / (
                        ((variant_std**2 / variant_count)**2 / (variant_count - 1)) +
                        ((control_std**2 / control_count)**2 / (control_count - 1))
                    )
                    
                    # For 95% confidence interval
                    t_crit = stats.t.ppf(0.975, dof)
                    
                    margin_of_error = t_crit * std_err_diff
                    ci_lower = absolute_difference - margin_of_error
                    ci_upper = absolute_difference + margin_of_error
                    
                    # Determine if the result is statistically significant
                    significant = p_value < self.significance_level
                    
                    # Store comparison results
                    results['comparison'][metric][variant_id] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'relative_improvement': float(relative_improvement),
                        'absolute_difference': float(absolute_difference),
                        'confidence_interval': [float(ci_lower), float(ci_upper)],
                        'significant': significant
                    }
                except Exception as e:
                    logger.error(f"Error calculating statistics for variant {variant_id}: {e}")
                    results['comparison'][metric][variant_id] = {
                        'error': str(e)
                    }
        
        # Determine the winning variant based on the primary metric
        try:
            winning_variant = None
            max_improvement = 0
            confidence_level = 0
            
            # Only consider variants with significant results
            significant_variants = []
            
            for variant in variants:
                variant_id = variant['variant_id']
                
                if variant_id == control_id:
                    continue
                    
                comparison = results['comparison'].get(primary_metric, {}).get(variant_id, {})
                
                if comparison.get('significant', False):
                    relative_improvement = comparison.get('relative_improvement', 0)
                    
                    # Only consider positive improvements
                    if relative_improvement > 0:
                        significant_variants.append((variant_id, relative_improvement, comparison.get('p_value', 1.0)))
                        
            if significant_variants:
                # Sort by improvement (descending)
                significant_variants.sort(key=lambda x: x[1], reverse=True)
                
                # Select the variant with the highest improvement
                winning_variant = significant_variants[0][0]
                max_improvement = significant_variants[0][1]
                
                # Confidence level is 1 - p_value
                p_value = significant_variants[0][2]
                confidence_level = 1 - p_value
                
            results['winning_variant'] = winning_variant
            results['confidence_level'] = confidence_level
        except Exception as e:
            logger.error(f"Error determining winning variant: {e}")
            
        # Store the results in the test
        test['results'] = results
        test['winning_variant'] = results['winning_variant']
        test['confidence_level'] = results['confidence_level']
        test['update_date'] = datetime.now().isoformat()
        
        # Calculate segment results if available
        for segment_key, segment in test['segment_results'].items():
            segment_metrics = segment['metrics_data']
            
            # Skip if not enough data
            if any(len(values) < self.min_sample_size 
                for metric_data in segment_metrics.values() 
                for values in metric_data.values()):
                continue
                
            # Calculate segment results (similar to main results)
            segment_results = {variant_id: {} for variant_id in segment['sample_sizes']}
            
            # Calculate metrics for each variant in this segment
            for metric in metrics:
                for variant in variants:
                    variant_id = variant['variant_id']
                    values = segment_metrics[metric][variant_id]
                    
                    if not values:
                        segment_results[variant_id][metric] = None
                        segment_results[variant_id][f'{metric}_p_value'] = None
                        continue
                        
                    values_array = np.array(values)
                    mean = float(np.mean(values_array))
                    segment_results[variant_id][metric] = mean
                    
                    # Calculate p-value compared to control (if not control)
                    if variant_id != control_id:
                        control_values = segment_metrics[metric][control_id]
                        
                        if control_values:
                            try:
                                t_stat, p_value = stats.ttest_ind(
                                    values_array, 
                                    np.array(control_values),
                                    equal_var=False  # Welch's t-test
                                )
                                segment_results[variant_id][f'{metric}_p_value'] = float(p_value)
                            except Exception as e:
                                logger.error(f"Error calculating p-value for segment {segment_key}, "
                                          f"variant {variant_id}: {e}")
                                segment_results[variant_id][f'{metric}_p_value'] = None
                        else:
                            segment_results[variant_id][f'{metric}_p_value'] = None
                    else:
                        segment_results[variant_id][f'{metric}_p_value'] = None
                        
            # Store segment results
            segment['results'] = segment_results
            
        logger.info(f"Calculated results for test '{test['test_name']}' with ID {test_id}")
        
        return results 

    def visualize_results(self, test_id: str, 
                        output_path: Optional[str] = None,
                        metric: Optional[str] = None,
                        include_confidence_intervals: bool = True,
                        include_segment_charts: bool = True) -> plt.Figure:
        """
        Create a visualization of test results.
        
        Args:
            test_id: ID of the test
            output_path: Optional path to save the visualization
            metric: Specific metric to visualize (defaults to primary metric)
            include_confidence_intervals: Whether to include confidence intervals
            include_segment_charts: Whether to include segment charts
            
        Returns:
            Matplotlib figure with the visualization
        """
        if test_id not in self.tests:
            raise ValueError(f"Test ID {test_id} not found")
            
        test = self.tests[test_id]
        results = test.get('results')
        
        if not results:
            logger.warning(f"No results available for test '{test['test_name']}'")
            return None
            
        # Use primary metric if not specified
        if metric is None:
            metric = test['primary_metric']
            
        if metric not in test['metrics']:
            raise ValueError(f"Metric '{metric}' not found in test '{test['test_name']}'")
            
        # Get variant names and metric values
        variant_names = []
        metric_values = []
        error_bars = None if not include_confidence_intervals else []
        colors = []
        
        control_id = test['variants'][test['control_index']]['variant_id']
        control_value = results['metrics'][metric][control_id]['mean']
        
        # Define color scheme
        color_map = {
            'control': 'blue',
            'winning': 'green',
            'losing': 'red',
            'neutral': 'grey'
        }
        
        # Create a bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for variant in test['variants']:
            variant_id = variant['variant_id']
            variant_name = variant.get('name', variant_id)
            variant_names.append(variant_name)
            
            variant_result = results['metrics'][metric].get(variant_id, {})
            value = variant_result.get('mean')
            metric_values.append(value)
            
            # Determine color based on performance
            if variant_id == control_id:
                color = color_map['control']
            elif variant_id == test['winning_variant']:
                color = color_map['winning']
            elif variant_id != control_id:
                comparison = results['comparison'][metric].get(variant_id, {})
                if comparison.get('significant', False):
                    if comparison.get('relative_improvement', 0) > 0:
                        color = color_map['winning']
                    else:
                        color = color_map['losing']
                else:
                    color = color_map['neutral']
            else:
                color = color_map['neutral']
                
            colors.append(color)
            
            # Add confidence intervals if requested
            if include_confidence_intervals and variant_id != control_id:
                comparison = results['comparison'][metric].get(variant_id, {})
                ci = comparison.get('confidence_interval')
                
                if ci is not None:
                    # Convert to error bar format (lower, upper relative to the mean)
                    lower_err = value - ci[0]
                    upper_err = ci[1] - value
                    error_bars.append((lower_err, upper_err))
                else:
                    error_bars.append((0, 0))
            elif include_confidence_intervals:
                # Control variant has no comparison
                error_bars.append((0, 0))
        
        # Create bar chart
        bars = ax.bar(variant_names, metric_values, color=colors, alpha=0.7)
        
        # Add error bars if requested
        if include_confidence_intervals and error_bars:
            # Convert error_bars to format required by errorbar
            err_low = [e[0] for e in error_bars]
            err_high = [e[1] for e in error_bars]
            
            ax.errorbar(
                x=range(len(variant_names)),
                y=metric_values,
                yerr=[err_low, err_high],
                fmt='none',  # No line connecting points
                ecolor='black',
                capsize=5
            )
        
        # Add value labels on top of bars
        for bar, value in zip(bars, metric_values):
            if value is not None:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{value:.3f}',
                      ha='center', va='bottom', rotation=0)
        
        # Set labels and title
        metric_label = metric.replace('_', ' ').title()
        ax.set_xlabel('Variant')
        ax.set_ylabel(metric_label)
        ax.set_title(f"{test['test_name']}: {metric_label} by Variant")
        
        # Add a legend explaining colors
        import matplotlib.patches as mpatches
        legend_patches = [
            mpatches.Patch(color=color_map['control'], label='Control'),
            mpatches.Patch(color=color_map['winning'], label='Winning (Significant)'),
            mpatches.Patch(color=color_map['losing'], label='Losing (Significant)'),
            mpatches.Patch(color=color_map['neutral'], label='No Significant Difference')
        ]
        ax.legend(handles=legend_patches, loc='best')
        
        plt.tight_layout()
        
        # Add segment charts if requested
        if include_segment_charts and test['segment_results']:
            segments = test['segment_results']
            
            # Only include segments with results
            segments_with_results = {k: v for k, v in segments.items() if v.get('results')}
            
            if segments_with_results:
                # Create a new figure for segments
                n_segments = len(segments_with_results)
                n_cols = min(3, n_segments)
                n_rows = (n_segments + n_cols - 1) // n_cols  # Ceiling division
                
                segment_fig, segment_axes = plt.subplots(
                    n_rows, n_cols, 
                    figsize=(5*n_cols, 4*n_rows),
                    squeeze=False
                )
                segment_axes = segment_axes.flatten()
                
                # Create a chart for each segment
                for i, (segment_key, segment) in enumerate(segments_with_results.items()):
                    if i >= len(segment_axes):
                        break
                        
                    ax = segment_axes[i]
                    
                    # Prepare data for this segment
                    variant_names = []
                    segment_metric_values = []
                    segment_colors = []
                    
                    control_id = test['variants'][test['control_index']]['variant_id']
                    control_value = segment['results'][control_id][metric]
                    
                    for variant in test['variants']:
                        variant_id = variant['variant_id']
                        variant_name = variant.get('name', variant_id)
                        variant_names.append(variant_name)
                        
                        value = segment['results'][variant_id][metric]
                        segment_metric_values.append(value)
                        
                        # Calculate confidence interval if not control
                        if variant_id != control_id:
                            p_value = segment['results'][variant_id].get(f'{metric}_p_value', 1.0)
                            significant = p_value is not None and p_value < self.significance_level
                            
                            # Set color based on significance and improvement
                            if significant and value > control_value:
                                segment_colors.append('green')
                            elif significant and value < control_value:
                                segment_colors.append('red')
                            else:
                                segment_colors.append('grey')
                        else:
                            segment_colors.append('blue')  # Control variant
                            
                    # Create bar chart for this segment
                    segment_bars = ax.bar(variant_names, segment_metric_values, color=segment_colors, alpha=0.7)
                    
                    # Add labels
                    ax.set_xlabel('Variant')
                    ax.set_ylabel(metric_label)
                    
                    # Create segment description
                    segment_desc = ", ".join([f"{k}={v}" for k, v in segment['segment_data'].items()])
                    ax.set_title(f"Segment: {segment_desc}")
                    
                    # Add value labels on top of bars
                    for bar, value in zip(segment_bars, segment_metric_values):
                        if value is not None:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                  f'{value:.3f}',
                                  ha='center', va='bottom', rotation=0)
                                  
                # Hide any unused subplots
                for j in range(i+1, len(segment_axes)):
                    segment_axes[j].set_visible(False)
                    
                segment_fig.suptitle(f"{test['test_name']}: {metric_label} by Segment", fontsize=16)
                segment_fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
                
                if output_path:
                    segment_fig_path = f"{os.path.splitext(output_path)[0]}_segments.png"
                    segment_fig.savefig(segment_fig_path)
                    logger.info(f"Segment visualization saved to {segment_fig_path}")
                    
                # Return both figures
                return fig, segment_fig
        
        # Save figure if output path provided
        if output_path:
            fig.savefig(output_path)
            logger.info(f"Visualization saved to {output_path}")
            
        return fig
    
    def get_variant_for_user(self, test_id: str, user_id: str, 
                          user_attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get the variant assignment for a user in a test.
        
        Args:
            test_id: ID of the test
            user_id: ID of the user
            user_attributes: Optional dictionary of user attributes for targeting
            
        Returns:
            Dictionary with variant information
        """
        if test_id not in self.tests:
            raise ValueError(f"Test ID {test_id} not found")
            
        test = self.tests[test_id]
        
        if test['status'] != 'running':
            logger.warning(f"Getting variant for non-running test '{test['test_name']}'. "
                          f"Current status: {test['status']}")
        
        # Check if user already has a variant assignment
        if user_id in self.variant_recommendations:
            recommendation = self.variant_recommendations[user_id]
            
            if recommendation['test_id'] == test_id:
                # Return the existing variant assignment
                variant_id = recommendation['variant_id']
                variant = next((v for v in test['variants'] if v['variant_id'] == variant_id), None)
                
                if variant:
                    return {
                        'test_id': test_id,
                        'variant_id': variant_id,
                        'variant_name': variant.get('name', variant_id),
                        'variant_content': variant.get('content'),
                        'is_control': variant_id == test['variants'][test['control_index']]['variant_id']
                    }
        
        # If we reach here, the user needs a new variant assignment
        variants = test['variants']
        
        # Get traffic allocations
        allocations = [(v['variant_id'], v['traffic_allocation']) for v in variants]
        variant_ids = [a[0] for a in allocations]
        weights = [a[1] for a in allocations]
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Randomly select a variant based on traffic allocation
        variant_id = random.choices(variant_ids, weights=normalized_weights, k=1)[0]
        
        # Get the selected variant
        variant = next(v for v in variants if v['variant_id'] == variant_id)
        
        # Store the assignment for future reference
        self.variant_recommendations[user_id] = {
            'test_id': test_id,
            'variant_id': variant_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # Return the variant information
        return {
            'test_id': test_id,
            'variant_id': variant_id,
            'variant_name': variant.get('name', variant_id),
            'variant_content': variant.get('content'),
            'is_control': variant_id == variants[test['control_index']]['variant_id']
        }
    
    def export_results(self, test_id: str, 
                      output_format: str = 'json',
                      output_path: Optional[str] = None,
                      include_segments: bool = True) -> Union[Dict[str, Any], str]:
        """
        Export test results.
        
        Args:
            test_id: ID of the test
            output_format: Format to export ('json', 'csv', or 'dict')
            output_path: Path to save the results (for 'json' and 'csv')
            include_segments: Whether to include segment results
            
        Returns:
            Dictionary with results or path to output file
        """
        if test_id not in self.tests:
            raise ValueError(f"Test ID {test_id} not found")
            
        test = self.tests[test_id]
        
        # Calculate results if not already done
        if not test.get('results'):
            logger.info(f"Calculating results for test '{test['test_name']}' before export")
            self._calculate_test_results(test_id)
            
        # Create export data
        export_data = {
            'test_id': test['test_id'],
            'test_name': test['test_name'],
            'status': test['status'],
            'creation_date': test['creation_date'],
            'variants': [{'variant_id': v['variant_id'], 'name': v.get('name', v['variant_id'])} 
                        for v in test['variants']],
            'control_variant': test['variants'][test['control_index']]['variant_id'],
            'metrics': test['metrics'],
            'primary_metric': test['primary_metric'],
            'sample_sizes': test['sample_sizes'],
            'results': test['results'],
            'winning_variant': test['winning_variant'],
            'confidence_level': test['confidence_level']
        }
        
        if include_segments and test['segment_results']:
            export_data['segments'] = {}
            
            for segment_key, segment in test['segment_results'].items():
                if segment.get('results'):
                    export_data['segments'][segment_key] = {
                        'segment_data': segment['segment_data'],
                        'sample_sizes': segment['sample_sizes'],
                        'results': segment['results']
                    }
        
        # Export based on format
        if output_format == 'dict':
            return export_data
        elif output_format == 'json':
            if output_path is None:
                output_path = f"test_results_{test['test_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            logger.info(f"Results exported to {output_path}")
            return output_path
        elif output_format == 'csv':
            if output_path is None:
                output_path = f"test_results_{test['test_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
            # Flatten the nested structure for CSV
            flat_results = []
            
            # Main results
            for variant in test['variants']:
                variant_id = variant['variant_id']
                variant_name = variant.get('name', variant_id)
                
                # Get results for this variant
                for metric in test['metrics']:
                    metric_result = test['results']['metrics'][metric].get(variant_id, {})
                    
                    row = {
                        'test_id': test['test_id'],
                        'test_name': test['test_name'],
                        'variant_id': variant_id,
                        'variant_name': variant_name,
                        'is_control': variant_id == test['variants'][test['control_index']]['variant_id'],
                        'is_winner': variant_id == test['winning_variant'],
                        'segment': 'Overall',
                        'metric': metric,
                        'value': metric_result.get('mean'),
                        'std': metric_result.get('std'),
                        'sample_size': test['sample_sizes'][variant_id]
                    }
                    
                    # Add comparison metrics if available and not control
                    if (variant_id != test['variants'][test['control_index']]['variant_id'] and
                        metric in test['results']['comparison'] and
                        variant_id in test['results']['comparison'][metric]):
                        
                        comparison = test['results']['comparison'][metric][variant_id]
                        
                        row.update({
                            'p_value': comparison.get('p_value'),
                            'relative_improvement': comparison.get('relative_improvement'),
                            'absolute_difference': comparison.get('absolute_difference'),
                            'significant': comparison.get('significant', False),
                            'ci_lower': comparison.get('confidence_interval', [None, None])[0],
                            'ci_upper': comparison.get('confidence_interval', [None, None])[1]
                        })
                        
                    flat_results.append(row)
            
            # Segment results
            if include_segments and test['segment_results']:
                for segment_key, segment in test['segment_results'].items():
                    if not segment.get('results'):
                        continue
                        
                    segment_name = ", ".join([f"{k}={v}" for k, v in segment['segment_data'].items()])
                    
                    for variant in test['variants']:
                        variant_id = variant['variant_id']
                        variant_name = variant.get('name', variant_id)
                        
                        # Get results for this variant in this segment
                        for metric in test['metrics']:
                            if (variant_id in segment['results'] and 
                                metric in segment['results'][variant_id]):
                                
                                value = segment['results'][variant_id][metric]
                                p_value = segment['results'][variant_id].get(f'{metric}_p_value')
                                
                                row = {
                                    'test_id': test['test_id'],
                                    'test_name': test['test_name'],
                                    'variant_id': variant_id,
                                    'variant_name': variant_name,
                                    'is_control': variant_id == test['variants'][test['control_index']]['variant_id'],
                                    'is_winner': False,  # No winning variant for segments
                                    'segment': segment_name,
                                    'metric': metric,
                                    'value': value,
                                    'std': None,  # No std for segments
                                    'sample_size': segment['sample_sizes'][variant_id],
                                    'p_value': p_value,
                                    'significant': p_value is not None and p_value < self.significance_level
                                }
                                
                                flat_results.append(row)
            
            # Convert to DataFrame and export
            df = pd.DataFrame(flat_results)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Results exported to {output_path}")
            return output_path
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def save_tests(self, file_path: str) -> None:
        """
        Save all tests to a JSON file.
        
        Args:
            file_path: Path to save the tests
        """
        # Convert tests to JSON-serializable format
        serializable_tests = {}
        
        for test_id, test in self.tests.items():
            serializable_test = {}
            
            for key, value in test.items():
                if isinstance(value, np.ndarray):
                    serializable_test[key] = value.tolist()
                elif isinstance(value, datetime):
                    serializable_test[key] = value.isoformat()
                else:
                    serializable_test[key] = value
                    
            serializable_tests[test_id] = serializable_test
            
        with open(file_path, 'w') as f:
            json.dump(serializable_tests, f, indent=2)
            
        logger.info(f"Saved {len(self.tests)} tests to {file_path}")
        
    def load_tests(self, file_path: str) -> None:
        """
        Load tests from a JSON file.
        
        Args:
            file_path: Path to load the tests from
        """
        with open(file_path, 'r') as f:
            tests = json.load(f)
            
        self.tests = tests
        logger.info(f"Loaded {len(self.tests)} tests from {file_path}")

    def analyze_segments(self, test_id: str, 
                       metric: Optional[str] = None,
                       min_sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Analyze segments to find interesting insights and recommendations.
        
        Args:
            test_id: ID of the test
            metric: Specific metric to analyze (defaults to primary metric)
            min_sample_size: Minimum sample size for a segment to be included
            
        Returns:
            List of segment insights sorted by importance
        """
        if test_id not in self.tests:
            raise ValueError(f"Test ID {test_id} not found")
            
        test = self.tests[test_id]
        
        # Use primary metric if not specified
        if metric is None:
            metric = test['primary_metric']
            
        if metric not in test['metrics']:
            raise ValueError(f"Metric '{metric}' not found in test '{test['test_name']}'")
            
        # Use default min sample size if not specified
        if min_sample_size is None:
            min_sample_size = self.min_sample_size
            
        # Get overall results
        results = test.get('results')
        
        if not results:
            logger.warning(f"No results available for test '{test['test_name']}'")
            return []
            
        # Calculate overall average for the metric
        control_id = test['variants'][test['control_index']]['variant_id']
        overall_average = {}
        
        for variant in test['variants']:
            variant_id = variant['variant_id']
            variant_result = results['metrics'][metric].get(variant_id, {})
            overall_average[variant_id] = variant_result.get('mean', 0)
            
        # Analyze each segment
        segment_insights = []
        
        for segment_key, segment in test['segment_results'].items():
            segment_results = segment.get('results')
            
            if not segment_results:
                continue
                
            # Check sample size for this segment
            if any(size < min_sample_size for size in segment['sample_sizes'].values()):
                continue
                
            # Calculate best variant for this segment
            best_variant_id = None
            best_variant_value = float('-inf')
            
            for variant_id, metrics in segment_results.items():
                if metric in metrics and metrics[metric] > best_variant_value:
                    best_variant_value = metrics[metric]
                    best_variant_id = variant_id
                    
            if best_variant_id is None:
                continue
                
            # Calculate improvement over overall average
            if overall_average.get(best_variant_id, 0) > 0:
                improvement = ((best_variant_value / overall_average[best_variant_id]) - 1) * 100
            else:
                improvement = 0
                
            # Get variant name
            best_variant = next((v for v in test['variants'] if v['variant_id'] == best_variant_id), None)
            best_variant_name = best_variant.get('name', best_variant_id) if best_variant else best_variant_id
            
            # Get p-value if not control
            p_value = None
            significant = False
            
            if best_variant_id != control_id:
                p_value = segment_results[best_variant_id].get(f'{metric}_p_value')
                significant = p_value is not None and p_value < self.significance_level
                
            # Add insight
            segment_insights.append({
                'segment_key': segment_key,
                'segment_data': segment['segment_data'],
                'best_variant_id': best_variant_id,
                'best_variant_name': best_variant_name,
                'metric': metric,
                'value': best_variant_value,
                'overall_value': overall_average.get(best_variant_id, 0),
                'improvement': improvement,
                'p_value': p_value,
                'significant': significant,
                'sample_size': segment['sample_sizes'][best_variant_id]
            })
            
        # Sort by improvement (descending)
        segment_insights.sort(key=lambda x: abs(x['improvement']), reverse=True)
        
        return segment_insights


# Example usage
if __name__ == "__main__":
    # Initialize the tester
    tester = MultiVariantTester(
        significance_level=0.05,
        min_sample_size=100,
        auto_stop_enabled=True
    )
    
    # Create a test
    subject_line_test = tester.create_test(
        test_name="Email Subject Line Test",
        variants=[
            {
                "name": "Control",
                "content": "Spring Sale: 20% off all products"
            },
            {
                "name": "Emoji Variant",
                "content": "Spring Sale: 20% off all products 🌺"
            },
            {
                "name": "Urgency Variant",
                "content": "Last Chance: 20% off all products ends today"
            },
            {
                "name": "Personalized Variant",
                "content": "[First Name], get 20% off your favorite products"
            }
        ],
        metrics=["open_rate", "click_rate", "conversion_rate"],
        primary_metric="conversion_rate",
        segment_dimensions=["device_type", "user_age_group", "subscription_tier"]
    )
    
    # Start the test
    tester.start_test(subject_line_test["test_id"])
    
    # Simulate adding data
    import random
    
    # Generate some realistic data
    for i in range(1000):
        user_id = f"user_{i}"
        
        # Randomly assign to variants (in a real scenario, this would be done by get_variant_for_user)
        variant = tester.get_variant_for_user(
            test_id=subject_line_test["test_id"],
            user_id=user_id
        )
        variant_id = variant["variant_id"]
        
        # Simulate engagement metrics
        if variant_id == "control":
            # Control metrics
            open_rate = random.random() < 0.20  # 20% open rate
            click_rate = open_rate and random.random() < 0.15  # 15% click rate for opens
            conversion_rate = click_rate and random.random() < 0.10  # 10% conversion rate for clicks
        elif variant_id == "variant_1":  # Emoji variant
            # Emoji tends to improve open rates but not necessarily other metrics
            open_rate = random.random() < 0.25  # 25% open rate
            click_rate = open_rate and random.random() < 0.14  # 14% click rate for opens
            conversion_rate = click_rate and random.random() < 0.10  # 10% conversion rate for clicks
        elif variant_id == "variant_2":  # Urgency variant
            # Urgency can improve open and click rates
            open_rate = random.random() < 0.23  # 23% open rate
            click_rate = open_rate and random.random() < 0.18  # 18% click rate for opens
            conversion_rate = click_rate and random.random() < 0.12  # 12% conversion rate for clicks
        else:  # Personalized variant
            # Personalization can improve all metrics
            open_rate = random.random() < 0.28  # 28% open rate
            click_rate = open_rate and random.random() < 0.20  # 20% click rate for opens
            conversion_rate = click_rate and random.random() < 0.15  # 15% conversion rate for clicks
            
        # Create metrics data
        metrics_data = {
            "open_rate": 1 if open_rate else 0,
            "click_rate": 1 if click_rate else 0,
            "conversion_rate": 1 if conversion_rate else 0
        }
        
        # Add segment data
        segment_data = {
            "device_type": random.choice(["mobile", "desktop", "tablet"]),
            "user_age_group": random.choice(["18-24", "25-34", "35-44", "45+"]),
            "subscription_tier": random.choice(["free", "basic", "premium"])
        }
        
        # Add data to the test
        tester.add_test_data(
            test_id=subject_line_test["test_id"],
            variant_id=variant_id,
            metrics_data=metrics_data,
            segment_data=segment_data,
            user_id=user_id
        )
    
    # Calculate results
    tester._calculate_test_results(subject_line_test["test_id"])
    
    # Print the results
    results = subject_line_test["results"]
    print("\nTest Results:")
    
    for metric in subject_line_test["metrics"]:
        print(f"\n{metric.replace('_', ' ').title()}:")
        
        for variant in subject_line_test["variants"]:
            variant_id = variant["variant_id"]
            variant_name = variant.get("name", variant_id)
            
            if variant_id in results["metrics"][metric]:
                value = results["metrics"][metric][variant_id]["mean"]
                print(f"  {variant_name}: {value:.3f}")
                
                # Print comparison if not control
                if variant_id != "control" and variant_id in results["comparison"][metric]:
                    comparison = results["comparison"][metric][variant_id]
                    
                    if comparison.get("significant", False):
                        relative_improvement = comparison.get("relative_improvement", 0) * 100
                        print(f"    {relative_improvement:+.1f}% vs control (significant)")
                    else:
                        relative_improvement = comparison.get("relative_improvement", 0) * 100
                        print(f"    {relative_improvement:+.1f}% vs control (not significant)")
    
    # Print winning variant
    if subject_line_test["winning_variant"]:
        winning_variant = next(v for v in subject_line_test["variants"] 
                             if v["variant_id"] == subject_line_test["winning_variant"])
        
        print(f"\nWinning Variant: {winning_variant.get('name', winning_variant['variant_id'])}")
        print(f"Confidence Level: {subject_line_test['confidence_level']:.2%}")
    else:
        print("\nNo statistically significant winner found.")
    
    # Visualize the results
    tester.visualize_results(
        test_id=subject_line_test["test_id"],
        output_path="test_results.png"
    )
    
    # Export the results
    export_path = tester.export_results(
        test_id=subject_line_test["test_id"],
        output_format="json",
        output_path="test_results.json"
    )
    
    print(f"\nResults exported to {export_path}")
    print("Visualization saved to test_results.png") 