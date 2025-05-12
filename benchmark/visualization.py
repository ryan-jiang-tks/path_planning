import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from .path_planning_benchmark import BenchmarkResult

def plot_benchmark_results(results: Dict[str, List[BenchmarkResult]]):
    """Create comprehensive visualization of benchmark results"""
    # Prepare data
    environments = list(set([k.split('_')[0] for k in results.keys()]))
    planners = list(set([k.split('_')[1] for k in results.keys()]))
    
    metrics = {
        'Success Rate (%)': lambda r: sum(x.success for x in r) / len(r) * 100,
        'Execution Time (s)': lambda r: np.mean([x.execution_time for x in r]),
        'Path Length': lambda r: np.mean([x.path_length for x in r if x.success]),
        'Path Smoothness': lambda r: np.mean([x.path_smoothness for x in r if x.success])
    }
    
    # Create separate window for each metric
    for metric_name, metric_func in metrics.items():
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        # Prepare data for plotting
        x = np.arange(len(environments))
        width = 0.8 / len(planners)
        
        for i, planner in enumerate(planners):
            values = []
            for env in environments:
                key = f"{env}_{planner}"
                if key in results:
                    values.append(metric_func(results[key]))
                else:
                    values.append(0)
            
            ax.bar(x + i * width, values, width, label=planner.upper())
        
        # Customize plot
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} by Environment')
        ax.set_xticks(x + width * (len(planners) - 1) / 2)
        ax.set_xticklabels(environments)
        ax.legend()
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45)
        plt.tight_layout()
        plt.show(block=False)

def plot_convergence_analysis(results: Dict[str, List[BenchmarkResult]]):
    """Plot convergence analysis for iterative planners (PSO, RRT)"""
    # Filter for iterative planners
    iterative_planners = ['pso', 'RRT', 'modifiedRRT']
    
    # Create new window
    plt.figure(figsize=(10, 6))
    
    for key in results:
        env, planner = key.split('_')
        if planner in iterative_planners:
            # Calculate average path length over successful trials
            successful_lengths = [r.path_length for r in results[key] if r.success]
            if successful_lengths:
                plt.plot(successful_lengths, label=f'{env}-{planner.upper()}')
    
    plt.xlabel('Trial Number')
    plt.ylabel('Path Length')
    plt.title('Convergence Analysis of Iterative Planners')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

def plot_performance_radar(results: Dict[str, List[BenchmarkResult]]):
    """Create radar plot comparing planner performance"""
    # Prepare metrics
    metrics = [
        ('Success', lambda r: sum(x.success for x in r) / len(r) * 100),
        ('Speed', lambda r: 1 / (np.mean([x.execution_time for x in r]) + 1e-6)),
        ('Length', lambda r: 1 / (np.mean([x.path_length for x in r if x.success]) + 1e-6)),
        ('Smoothness', lambda r: np.mean([x.path_smoothness for x in r if x.success]))
    ]
    
    # Create new window
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='polar')
    
    # Prepare data
    planners = list(set([k.split('_')[1] for k in results.keys()]))
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    for planner in planners:
        values = []
        for metric_name, metric_func in metrics:
            planner_results = [results[k] for k in results.keys() if k.split('_')[1] == planner]
            if planner_results:
                value = np.mean([metric_func(r) for r in planner_results])
                values.append(value)
        
        values = np.array(values)
        values = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-6)
        values = np.concatenate((values, [values[0]]))
        
        ax.plot(angles, values, 'o-', linewidth=2, label=planner.upper())
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[0] for m in metrics])
    plt.title('Planner Performance Comparison')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.show(block=False)

    # Add this line to keep windows open
    plt.show()
