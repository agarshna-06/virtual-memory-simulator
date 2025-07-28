"""
Virtual memory simulator with visualization capabilities.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Any
import random

from algorithms import FIFOAlgorithm, LRUAlgorithm, OptimalAlgorithm, simulate_algorithm

class VirtualMemorySimulator:
    """Main simulator class for virtual memory page replacement."""
    
    def __init__(self):
        self.algorithms = {
            'FIFO': FIFOAlgorithm,
            'LRU': LRUAlgorithm,
            'Optimal': OptimalAlgorithm
        }
        self.results = {}
        
    def generate_random_reference_string(self, length: int, page_range: int) -> List[int]:
        """Generate a random reference string."""
        return [random.randint(0, page_range - 1) for _ in range(length)]
    
    def run_simulation(self, frame_count: int, reference_string: List[int]) -> Dict[str, Any]:
        """Run simulation for all algorithms."""
        results = {}
        
        for name, algorithm_class in self.algorithms.items():
            algorithm, history = simulate_algorithm(algorithm_class, frame_count, reference_string)
            results[name] = {
                'algorithm': algorithm,
                'history': history,
                'stats': algorithm.get_stats()
            }
        
        self.results = results
        return results
    
    def create_frame_visualization(self, algorithm_name: str, step: int = None) -> go.Figure:
        """Create frame table visualization for a specific algorithm and step."""
        if algorithm_name not in self.results:
            return go.Figure()
        
        history = self.results[algorithm_name]['history']
        if not history:
            return go.Figure()
        
        if step is None:
            step = len(history) - 1
        
        step = min(step, len(history) - 1)
        current_state = history[step]
        
        # Create frame visualization
        fig = go.Figure()
        
        frames = current_state['frames']
        frame_count = len(frames) if frames else 0
        
        # Add frame boxes
        for i, page in enumerate(frames):
            fig.add_shape(
                type="rect",
                x0=i, y0=0, x1=i+1, y1=1,
                line=dict(color="blue", width=2),
                fillcolor="lightblue"
            )
            fig.add_annotation(
                x=i+0.5, y=0.5,
                text=str(page),
                showarrow=False,
                font=dict(size=16, color="black")
            )
        
        # Add empty frames if any
        max_frames = self.results[algorithm_name]['algorithm'].frame_count
        for i in range(frame_count, max_frames):
            fig.add_shape(
                type="rect",
                x0=i, y0=0, x1=i+1, y1=1,
                line=dict(color="gray", width=2),
                fillcolor="lightgray"
            )
            fig.add_annotation(
                x=i+0.5, y=0.5,
                text="Empty",
                showarrow=False,
                font=dict(size=12, color="gray")
            )
        
        fig.update_layout(
            title=f"{algorithm_name} - Frame Table (Step {step + 1})",
            xaxis=dict(range=[-0.5, max_frames + 0.5], showticklabels=False, showgrid=False),
            yaxis=dict(range=[-0.5, 1.5], showticklabels=False, showgrid=False),
            showlegend=False,
            height=200,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig
    
    def create_page_fault_timeline(self) -> go.Figure:
        """Create timeline visualization of page faults for all algorithms."""
        fig = make_subplots(
            rows=len(self.algorithms), cols=1,
            subplot_titles=list(self.algorithms.keys()),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        colors = ['red', 'blue', 'green']
        
        for i, (name, result) in enumerate(self.results.items()):
            history = result['history']
            steps = list(range(1, len(history) + 1))
            pages = [h['page'] for h in history]
            faults = [1 if h['is_fault'] else 0 for h in history]
            
            # Add page access trace
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=pages,
                    mode='markers+lines',
                    name=f'{name} - Pages',
                    marker=dict(
                        color=[colors[i] if f else 'lightgray' for f in faults],
                        size=8,
                        symbol=['x' if f else 'circle' for f in faults]
                    ),
                    line=dict(color=colors[i], width=2)
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            title="Page Access Timeline (X = Page Fault, O = Page Hit)",
            height=200 * len(self.algorithms),
            showlegend=True
        )
        
        return fig
    
    def create_comparison_chart(self) -> go.Figure:
        """Create comparison chart of algorithm performance."""
        if not self.results:
            return go.Figure()
        
        algorithms = list(self.results.keys())
        page_faults = [self.results[alg]['stats']['page_faults'] for alg in algorithms]
        hit_ratios = [self.results[alg]['stats']['hit_ratio'] * 100 for alg in algorithms]
        access_latencies = [self.results[alg]['stats']['access_latency'] for alg in algorithms]
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Page Faults', 'Hit Ratio (%)', 'Average Access Latency'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Page faults bar chart
        fig.add_trace(
            go.Bar(x=algorithms, y=page_faults, name='Page Faults', marker_color='red'),
            row=1, col=1
        )
        
        # Hit ratio bar chart
        fig.add_trace(
            go.Bar(x=algorithms, y=hit_ratios, name='Hit Ratio', marker_color='green'),
            row=1, col=2
        )
        
        # Access latency bar chart
        fig.add_trace(
            go.Bar(x=algorithms, y=access_latencies, name='Access Latency', marker_color='blue'),
            row=1, col=3
        )
        
        fig.update_layout(
            title="Algorithm Performance Comparison",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_statistics_table(self) -> pd.DataFrame:
        """Create statistics comparison table."""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for name, result in self.results.items():
            stats = result['stats']
            data.append({
                'Algorithm': name,
                'Page Faults': stats['page_faults'],
                'Page Hits': stats['page_hits'],
                'Total Accesses': stats['total_accesses'],
                'Hit Ratio (%)': f"{stats['hit_ratio'] * 100:.2f}%",
                'Fault Ratio (%)': f"{stats['fault_ratio'] * 100:.2f}%",
                'Avg Access Latency': f"{stats['access_latency']:.2f}"
            })
        
        return pd.DataFrame(data)
    
    def export_results(self) -> Dict[str, Any]:
        """Export simulation results for download."""
        export_data = {
            'statistics': self.create_statistics_table().to_dict('records'),
            'detailed_history': {}
        }
        
        for name, result in self.results.items():
            export_data['detailed_history'][name] = result['history']
        
        return export_data
