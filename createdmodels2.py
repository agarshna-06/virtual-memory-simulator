"""
Streamlit-based Virtual Memory Simulator
Implements LRU, FIFO, and Optimal page replacement algorithms with interactive visualization.
"""

import streamlit as st
import pandas as pd
import json
import random
from typing import List

from simulator import VirtualMemorySimulator

# Configure page
st.set_page_config(
    page_title="Virtual Memory Simulator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'simulator' not in st.session_state:
    st.session_state.simulator = VirtualMemorySimulator()
if 'reference_string' not in st.session_state:
    st.session_state.reference_string = []
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

# Main title
st.title("🧠 Virtual Memory Simulator")
st.markdown("""
This simulator demonstrates three page replacement algorithms: **FIFO**, **LRU**, and **Optimal**.
Configure the parameters below and run the simulation to compare their performance.
""")

# Sidebar for configuration
st.sidebar.header("⚙️ Simulation Configuration")

# Frame count configuration
frame_count = st.sidebar.slider(
    "Number of Frames",
    min_value=1,
    max_value=10,
    value=3,
    help="Number of physical memory frames available"
)

# Reference string configuration
st.sidebar.subheader("Reference String")
ref_string_option = st.sidebar.radio(
    "Choose input method:",
    ["Generate Random", "Manual Input"],
    help="How to create the page reference string"
)

if ref_string_option == "Generate Random":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        string_length = st.number_input(
            "Length",
            min_value=5,
            max_value=50,
            value=20,
            help="Number of page references"
        )
    with col2:
        page_range = st.number_input(
            "Page Range",
            min_value=3,
            max_value=20,
            value=8,
            help="Pages numbered from 0 to this value-1"
        )
    
    if st.sidebar.button("Generate Random String"):
        st.session_state.reference_string = st.session_state.simulator.generate_random_reference_string(
            string_length, page_range
        )
        st.session_state.simulation_run = False

else:
    manual_input = st.sidebar.text_area(
        "Enter reference string (comma-separated):",
        value="1,2,3,4,1,2,5,1,2,3,4,5",
        help="Enter page numbers separated by commas"
    )
    
    if st.sidebar.button("Parse Reference String"):
        try:
            st.session_state.reference_string = [
                int(x.strip()) for x in manual_input.split(',') if x.strip()
            ]
            st.session_state.simulation_run = False
        except ValueError:
            st.sidebar.error("Invalid input! Please enter numbers separated by commas.")

# Display current reference string
if st.session_state.reference_string:
    st.sidebar.write("**Current Reference String:**")
    st.sidebar.write(str(st.session_state.reference_string))

# Run simulation button
if st.sidebar.button("🚀 Run Simulation", type="primary"):
    if st.session_state.reference_string:
        with st.spinner("Running simulation..."):
            st.session_state.simulator.run_simulation(
                frame_count, 
                st.session_state.reference_string
            )
            st.session_state.simulation_run = True
            st.session_state.current_step = len(st.session_state.reference_string) - 1
        st.success("Simulation completed!")
    else:
        st.sidebar.error("Please generate or enter a reference string first!")

# Simulation mode selection
if st.session_state.simulation_run:
    st.sidebar.subheader("📺 Visualization Mode")
    
    viz_mode = st.sidebar.radio(
        "Select mode:",
        ["Overview", "Step-by-Step"],
        help="Choose between complete results or step-by-step analysis"
    )
    
    if viz_mode == "Step-by-Step":
        max_steps = len(st.session_state.reference_string)
        st.session_state.current_step = st.sidebar.slider(
            "Current Step",
            min_value=0,
            max_value=max_steps - 1,
            value=st.session_state.current_step,
            help="Navigate through simulation steps"
        )

# Main content area
if not st.session_state.simulation_run:
    # Welcome screen with algorithm explanations
    st.markdown("## 📚 Algorithm Explanations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔄 FIFO (First In First Out)")
        st.markdown("""
        - **Strategy**: Replace the oldest page in memory
        - **Implementation**: Uses a queue to track page order
        - **Pros**: Simple to implement and understand
        - **Cons**: May not reflect actual page usage patterns
        - **Best for**: Systems with predictable access patterns
        """)
    
    with col2:
        st.subheader("⏰ LRU (Least Recently Used)")
        st.markdown("""
        - **Strategy**: Replace the page that hasn't been used for the longest time
        - **Implementation**: Tracks access timestamps or usage order
        - **Pros**: Good approximation of optimal behavior
        - **Cons**: More complex implementation overhead
        - **Best for**: General-purpose systems with locality of reference
        """)
    
    with col3:
        st.subheader("🎯 Optimal (Belady's Algorithm)")
        st.markdown("""
        - **Strategy**: Replace the page that will be accessed farthest in the future
        - **Implementation**: Requires future knowledge (theoretical)
        - **Pros**: Provably optimal (minimum page faults)
        - **Cons**: Impossible to implement in practice
        - **Best for**: Benchmark for comparing other algorithms
        """)
    
    st.markdown("---")
    st.info("👈 Configure your simulation parameters in the sidebar and click 'Run Simulation' to begin!")

else:
    # Display simulation results
    simulator = st.session_state.simulator
    
    if viz_mode == "Overview":
        # Performance comparison
        st.markdown("## 📊 Algorithm Performance Comparison")
        
        # Statistics table
        stats_df = simulator.create_statistics_table()
        st.dataframe(stats_df, use_container_width=True)
        
        # Performance charts
        comparison_chart = simulator.create_comparison_chart()
        st.plotly_chart(comparison_chart, use_container_width=True)
        
        # Page fault timeline
        st.markdown("## 📈 Page Access Timeline")
        timeline_chart = simulator.create_page_fault_timeline()
        st.plotly_chart(timeline_chart, use_container_width=True)
        
        # Current frame states for all algorithms
        st.markdown("## 🖼️ Final Frame States")
        cols = st.columns(3)
        for i, alg_name in enumerate(['FIFO', 'LRU', 'Optimal']):
            with cols[i]:
                frame_viz = simulator.create_frame_visualization(alg_name)
                st.plotly_chart(frame_viz, use_container_width=True)
    
    else:  # Step-by-Step mode
        step = st.session_state.current_step
        current_page = st.session_state.reference_string[step]
        
        st.markdown(f"## 🔍 Step-by-Step Analysis - Step {step + 1}")
        st.markdown(f"**Accessing Page: {current_page}**")
        
        # Navigation buttons
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
        with col1:
            if st.button("⏮️ First") and step > 0:
                st.session_state.current_step = 0
                st.rerun()
        with col2:
            if st.button("⏪ Previous") and step > 0:
                st.session_state.current_step = step - 1
                st.rerun()
        with col3:
            if st.button("⏩ Next") and step < len(st.session_state.reference_string) - 1:
                st.session_state.current_step = step + 1
                st.rerun()
        with col4:
            if st.button("⏭️ Last") and step < len(st.session_state.reference_string) - 1:
                st.session_state.current_step = len(st.session_state.reference_string) - 1
                st.rerun()
        
        # Frame visualizations for current step
        st.markdown("### Memory Frame States")
        cols = st.columns(3)
        
        for i, alg_name in enumerate(['FIFO', 'LRU', 'Optimal']):
            with cols[i]:
                frame_viz = simulator.create_frame_visualization(alg_name, step)
                st.plotly_chart(frame_viz, use_container_width=True)
                
                # Show if this step was a page fault or hit
                if alg_name in simulator.results and step < len(simulator.results[alg_name]['history']):
                    history_entry = simulator.results[alg_name]['history'][step]
                    if history_entry['is_fault']:
                        st.error(f"🚨 Page Fault")
                    else:
                        st.success(f"✅ Page Hit")
        
        # Progress indicator
        progress = (step + 1) / len(st.session_state.reference_string)
        st.progress(progress)
        st.caption(f"Progress: {step + 1}/{len(st.session_state.reference_string)} steps")
    
    # Export functionality
    st.markdown("---")
    st.markdown("## 💾 Export Results")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📋 Copy Statistics to Clipboard"):
            stats_df = simulator.create_statistics_table()
            st.write("Statistics table:")
            st.dataframe(stats_df)
    
    with col2:
        export_data = simulator.export_results()
        export_json = json.dumps(export_data, indent=2)
        
        st.download_button(
            label="📥 Download Full Results (JSON)",
            data=export_json,
            file_name="vm_simulation_results.json",
            mime="application/json"
        )

# Footer with educational information
st.markdown("---")
st.markdown("""
### 🎓 Educational Notes

**Page Fault**: Occurs when a requested page is not in physical memory and must be loaded from secondary storage.

**Access Latency**: Average time to access a page, considering that page faults take significantly longer than page hits.

**Hit Ratio**: Percentage of memory accesses that result in page hits (found in memory).

**Frame Table**: Visual representation of physical memory frames and their current contents.

*This simulator is designed for educational purposes to help understand virtual memory management concepts.*
""")
