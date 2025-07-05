#!/usr/bin/env python3
"""
Interactive visualization script for the gravitational field system.
This will create actual browser-based visualizations you can interact with.
"""

import numpy as np
import sys
from pathlib import Path
import tempfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from sfa.core.storage import HDF5EmbeddingStore
from sfa.core.gravity import HighDimensionalFlowMetrics
from sfa.visualization.gravity_viz import UMAPVisualizationLayer
from sfa.config.gravity_config import create_demo_config
from sfa.engines.hierarchical_engine import HierarchicalVisualizationEngine

def create_sample_data_with_patterns(store: HDF5EmbeddingStore):
    """Create more interesting sample data with clear patterns."""
    print("📊 Creating sample data with semantic patterns...")
    
    # Enhanced vocabulary with clear semantic relationships
    vocab_groups = {
        'ai_core': ['artificial', 'intelligence', 'machine', 'learning', 'neural', 'network'],
        'ai_tech': ['algorithm', 'model', 'training', 'inference', 'optimization'],
        'social': ['community', 'echo', 'chamber', 'discourse', 'polarization'],
        'media': ['twitter', 'reddit', 'facebook', 'social', 'media', 'platform'],
        'science': ['research', 'experiment', 'hypothesis', 'theory', 'analysis']
    }
    
    # Timestamps with meaningful progression
    timestamps = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05']
    
    embedding_dim = 128
    
    # Create embeddings with strong patterns
    for t_idx, timestamp in enumerate(timestamps):
        print(f"  ⏱️  Generating {timestamp}...")
        
        for group_name, words in vocab_groups.items():
            for word_idx, word in enumerate(words):
                # Base embedding for group
                embedding = np.random.randn(embedding_dim) * 0.1
                
                # Strong group signal
                if group_name == 'ai_core':
                    embedding[:20] += 3.0
                elif group_name == 'ai_tech':
                    embedding[10:30] += 3.0
                elif group_name == 'social':
                    embedding[40:60] += 3.0
                elif group_name == 'media':
                    embedding[50:70] += 3.0
                elif group_name == 'science':
                    embedding[80:100] += 3.0
                
                # Temporal evolution - AI terms get stronger over time
                if group_name.startswith('ai'):
                    embedding *= (1 + 0.2 * t_idx)
                
                # Create "burst" at timestamp 3 for social media terms
                if group_name in ['social', 'media'] and t_idx == 2:
                    embedding *= 2.5  # Burst event!
                
                # Normalize
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                
                store.store_embedding(word, timestamp, embedding)
    
    print("✅ Created rich semantic embedding dataset")
    return vocab_groups, timestamps

def create_interactive_visualization():
    """Create and display interactive gravitational field visualization."""
    print("\n🚀 Starting Interactive Gravitational Field Visualization")
    print("=" * 80)
    
    # Create temporary storage
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        h5_path = tmp_file.name
    
    # Initialize components
    print("💾 Initializing system...")
    store = HDF5EmbeddingStore(h5_path, mode='w', embedding_dim=128)
    
    # Create interesting data
    vocab_groups, timestamps = create_sample_data_with_patterns(store)
    
    # Initialize analysis components
    flow_metrics = HighDimensionalFlowMetrics(store)
    umap_viz = UMAPVisualizationLayer(store, flow_metrics)
    
    # Create configuration for demo
    config = create_demo_config()
    config.hierarchical_viz.enabled_layers = [
        'density_field',
        'orbital_mechanics',
        'communities',
        'motion_trails'
    ]
    
    engine = HierarchicalVisualizationEngine(store, flow_metrics, umap_viz, config)
    
    print("\n🎨 Creating visualizations...")
    
    # 1. Single timestamp gravitational field
    focal_word = 'artificial'
    timestamp = timestamps[2]  # Middle timestamp
    
    print(f"\n📍 Visualization 1: Gravitational field for '{focal_word}' at {timestamp}")
    fig1 = engine.create_complete_visualization(focal_word, timestamp, k_neighbors=15)
    fig1.update_layout(title=f"Gravitational Semantic Field: {focal_word} @ {timestamp}")
    
    # 2. Temporal comparison
    print(f"\n📍 Visualization 2: Temporal evolution of '{focal_word}'")
    fig2 = engine.create_temporal_comparison(focal_word, timestamps, k_neighbors=10)
    
    # 3. Burst detection visualization
    print(f"\n📍 Visualization 3: Semantic burst detection")
    
    # Create custom burst visualization
    bursts = flow_metrics.detect_semantic_bursts('echo', timestamps, k_neighbors=10)
    
    fig3 = go.Figure()
    
    # Plot burst magnitudes over time
    burst_times = []
    burst_magnitudes = []
    
    for burst in bursts:
        burst_times.append(burst.timestamp)
        burst_magnitudes.append(burst.burst_magnitude)
    
    fig3.add_trace(go.Scatter(
        x=timestamps,
        y=[0.5] * len(timestamps),
        mode='lines+markers',
        name='Normal Activity',
        line=dict(color='lightblue', width=2)
    ))
    
    if bursts:
        fig3.add_trace(go.Scatter(
            x=burst_times,
            y=burst_magnitudes,
            mode='markers',
            name='Burst Events',
            marker=dict(
                size=20,
                color='red',
                symbol='star',
                line=dict(width=2, color='darkred')
            ),
            text=[f"Magnitude: {b.burst_magnitude:.2f}<br>Affected: {len(b.affected_words)} words" 
                  for b in bursts],
            hovertemplate='<b>Burst Event</b><br>%{text}<extra></extra>'
        ))
    
    fig3.update_layout(
        title="Semantic Burst Detection Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Activity Level",
        height=400
    )
    
    # 4. Real-time metrics dashboard
    print(f"\n📍 Visualization 4: Real-time metrics dashboard")
    
    # Create subplot figure for metrics
    fig4 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Gravitational Force Distribution', 'Orbital Stability', 
                       'Community Cohesion', 'Temporal Drift'),
        specs=[[{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "scatter3d"}]]
    )
    
    # Get metrics for visualization
    gravity_field = flow_metrics.compute_gravitational_field(focal_word, timestamp, k=20)
    
    if gravity_field:
        # Subplot 1: Force distribution
        fig4.add_trace(
            go.Histogram(x=gravity_field.gravitational_strengths, nbinsx=20, name='Force Distribution'),
            row=1, col=1
        )
        
        # Subplot 2: Orbital stability
        fig4.add_trace(
            go.Scatter(
                x=gravity_field.orbit_radii,
                y=gravity_field.stability_indices,
                mode='markers',
                marker=dict(
                    size=10,
                    color=gravity_field.gravitational_strengths,
                    colorscale='Viridis',
                    showscale=True
                ),
                text=gravity_field.neighbor_words,
                name='Orbital Stability'
            ),
            row=1, col=2
        )
        
        # Subplot 3: Top neighbors by force
        top_10_idx = np.argsort(gravity_field.gravitational_strengths)[-10:]
        fig4.add_trace(
            go.Bar(
                x=[gravity_field.neighbor_words[i] for i in top_10_idx],
                y=[gravity_field.gravitational_strengths[i] for i in top_10_idx],
                name='Top Gravitational Forces'
            ),
            row=2, col=1
        )
    
    fig4.update_layout(height=800, title_text="Gravitational Field Metrics Dashboard")
    
    # Save and display all visualizations
    print("\n🌐 Opening visualizations in browser...")
    
    # Save as HTML files
    output_dir = Path("gravitational_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    visualizations = [
        (fig1, "gravitational_field.html", "Main gravitational field"),
        (fig2, "temporal_evolution.html", "Temporal evolution"),
        (fig3, "burst_detection.html", "Burst events"),
        (fig4, "metrics_dashboard.html", "Real-time metrics")
    ]
    
    for fig, filename, description in visualizations:
        filepath = output_dir / filename
        fig.write_html(str(filepath), auto_open=False)
        print(f"  ✅ Saved: {description} → {filepath}")
    
    # Create index page
    create_index_page(output_dir, visualizations)
    
    # Open index in browser
    index_path = output_dir / "index.html"
    webbrowser.open(f"file://{index_path.absolute()}")
    
    print("\n🎉 Visualizations ready! Check your browser.")
    print(f"📁 Files saved in: {output_dir.absolute()}")
    
    # Cleanup
    store.close()
    Path(h5_path).unlink()

def create_index_page(output_dir, visualizations):
    """Create an index page for all visualizations."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gravitational Semantic Field Visualizations</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f0f0f0;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .viz-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin-top: 30px;
            }
            .viz-card {
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;
            }
            .viz-card h2 {
                color: #555;
                margin-top: 0;
            }
            .viz-card a {
                display: inline-block;
                margin-top: 10px;
                padding: 10px 20px;
                background: #4CAF50;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                transition: background 0.3s;
            }
            .viz-card a:hover {
                background: #45a049;
            }
            .description {
                color: #666;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <h1>🌌 Gravitational Semantic Field Visualizations</h1>
        <div class="viz-grid">
    """
    
    descriptions = {
        "gravitational_field.html": "Interactive 3D visualization showing words orbiting around a focal word with gravitational forces",
        "temporal_evolution.html": "See how semantic fields evolve across multiple timestamps",
        "burst_detection.html": "Identify sudden disruptions in the semantic gravitational field",
        "metrics_dashboard.html": "Real-time metrics showing force distributions, stability, and relationships"
    }
    
    for fig, filename, title in visualizations:
        desc = descriptions.get(filename, "")
        html += f"""
            <div class="viz-card">
                <h2>{title}</h2>
                <p class="description">{desc}</p>
                <a href="{filename}" target="_blank">Open Visualization</a>
            </div>
        """
    
    html += """
        </div>
        <div style="text-align: center; margin-top: 40px; color: #666;">
            <p>💡 Tip: Each visualization is interactive! Use your mouse to rotate, zoom, and explore.</p>
        </div>
    </body>
    </html>
    """
    
    with open(output_dir / "index.html", "w") as f:
        f.write(html)

if __name__ == "__main__":
    create_interactive_visualization()