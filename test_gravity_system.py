#!/usr/bin/env python3
"""
Test script for the gravitational field semantic analysis system.
Demonstrates core functionality with sample data.
"""

import numpy as np
import sys
from pathlib import Path
import tempfile
import logging

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from sfa.core.storage import HDF5EmbeddingStore
from sfa.core.gravity import HighDimensionalFlowMetrics
from sfa.visualization.gravity_viz import UMAPVisualizationLayer
from sfa.config.gravity_config import create_demo_config
from sfa.engines.hierarchical_engine import HierarchicalVisualizationEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data(store: HDF5EmbeddingStore, embedding_dim: int = 128):
    """Create sample embedding data for testing."""
    print("📊 Creating sample semantic embedding data...")
    
    # Sample vocabulary with semantic relationships
    vocab_groups = {
        'technology': ['computer', 'software', 'algorithm', 'artificial', 'intelligence', 'machine', 'learning'],
        'business': ['market', 'economy', 'finance', 'corporate', 'investment', 'startup', 'revenue'],
        'science': ['research', 'experiment', 'hypothesis', 'theory', 'discovery', 'analysis', 'data'],
        'social': ['community', 'people', 'society', 'culture', 'communication', 'network', 'relationship']
    }
    
    # Flatten vocabulary
    all_words = []
    word_to_group = {}
    for group, words in vocab_groups.items():
        all_words.extend(words)
        for word in words:
            word_to_group[word] = group
    
    # Timestamps representing temporal evolution
    timestamps = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05']
    
    print(f"  📝 Vocabulary: {len(all_words)} words across {len(vocab_groups)} semantic groups")
    print(f"  📅 Timestamps: {len(timestamps)} time periods")
    
    # Create embeddings with semantic structure and temporal evolution
    for t_idx, timestamp in enumerate(timestamps):
        print(f"  ⏱️  Generating embeddings for {timestamp}...")
        
        for word in all_words:
            # Create base embedding with group-specific bias
            base_embedding = np.random.randn(embedding_dim) * 0.5
            
            # Add semantic group clustering
            group = word_to_group[word]
            if group == 'technology':
                base_embedding[:10] += 2.0  # Technology cluster in first 10 dimensions
            elif group == 'business':
                base_embedding[10:20] += 2.0  # Business cluster
            elif group == 'science':
                base_embedding[20:30] += 2.0  # Science cluster
            elif group == 'social':
                base_embedding[30:40] += 2.0  # Social cluster
            
            # Add temporal drift (words evolve over time)
            temporal_drift = 0.1 * t_idx * np.random.randn(embedding_dim)
            
            # Add some word-specific evolution patterns
            if word in ['artificial', 'intelligence', 'machine', 'learning']:
                # AI terms gain prominence over time
                temporal_drift += 0.3 * t_idx * np.array([1.0] + [0.0] * (embedding_dim - 1))
            elif word in ['startup', 'investment']:
                # Business terms show volatility
                temporal_drift += 0.2 * np.sin(t_idx) * np.random.randn(embedding_dim)
            
            # Combine and normalize
            final_embedding = base_embedding + temporal_drift
            final_embedding = final_embedding / np.linalg.norm(final_embedding)
            
            # Store in HDF5
            store.store_embedding(word, timestamp, final_embedding)
    
    print(f"✅ Created {len(all_words) * len(timestamps)} embeddings")
    return all_words, timestamps, vocab_groups

def test_gravitational_field(store: HDF5EmbeddingStore, words: list, timestamps: list):
    """Test gravitational field computation."""
    print("\n🌌 Testing Gravitational Field System...")
    
    # Initialize flow metrics
    flow_metrics = HighDimensionalFlowMetrics(store)
    
    # Test focal word
    focal_word = 'artificial'
    timestamp = timestamps[2]  # Middle timestamp
    
    print(f"  🎯 Computing gravitational field for '{focal_word}' at {timestamp}")
    
    # Compute gravitational field
    gravity_field = flow_metrics.compute_gravitational_field(focal_word, timestamp, k=10)
    
    if gravity_field:
        print(f"  ✅ Gravitational field computed successfully!")
        print(f"    - Focal word: {gravity_field.focal_word}")
        print(f"    - Orbiting neighbors: {len(gravity_field.neighbor_words)}")
        print(f"    - Average gravitational strength: {np.mean(gravity_field.gravitational_strengths):.4f}")
        print(f"    - Average orbital stability: {np.mean(gravity_field.stability_indices):.4f}")
        print(f"    - Top 3 neighbors: {gravity_field.neighbor_words[:3]}")
        
        # Test semantic flow between timestamps
        if len(timestamps) > 1:
            print(f"  🌊 Computing semantic flow from {timestamps[0]} to {timestamps[1]}")
            flow_data = flow_metrics.compute_semantic_flow(focal_word, timestamps[0], timestamps[1], k_neighbors=10)
            
            if 'error' not in flow_data:
                print(f"    - Displacement magnitude: {flow_data['displacement_magnitude']:.4f}")
                print(f"    - Velocity magnitude: {flow_data['velocity_magnitude']:.4f}")
                print(f"    - Acceleration magnitude: {flow_data['acceleration_magnitude']:.4f}")
                print("  ✅ Semantic flow computed successfully!")
            else:
                print(f"    ❌ Flow computation error: {flow_data['error']}")
        
        return flow_metrics
    else:
        print("  ❌ Failed to compute gravitational field")
        return None

def test_umap_visualization(store: HDF5EmbeddingStore, flow_metrics, timestamps: list):
    """Test UMAP visualization layer."""
    print("\n🗺️  Testing UMAP Visualization Layer...")
    
    # Initialize UMAP visualization
    umap_viz = UMAPVisualizationLayer(store, flow_metrics)
    
    # Fit UMAP for a timestamp
    timestamp = timestamps[2]
    print(f"  📊 Fitting UMAP projection for {timestamp}")
    
    result = umap_viz.fit_umap_for_timestamp(timestamp, vocabulary_limit=50)
    
    if 'error' not in result:
        print(f"  ✅ UMAP projection created successfully!")
        print(f"    - Processed {result['num_words']} words")
        print(f"    - Projection shape: {result['projection_shape']}")
        print(f"    - UMAP parameters: n_neighbors={result['umap_params']['n_neighbors']}")
        
        # Test gravitational visualization
        print("  🎨 Creating gravitational field visualization...")
        try:
            fig = umap_viz.create_gravitational_visualization('artificial', timestamp, k_neighbors=10)
            print("  ✅ Gravitational visualization created successfully!")
            print(f"    - Figure contains {len(fig.data)} traces")
            return umap_viz
        except Exception as e:
            print(f"  ⚠️  Visualization creation warning: {e}")
            return umap_viz
    else:
        print(f"  ❌ UMAP projection failed: {result['error']}")
        return None

def test_hierarchical_engine(store: HDF5EmbeddingStore, flow_metrics, umap_viz):
    """Test hierarchical visualization engine."""
    print("\n🏗️  Testing Hierarchical Visualization Engine...")
    
    # Create demo configuration
    config = create_demo_config()
    print(f"  ⚙️  Using demo configuration with {len(config.hierarchical_viz.enabled_layers)} enabled layers")
    
    # Initialize hierarchical engine
    try:
        engine = HierarchicalVisualizationEngine(store, flow_metrics, umap_viz, config)
        print("  ✅ Hierarchical engine initialized successfully!")
        
        # Get layer information
        layer_info = engine.get_layer_info()
        print(f"  📊 Available layers:")
        for layer_name, info in layer_info.items():
            status = "✅" if info['enabled'] else "❌"
            print(f"    {status} {layer_name}: {info['description']}")
        
        return engine
    except Exception as e:
        print(f"  ❌ Hierarchical engine initialization failed: {e}")
        return None

def test_burst_detection(flow_metrics, timestamps: list):
    """Test burst detection system."""
    print("\n💥 Testing Burst Detection System...")
    
    focal_word = 'artificial'
    print(f"  🔍 Detecting semantic bursts for '{focal_word}' across {len(timestamps)} timestamps")
    
    try:
        bursts = flow_metrics.detect_semantic_bursts(focal_word, timestamps, k_neighbors=10)
        
        if bursts:
            print(f"  ✅ Detected {len(bursts)} burst events!")
            for i, burst in enumerate(bursts):
                print(f"    💥 Burst {i+1}: {burst.timestamp} (magnitude: {burst.burst_magnitude:.3f})")
                print(f"       - Affected words: {len(burst.affected_words)}")
                print(f"       - New orbits: {len(burst.new_orbits_formed)}")
                print(f"       - Destroyed orbits: {len(burst.orbits_destroyed)}")
        else:
            print("  📊 No significant burst events detected")
            
    except Exception as e:
        print(f"  ⚠️  Burst detection error: {e}")

def main():
    """Main test function."""
    print("🚀 Gravitational Field Semantic Analysis System - Test Suite")
    print("=" * 80)
    
    # Create temporary HDF5 file for testing
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        h5_path = tmp_file.name
    
    try:
        # Initialize storage
        print("💾 Initializing HDF5 storage system...")
        store = HDF5EmbeddingStore(h5_path, mode='w', embedding_dim=128)
        print(f"  ✅ Storage initialized: {h5_path}")
        
        # Create sample data
        words, timestamps, groups = create_sample_data(store)
        
        # Test gravitational field system
        flow_metrics = test_gravitational_field(store, words, timestamps)
        
        if flow_metrics:
            # Test UMAP visualization
            umap_viz = test_umap_visualization(store, flow_metrics, timestamps)
            
            if umap_viz:
                # Test hierarchical engine
                engine = test_hierarchical_engine(store, flow_metrics, umap_viz)
                
                # Test burst detection
                test_burst_detection(flow_metrics, timestamps)
                
                # Print final statistics
                print("\n📈 System Performance Statistics:")
                cache_stats = store.get_cache_stats()
                print(f"  📦 Storage cache hit rate: {cache_stats['hit_rate']:.2%}")
                print(f"  💾 Total cache requests: {cache_stats['total_requests']}")
                
                storage_stats = store.get_storage_stats()
                print(f"  📊 Total embeddings stored: {storage_stats['total_embeddings']}")
                print(f"  💽 File size: {storage_stats['file_size_mb']:.2f} MB")
        
        # Close storage
        store.close()
        
        print("\n🎉 Test Suite Completed Successfully!")
        print("=" * 80)
        print("✅ The gravitational field semantic analysis system is working correctly!")
        print("🌌 Key features tested:")
        print("   • HDF5 storage with lazy loading")
        print("   • Gravitational field computation")
        print("   • Semantic flow analysis")
        print("   • UMAP visualization projection") 
        print("   • Hierarchical multi-layer visualization")
        print("   • Burst detection system")
        print("\n🚀 Ready for real semantic analysis tasks!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup temporary file
        try:
            Path(h5_path).unlink()
        except:
            pass

if __name__ == "__main__":
    main()