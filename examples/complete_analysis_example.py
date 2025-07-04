#!/usr/bin/env python3
"""
Complete Semantic Flow Analysis Example

This example demonstrates the full capabilities of the Semantic Flow Analyzer system,
including data loading, comprehensive analysis, theoretical analogies, and visualization.
"""

import numpy as np
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add the semantic flow analyzer to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the semantic flow analyzer
try:
    from sfa import (
        SemanticFlowAnalyzer, TemporalEmbeddingStore, FlowConfig, 
        RedditDataLoader, ExportManager, AnalogyEnsemble,
        VisualizationLayerManager, StreamlitDashboard
    )
    print("✅ Successfully imported Semantic Flow Analyzer")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure the semantic flow analyzer is properly installed")
    sys.exit(1)

def create_sample_embeddings():
    """Create sample embedding data for demonstration"""
    print("📊 Creating sample embedding data...")
    
    # Sample vocabulary
    vocabulary = [
        'technology', 'innovation', 'artificial', 'intelligence', 'machine', 'learning',
        'data', 'science', 'neural', 'network', 'algorithm', 'model', 'deep', 'analysis',
        'automation', 'robotics', 'future', 'digital', 'computer', 'software',
        'development', 'programming', 'code', 'application', 'system', 'platform',
        'internet', 'web', 'online', 'cloud', 'computing', 'database', 'server',
        'security', 'privacy', 'encryption', 'blockchain', 'cryptocurrency',
        'startup', 'business', 'economy', 'market', 'industry', 'company',
        'research', 'scientific', 'discovery', 'breakthrough', 'advancement'
    ]
    
    # Sample timestamps
    timestamps = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05']
    
    # Create temporal embedding store
    embedding_store = TemporalEmbeddingStore(embedding_dim=128)
    
    # Generate sample embeddings with temporal evolution
    for i, timestamp in enumerate(timestamps):
        print(f"  📅 Generating embeddings for {timestamp}")
        
        for j, word in enumerate(vocabulary):
            # Create base embedding
            base_embedding = np.random.randn(128)
            base_embedding = base_embedding / np.linalg.norm(base_embedding)
            
            # Add temporal drift
            drift = 0.1 * i * np.random.randn(128)
            evolved_embedding = base_embedding + drift
            evolved_embedding = evolved_embedding / np.linalg.norm(evolved_embedding)
            
            # Add some semantic clustering
            if 'tech' in word or 'data' in word or 'computer' in word:
                # Technology cluster
                tech_center = np.array([1.0] + [0.0] * 127)
                evolved_embedding = 0.7 * evolved_embedding + 0.3 * tech_center
            elif 'business' in word or 'market' in word or 'economy' in word:
                # Business cluster  
                business_center = np.array([0.0, 1.0] + [0.0] * 126)
                evolved_embedding = 0.7 * evolved_embedding + 0.3 * business_center
            elif 'research' in word or 'science' in word or 'discovery' in word:
                # Research cluster
                research_center = np.array([0.0, 0.0, 1.0] + [0.0] * 125)
                evolved_embedding = 0.7 * evolved_embedding + 0.3 * research_center
            
            evolved_embedding = evolved_embedding / np.linalg.norm(evolved_embedding)
            
            # Store embedding
            embedding_store.store_embedding(word, timestamp, evolved_embedding)
    
    print(f"✅ Created embeddings for {len(vocabulary)} words across {len(timestamps)} timestamps")
    return embedding_store

def run_complete_analysis():
    """Run complete semantic flow analysis"""
    print("\n🚀 Starting Complete Semantic Flow Analysis")
    print("=" * 60)
    
    # Step 1: Create sample data
    print("\n📋 Step 1: Data Preparation")
    embedding_store = create_sample_embeddings()
    
    # Step 2: Configure analysis
    print("\n⚙️ Step 2: Configuration")
    config = FlowConfig(
        default_k_neighbors=25,
        flow_similarity_threshold=0.3,
        burst_z_threshold=3.0,
        cascade_risk_threshold=1.0
    )
    print(f"  ✅ Flow configuration: k={config.default_k_neighbors}, threshold={config.flow_similarity_threshold}")
    
    # Step 3: Initialize analyzer
    print("\n🔬 Step 3: Initialize Analyzer")
    analyzer = SemanticFlowAnalyzer(embedding_store, config)
    print("  ✅ Semantic Flow Analyzer initialized")
    
    # Step 4: Run comprehensive analysis
    print("\n📊 Step 4: Running Comprehensive Analysis")
    print("  This may take a few moments...")
    
    try:
        results = analyzer.analyze_complete_timeline(
            focus_words=['technology', 'innovation', 'artificial', 'intelligence'],
            compute_umap=True,
            save_results=False
        )
        print("  ✅ Analysis completed successfully!")
        
        # Display key results
        print("\n📈 Analysis Results Summary:")
        
        if 'summary' in results:
            summary = results['summary']
            print(f"  🔹 Flow Events Detected: {summary.get('total_flow_events', 0)}")
            print(f"  🔹 Burst Events: {summary.get('burst_events', 0)}")
            print(f"  🔹 Communities Tracked: {summary.get('active_communities', 0)}")
            print(f"  🔹 Phase Transitions: {summary.get('phase_transitions', 0)}")
        
        if 'flow_analysis' in results:
            flow_analysis = results['flow_analysis']
            if 'summary' in flow_analysis:
                flow_summary = flow_analysis['summary']
                print(f"  🔹 Total Flows: {flow_summary.get('total_flows', 0)}")
                print(f"  🔹 Average Flow Strength: {flow_summary.get('average_flow_strength', 0):.3f}")
        
        # Step 5: Analogy Analysis
        print("\n🧠 Step 5: Theoretical Analogy Analysis")
        
        if 'analogy_ensemble' in results:
            analogy_results = results['analogy_ensemble']
            
            if 'individual_results' in analogy_results:
                individual_results = analogy_results['individual_results']
                
                print("  📊 Analogy Results:")
                for analogy_name, analogy_data in individual_results.items():
                    if 'regime' in analogy_data:
                        regime = analogy_data['regime']
                        print(f"    🔸 {analogy_name.title()}: {regime}")
                    
                    if 'validation_metrics' in analogy_data:
                        validation = analogy_data['validation_metrics']
                        goodness = validation.get('goodness_of_fit', 0)
                        confidence = validation.get('confidence', 0)
                        print(f"      Fit Quality: {goodness:.3f}, Confidence: {confidence:.3f}")
        
        # Step 6: Risk Assessment
        print("\n⚠️ Step 6: Risk Assessment")
        
        if 'cascade_analysis' in results:
            cascade_data = results['cascade_analysis']
            cascade_risk = cascade_data.get('cascade_risk', 0)
            r0 = cascade_data.get('r0', 1.0)
            
            print(f"  🔹 Cascade Risk: {cascade_risk:.3f}")
            print(f"  🔹 Basic Reproduction Number (R₀): {r0:.3f}")
            
            if r0 > 1.0:
                print("  ⚠️  WARNING: R₀ > 1, system in supercritical regime")
            else:
                print("  ✅ System in subcritical regime")
        
        # Step 7: Export Results
        print("\n📤 Step 7: Export Results")
        
        export_manager = ExportManager(config)
        export_info = export_manager.export_analysis_results(
            results,
            output_dir="./analysis_output",
            formats=['json', 'csv'],
            prefix="sample_analysis"
        )
        
        print(f"  ✅ Results exported to:")
        for format_name, filepath in export_info['files'].items():
            print(f"    📄 {format_name.upper()}: {filepath}")
        
        # Step 8: Visualization (if available)
        print("\n🎨 Step 8: Visualization")
        
        try:
            viz_manager = VisualizationLayerManager(analyzer, config)
            viz_results = viz_manager.analyze()
            
            if viz_results and 'layer_visualizations' in viz_results:
                print("  ✅ Visualizations generated:")
                for layer_name in viz_results['layer_visualizations'].keys():
                    print(f"    🎯 {layer_name}")
            
        except Exception as e:
            print(f"  ⚠️  Visualization generation failed: {e}")
        
        # Step 9: Dashboard (if Streamlit available)
        print("\n📱 Step 9: Interactive Dashboard")
        
        try:
            dashboard = StreamlitDashboard(analyzer, config)
            dashboard_info = dashboard.analyze()
            
            if dashboard_info.get('dashboard_available', False):
                print("  ✅ Dashboard available")
                print("  💡 To launch dashboard, run:")
                print("     streamlit run examples/dashboard_launcher.py")
            else:
                print("  ⚠️  Dashboard not available (Streamlit not installed)")
                
        except Exception as e:
            print(f"  ⚠️  Dashboard initialization failed: {e}")
        
        print("\n🎉 Analysis Complete!")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def demonstrate_individual_components():
    """Demonstrate individual system components"""
    print("\n🧩 Demonstrating Individual Components")
    print("=" * 50)
    
    # Create sample data
    embedding_store = create_sample_embeddings()
    config = FlowConfig()
    
    # 1. Flow Tracking
    print("\n1. 🌊 Flow Tracking")
    try:
        from sfa.dynamics import SemanticFlowTracker
        
        flow_tracker = SemanticFlowTracker(embedding_store, config)
        flows = flow_tracker.track_flows_for_timeline()
        
        print(f"   ✅ Tracked {len(flows)} semantic flows")
        
        if flows:
            avg_magnitude = np.mean([flow.magnitude for flow in flows])
            print(f"   📊 Average flow magnitude: {avg_magnitude:.3f}")
            
    except Exception as e:
        print(f"   ❌ Flow tracking failed: {e}")
    
    # 2. Burst Detection
    print("\n2. 💥 Burst Detection")
    try:
        from sfa.dynamics import BurstDetector
        
        burst_detector = BurstDetector(embedding_store, config)
        burst_results = burst_detector.analyze()
        
        if 'burst_events' in burst_results:
            burst_count = len(burst_results['burst_events'])
            print(f"   ✅ Detected {burst_count} burst events")
        
    except Exception as e:
        print(f"   ❌ Burst detection failed: {e}")
    
    # 3. Community Evolution
    print("\n3. 👥 Community Evolution")
    try:
        from sfa.dynamics import CommunityEvolutionTracker
        
        community_tracker = CommunityEvolutionTracker(embedding_store, config)
        community_results = community_tracker.analyze()
        
        if 'summary_statistics' in community_results:
            stats = community_results['summary_statistics']
            total_lineages = stats.get('total_lineages', 0)
            active_lineages = stats.get('active_lineages', 0)
            
            print(f"   ✅ Tracked {total_lineages} community lineages")
            print(f"   📈 Active communities: {active_lineages}")
        
    except Exception as e:
        print(f"   ❌ Community tracking failed: {e}")
    
    # 4. Theoretical Analogies
    print("\n4. 🧠 Theoretical Analogies")
    try:
        from sfa.analogies import EpidemicAnalogy, FerromagneticAnalogy
        
        # Epidemic analogy
        epidemic_analogy = EpidemicAnalogy(embedding_store, config)
        epidemic_results = epidemic_analogy.analyze()
        
        if 'regime' in epidemic_results:
            regime = epidemic_results['regime']
            print(f"   🦠 Epidemic regime: {regime}")
        
        # Ferromagnetic analogy
        ferro_analogy = FerromagneticAnalogy(embedding_store, config)
        ferro_results = ferro_analogy.analyze()
        
        if 'regime' in ferro_results:
            regime = ferro_results['regime']
            print(f"   🧲 Magnetic regime: {regime}")
        
    except Exception as e:
        print(f"   ❌ Analogy analysis failed: {e}")
    
    print("\n✅ Component demonstration complete!")

def show_system_capabilities():
    """Show system capabilities and features"""
    print("\n🎯 System Capabilities Overview")
    print("=" * 50)
    
    capabilities = {
        "🌊 Semantic Flow Analysis": [
            "Track word meaning evolution over time",
            "Detect semantic shifts and drift",
            "Quantify flow magnitudes and directions"
        ],
        "💥 Event Detection": [
            "Burst detection with statistical significance",
            "Cascade analysis and risk assessment", 
            "Phase transition identification"
        ],
        "👥 Community Dynamics": [
            "Community formation and dissolution",
            "Lineage tracking (births, deaths, splits, merges)",
            "Community evolution patterns"
        ],
        "🧠 Theoretical Analogies": [
            "Epidemic dynamics (SIR/SEIR models)",
            "Ferromagnetic phase transitions (Ising model)",
            "Evolutionary dynamics (population genetics)",
            "Bounded confidence opinion dynamics",
            "Multi-analogy ensemble integration"
        ],
        "📊 Complexity Science": [
            "Critical phenomena detection",
            "Scaling law analysis", 
            "Network topology evolution",
            "Early warning signals"
        ],
        "🎨 Visualization": [
            "3D animated flow networks",
            "UMAP dimensionality reduction",
            "Interactive Streamlit dashboards",
            "4-layer hierarchical views"
        ],
        "💾 Data Management": [
            "Sparse data handling for Reddit/social media",
            "Adaptive temporal windowing",
            "Intelligent caching and storage",
            "Multi-format export (JSON, CSV, HDF5, LaTeX)"
        ],
        "🤖 AI-Powered Features": [
            "Automated intervention recommendations",
            "Event correlation analysis",
            "Uncertainty quantification",
            "Predictive modeling"
        ]
    }
    
    for category, features in capabilities.items():
        print(f"\n{category}")
        for feature in features:
            print(f"  ✅ {feature}")
    
    print(f"\n📋 Total: {sum(len(features) for features in capabilities.values())} capabilities across {len(capabilities)} categories")

def main():
    """Main example function"""
    print("🎭 Semantic Flow Analyzer - Complete Example")
    print("=" * 60)
    print("This example demonstrates the full capabilities of the")
    print("Semantic Flow Analyzer system for complexity science-based")
    print("analysis of semantic evolution in textual data.")
    print("=" * 60)
    
    # Show system capabilities
    show_system_capabilities()
    
    # Demonstrate individual components
    demonstrate_individual_components()
    
    # Run complete analysis
    results = run_complete_analysis()
    
    if results:
        print("\n🎊 Example completed successfully!")
        print("\n📚 Next Steps:")
        print("  1. Explore the generated output files")
        print("  2. Try the interactive dashboard with:")
        print("     streamlit run examples/dashboard_launcher.py")
        print("  3. Modify the configuration for your specific use case")
        print("  4. Load your own embedding data")
        print("  5. Explore the theoretical analogy insights")
        
    else:
        print("\n😞 Example encountered errors. Please check the output above.")
    
    print("\n📖 For more information, see the documentation and README.")

if __name__ == "__main__":
    main()