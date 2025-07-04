#!/usr/bin/env python3
"""
Streamlit Dashboard Launcher for Semantic Flow Analyzer

Launch this script with: streamlit run examples/dashboard_launcher.py
"""

import sys
from pathlib import Path
import numpy as np

# Add the semantic flow analyzer to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import streamlit as st
    from sfa import (
        SemanticFlowAnalyzer, TemporalEmbeddingStore, FlowConfig,
        StreamlitDashboard
    )
    from sfa.visualization.dashboard import StreamlitDashboard
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all dependencies are installed")
    st.stop()

@st.cache_data
def create_sample_data():
    """Create sample data for dashboard demonstration"""
    
    # Sample vocabulary focused on technology and AI
    vocabulary = [
        'artificial', 'intelligence', 'machine', 'learning', 'deep', 'neural',
        'network', 'algorithm', 'data', 'science', 'analytics', 'model',
        'automation', 'robotics', 'computer', 'vision', 'processing', 'language',
        'technology', 'innovation', 'digital', 'transformation', 'platform',
        'cloud', 'computing', 'software', 'development', 'programming',
        'startup', 'business', 'market', 'industry', 'economy', 'growth',
        'research', 'scientific', 'discovery', 'breakthrough', 'advancement'
    ]
    
    # Time periods
    timestamps = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06']
    
    # Create embedding store
    embedding_store = TemporalEmbeddingStore(embedding_dim=128)
    
    # Generate embeddings with realistic evolution
    for i, timestamp in enumerate(timestamps):
        for j, word in enumerate(vocabulary):
            # Base embedding
            base_embedding = np.random.randn(128)
            base_embedding = base_embedding / np.linalg.norm(base_embedding)
            
            # Add temporal evolution
            drift = 0.05 * i * np.random.randn(128)
            evolved_embedding = base_embedding + drift
            
            # Add semantic clustering
            if any(tech_word in word for tech_word in ['artificial', 'machine', 'neural', 'algorithm']):
                # AI cluster
                ai_center = np.array([1.0, 0.5] + [0.0] * 126)
                evolved_embedding = 0.8 * evolved_embedding + 0.2 * ai_center
            elif any(biz_word in word for biz_word in ['business', 'market', 'startup', 'economy']):
                # Business cluster
                biz_center = np.array([0.0, 1.0, 0.5] + [0.0] * 125)
                evolved_embedding = 0.8 * evolved_embedding + 0.2 * biz_center
            elif any(research_word in word for research_word in ['research', 'scientific', 'discovery']):
                # Research cluster
                research_center = np.array([0.5, 0.0, 1.0] + [0.0] * 125)
                evolved_embedding = 0.8 * evolved_embedding + 0.2 * research_center
            
            evolved_embedding = evolved_embedding / np.linalg.norm(evolved_embedding)
            embedding_store.store_embedding(word, timestamp, evolved_embedding)
    
    return embedding_store, timestamps

@st.cache_data
def run_analysis(embedding_store, timestamps):
    """Run semantic flow analysis and cache results"""
    
    config = FlowConfig(
        default_k_neighbors=20,
        flow_similarity_threshold=0.25,
        burst_z_threshold=2.5,
        cascade_risk_threshold=0.8
    )
    
    # Initialize analyzer
    analyzer = SemanticFlowAnalyzer(embedding_store, config)
    
    # Run analysis
    with st.spinner("Running semantic flow analysis..."):
        results = analyzer.analyze_complete_timeline(
            focus_words=['artificial', 'intelligence', 'machine', 'learning'],
            compute_umap=True,
            save_results=False
        )
    
    return results, analyzer, config

def main():
    """Main dashboard application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Semantic Flow Analysis Dashboard",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🌊 Semantic Flow Analysis Dashboard")
    st.markdown("Interactive exploration of semantic evolution and dynamics")
    st.markdown("---")
    
    # Sidebar for data loading
    with st.sidebar:
        st.header("🔧 Dashboard Controls")
        
        # Data source selection
        data_source = st.selectbox(
            "Data Source",
            ["Sample Data (Demo)", "Upload Embeddings", "Connect to Database"],
            index=0
        )
        
        if data_source == "Sample Data (Demo)":
            st.info("Using AI/Technology themed sample data for demonstration")
            
            # Load sample data
            try:
                embedding_store, timestamps = create_sample_data()
                st.success(f"✅ Loaded sample data: {len(timestamps)} time periods")
                
                # Run analysis
                analysis_results, analyzer, config = run_analysis(embedding_store, timestamps)
                st.success("✅ Analysis completed")
                
                # Initialize dashboard
                dashboard = StreamlitDashboard(analyzer, config)
                
                # Add analysis results to dashboard
                dashboard.analysis_results = analysis_results
                
                # Run dashboard interface
                dashboard.run_dashboard(analysis_results)
                
            except Exception as e:
                st.error(f"❌ Error loading sample data: {e}")
                st.error("Please check the console for detailed error information")
                
        elif data_source == "Upload Embeddings":
            st.info("📁 Upload your own embedding files")
            
            uploaded_file = st.file_uploader(
                "Choose embedding file",
                type=['json', 'csv', 'pickle'],
                help="Upload embeddings in JSON, CSV, or pickle format"
            )
            
            if uploaded_file is not None:
                st.info("File upload functionality would be implemented here")
                
        else:  # Database connection
            st.info("🔗 Database connection")
            
            db_host = st.text_input("Database Host", "localhost")
            db_port = st.number_input("Port", value=5432)
            db_name = st.text_input("Database Name")
            
            if st.button("Connect"):
                st.info("Database connection functionality would be implemented here")
    
    # Help section
    with st.expander("ℹ️ Help & Information", expanded=False):
        st.markdown("""
        ### 🎯 Dashboard Features
        
        **📊 Overview Tab**
        - System-wide metrics and health indicators
        - Temporal trends and evolution patterns
        - Quick statistical summaries
        
        **🌐 Network View**
        - Interactive 3D semantic network visualization
        - Customizable node and edge attributes
        - Real-time filtering and exploration
        
        **📈 Temporal Analysis**
        - Animated evolution of semantic flows
        - Word trajectory tracking
        - Community formation and dissolution
        
        **🔬 Analogies**
        - Theoretical model insights (Epidemic, Ferromagnetic, Evolutionary, etc.)
        - Model parameter interpretation
        - Predictive analytics
        
        **🎯 Insights**
        - AI-powered recommendations
        - Risk assessment and early warnings
        - Intervention strategies
        
        ### 🚀 Getting Started
        
        1. **Select Data Source**: Choose sample data or upload your own
        2. **Configure Filters**: Use sidebar controls to customize analysis
        3. **Explore Tabs**: Navigate through different analysis views
        4. **Export Results**: Download insights and visualizations
        
        ### 🔧 Technical Details
        
        This dashboard is powered by the Semantic Flow Analyzer system, which combines:
        - **NLP Techniques**: Embedding analysis and semantic similarity
        - **Complexity Science**: Phase transitions, critical phenomena, network dynamics
        - **Theoretical Models**: Epidemic dynamics, ferromagnetism, evolutionary theory
        - **Interactive Visualization**: Real-time exploration and animation
        
        ### 📚 Learn More
        
        - See the complete example: `examples/complete_analysis_example.py`
        - Read the documentation for detailed API reference
        - Explore the theoretical foundations in the research papers
        """)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("🌊 **Semantic Flow Analyzer**")
        st.markdown("Complexity science meets NLP")
    
    with col2:
        st.markdown("🔬 **Features**")
        st.markdown("Multi-analogy theoretical framework")
    
    with col3:
        st.markdown("📊 **Dashboard**")
        st.markdown(f"Last updated: {st.session_state.get('last_update', 'Never')}")

if __name__ == "__main__":
    main()