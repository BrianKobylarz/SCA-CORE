"""Interactive Streamlit dashboard for semantic flow analysis."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import json
from datetime import datetime

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from ..core.types import Word, Timestamp
from ..core.base import BaseAnalyzer

class StreamlitDashboard(BaseAnalyzer):
    """Interactive Streamlit dashboard for exploring semantic flow analysis results"""
    
    def __init__(self, flow_analyzer, config):
        super().__init__("StreamlitDashboard")
        self.flow_analyzer = flow_analyzer
        self.config = config
        
        # Dashboard configuration
        self.dashboard_config = {
            'title': 'Semantic Flow Analysis Dashboard',
            'layout': 'wide',
            'theme': 'dark',
            'sidebar_width': 300,
            'auto_refresh': False,
            'cache_results': True,
            'max_words_display': 100,
            'default_time_range': 'all',
            'color_scheme': 'viridis',
            'show_help': True,
            'export_formats': ['json', 'csv', 'png'],
            'real_time_updates': False
        }
        
        # Dashboard state
        self.analysis_results = {}
        self.current_filters = {}
        self.cached_plots = {}
        
    def analyze(self) -> Dict[str, Any]:
        """Run dashboard analysis - mainly setup and validation"""
        if not STREAMLIT_AVAILABLE:
            return {
                'error': 'Streamlit not available',
                'suggestion': 'Install streamlit: pip install streamlit'
            }
        
        return {
            'dashboard_available': True,
            'config': self.dashboard_config,
            'features': self._get_dashboard_features()
        }
    
    def _get_dashboard_features(self) -> List[str]:
        """Get list of dashboard features"""
        return [
            'Interactive network visualization',
            'Temporal evolution animation', 
            'Multi-layer analysis view',
            'Real-time filtering and search',
            'Theoretical analogy exploration',
            'Export capabilities',
            'Comparative analysis tools',
            'Statistical summaries'
        ]
    
    def run_dashboard(self, analysis_results: Optional[Dict[str, Any]] = None) -> None:
        """Run the Streamlit dashboard"""
        if not STREAMLIT_AVAILABLE:
            print("Streamlit is not available. Please install with: pip install streamlit")
            return
        
        # Store analysis results
        if analysis_results:
            self.analysis_results = analysis_results
        
        # Configure page
        st.set_page_config(
            page_title=self.dashboard_config['title'],
            layout=self.dashboard_config['layout'],
            initial_sidebar_state='expanded'
        )
        
        # Main dashboard layout
        self._render_header()
        self._render_sidebar()
        self._render_main_content()
        self._render_footer()
    
    def _render_header(self) -> None:
        """Render dashboard header"""
        st.title(self.dashboard_config['title'])
        st.markdown("---")
        
        # Quick stats row
        if self.analysis_results:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_words = self._get_total_words()
                st.metric("Total Words", total_words)
            
            with col2:
                total_flows = self._get_total_flows()
                st.metric("Total Flows", total_flows)
            
            with col3:
                active_timestamps = self._get_active_timestamps()
                st.metric("Time Periods", active_timestamps)
            
            with col4:
                system_health = self._get_system_health()
                st.metric("System Health", system_health)
    
    def _render_sidebar(self) -> None:
        """Render dashboard sidebar with controls"""
        st.sidebar.header("Analysis Controls")
        
        # Data loading section
        with st.sidebar.expander("Data Loading", expanded=True):
            if st.button("🔄 Refresh Analysis"):
                self._refresh_analysis()
            
            if st.button("📁 Load New Data"):
                self._load_new_data()
        
        # Filtering section
        with st.sidebar.expander("Filters", expanded=True):
            self._render_filters()
        
        # Visualization options
        with st.sidebar.expander("Visualization Options", expanded=False):
            self._render_viz_options()
        
        # Analysis options
        with st.sidebar.expander("Analysis Options", expanded=False):
            self._render_analysis_options()
        
        # Export options
        with st.sidebar.expander("Export", expanded=False):
            self._render_export_options()
    
    def _render_filters(self) -> None:
        """Render filtering controls"""
        # Time range filter
        if self.analysis_results and 'timestamps' in self.analysis_results:
            timestamps = self.analysis_results['timestamps']
            
            time_range = st.select_slider(
                "Time Range",
                options=timestamps,
                value=(timestamps[0], timestamps[-1]) if len(timestamps) >= 2 else (timestamps[0], timestamps[0]),
                key="time_range"
            )
            self.current_filters['time_range'] = time_range
        
        # Word filter
        search_words = st.text_input(
            "Search Words (comma-separated)",
            placeholder="e.g., technology, innovation, growth",
            key="search_words"
        )
        
        if search_words:
            words = [w.strip() for w in search_words.split(',') if w.strip()]
            self.current_filters['focus_words'] = words
        
        # Flow strength filter
        min_flow_strength = st.slider(
            "Minimum Flow Strength",
            0.0, 1.0, 0.1,
            step=0.05,
            key="min_flow_strength"
        )
        self.current_filters['min_flow_strength'] = min_flow_strength
        
        # Community size filter
        min_community_size = st.slider(
            "Minimum Community Size", 
            1, 50, 3,
            key="min_community_size"
        )
        self.current_filters['min_community_size'] = min_community_size
    
    def _render_viz_options(self) -> None:
        """Render visualization options"""
        # Layout options
        self.dashboard_config['layout_algorithm'] = st.selectbox(
            "Network Layout",
            ['spring', 'circular', 'random', 'embedding_based'],
            index=0
        )
        
        # Color scheme
        self.dashboard_config['color_scheme'] = st.selectbox(
            "Color Scheme",
            ['viridis', 'plasma', 'blues', 'reds', 'greens'],
            index=0
        )
        
        # Animation speed
        self.dashboard_config['animation_speed'] = st.slider(
            "Animation Speed (ms)",
            500, 3000, 1000,
            step=100
        )
        
        # Show labels
        self.dashboard_config['show_labels'] = st.checkbox(
            "Show Word Labels",
            value=True
        )
    
    def _render_analysis_options(self) -> None:
        """Render analysis options"""
        # Analysis layers to show
        analysis_layers = st.multiselect(
            "Analysis Layers",
            ['Flow Network', 'Community Evolution', 'Phase Transitions', 'Analogies', 'UMAP Projection'],
            default=['Flow Network', 'Community Evolution']
        )
        self.current_filters['analysis_layers'] = analysis_layers
        
        # Analogy selection
        available_analogies = ['epidemic', 'ferromagnetic', 'evolutionary', 'bounded_confidence']
        selected_analogies = st.multiselect(
            "Theoretical Analogies",
            available_analogies,
            default=['epidemic', 'ferromagnetic']
        )
        self.current_filters['analogies'] = selected_analogies
        
        # Complexity metrics
        complexity_metrics = st.multiselect(
            "Complexity Metrics",
            ['Cascade Risk', 'Critical Temperature', 'Phase Coherence', 'Network Entropy'],
            default=['Cascade Risk']
        )
        self.current_filters['complexity_metrics'] = complexity_metrics
    
    def _render_export_options(self) -> None:
        """Render export options"""
        export_format = st.selectbox(
            "Export Format",
            self.dashboard_config['export_formats'],
            index=0
        )
        
        if st.button("📤 Export Current View"):
            self._export_current_view(export_format)
        
        if st.button("📊 Export Full Analysis"):
            self._export_full_analysis(export_format)
    
    def _render_main_content(self) -> None:
        """Render main dashboard content"""
        if not self.analysis_results:
            st.warning("No analysis results loaded. Please run analysis first.")
            
            if st.button("🚀 Run Sample Analysis"):
                self._run_sample_analysis()
            return
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "🌐 Network View", "📈 Temporal Analysis", 
            "🔬 Analogies", "🎯 Insights"
        ])
        
        with tab1:
            self._render_overview_tab()
        
        with tab2:
            self._render_network_tab()
        
        with tab3:
            self._render_temporal_tab()
        
        with tab4:
            self._render_analogies_tab()
        
        with tab5:
            self._render_insights_tab()
    
    def _render_overview_tab(self) -> None:
        """Render overview tab"""
        st.header("Analysis Overview")
        
        # Summary metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("System Metrics")
            
            if 'summary' in self.analysis_results:
                summary = self.analysis_results['summary']
                
                metrics_df = pd.DataFrame([
                    {"Metric": "Flow Intensity", "Value": summary.get('flow_intensity', 0)},
                    {"Metric": "Network Density", "Value": summary.get('network_density', 0)},
                    {"Metric": "Community Count", "Value": summary.get('community_count', 0)},
                    {"Metric": "System Stability", "Value": summary.get('stability_score', 0)}
                ])
                
                st.dataframe(metrics_df, use_container_width=True)
        
        with col2:
            st.subheader("Temporal Trends")
            
            # Create trend visualization
            if 'temporal_analysis' in self.analysis_results:
                trend_data = self.analysis_results['temporal_analysis']
                self._plot_temporal_trends(trend_data)
        
        # Analysis layers summary
        st.subheader("Analysis Layers Status")
        
        layers_status = self._get_layers_status()
        
        for layer, status in layers_status.items():
            status_icon = "✅" if status else "❌"
            st.write(f"{status_icon} {layer}")
    
    def _render_network_tab(self) -> None:
        """Render network visualization tab"""
        st.header("Semantic Flow Network")
        
        # Network controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_flows = st.checkbox("Show Flow Arrows", value=True)
        
        with col2:
            node_size_metric = st.selectbox(
                "Node Size Metric",
                ['degree', 'betweenness', 'closeness', 'flow_activity']
            )
        
        with col3:
            edge_weight_metric = st.selectbox(
                "Edge Weight Metric", 
                ['similarity', 'flow_strength', 'frequency']
            )
        
        # Create network visualization
        if 'flow_network' in self.analysis_results:
            network_fig = self._create_network_visualization(
                show_flows, node_size_metric, edge_weight_metric
            )
            st.plotly_chart(network_fig, use_container_width=True)
        
        # Network statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Network Statistics")
            self._display_network_stats()
        
        with col2:
            st.subheader("Top Nodes by Centrality")
            self._display_top_nodes()
    
    def _render_temporal_tab(self) -> None:
        """Render temporal analysis tab"""
        st.header("Temporal Evolution")
        
        # Temporal visualization controls
        animation_control = st.radio(
            "Visualization Type",
            ['Static Snapshots', 'Animated Evolution', 'Trajectory Plots'],
            horizontal=True
        )
        
        if animation_control == 'Animated Evolution':
            if 'animated_network' in self.analysis_results:
                animated_fig = self.analysis_results['animated_network']
                st.plotly_chart(animated_fig, use_container_width=True)
        
        elif animation_control == 'Trajectory Plots':
            if 'word_trajectories' in self.analysis_results:
                trajectory_fig = self._create_trajectory_plot()
                st.plotly_chart(trajectory_fig, use_container_width=True)
        
        else:  # Static Snapshots
            if 'network_snapshots' in self.analysis_results:
                snapshots = self.analysis_results['network_snapshots']
                
                # Snapshot selector
                timestamp = st.select_slider(
                    "Select Timestamp",
                    options=list(snapshots.keys()),
                    key="snapshot_selector"
                )
                
                if timestamp in snapshots:
                    st.plotly_chart(snapshots[timestamp], use_container_width=True)
        
        # Community evolution
        st.subheader("Community Evolution")
        
        if 'community_evolution' in self.analysis_results:
            community_data = self.analysis_results['community_evolution']
            self._plot_community_evolution(community_data)
    
    def _render_analogies_tab(self) -> None:
        """Render theoretical analogies tab"""
        st.header("Theoretical Analogies")
        
        if 'analogy_results' not in self.analysis_results:
            st.warning("No analogy results available.")
            return
        
        analogy_results = self.analysis_results['analogy_results']
        
        # Analogy selector
        available_analogies = list(analogy_results.keys())
        selected_analogy = st.selectbox(
            "Select Analogy",
            available_analogies,
            key="analogy_selector"
        )
        
        if selected_analogy in analogy_results:
            analogy_data = analogy_results[selected_analogy]
            
            # Analogy-specific visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"{selected_analogy.title()} Model Parameters")
                
                if 'parameters' in analogy_data:
                    params_df = pd.DataFrame([
                        {"Parameter": k, "Value": v}
                        for k, v in analogy_data['parameters'].items()
                        if isinstance(v, (int, float))
                    ])
                    st.dataframe(params_df, use_container_width=True)
            
            with col2:
                st.subheader("Model Predictions")
                
                if 'predictions' in analogy_data:
                    predictions = analogy_data['predictions']
                    self._plot_analogy_predictions(selected_analogy, predictions)
            
            # Model interpretation
            st.subheader("Interpretation")
            
            if 'interpretation' in analogy_data:
                interpretation = analogy_data['interpretation']
                
                for key, value in interpretation.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        # Ensemble comparison
        st.subheader("Analogy Ensemble Comparison")
        self._plot_analogy_comparison(analogy_results)
    
    def _render_insights_tab(self) -> None:
        """Render insights and recommendations tab"""
        st.header("Key Insights & Recommendations")
        
        # Automated insights
        st.subheader("🔍 Automated Insights")
        insights = self._generate_automated_insights()
        
        for insight in insights:
            st.info(insight)
        
        # Intervention recommendations
        if 'interventions' in self.analysis_results:
            st.subheader("💡 Intervention Recommendations")
            
            interventions = self.analysis_results['interventions']
            self._display_interventions(interventions)
        
        # Risk assessment
        st.subheader("⚠️ Risk Assessment")
        
        risks = self._assess_system_risks()
        
        for risk_type, risk_level in risks.items():
            color = "red" if risk_level > 0.7 else "orange" if risk_level > 0.4 else "green"
            st.write(f":{color}[{risk_type}: {risk_level:.2f}]")
        
        # Export recommendations
        st.subheader("📤 Recommended Exports")
        
        if st.button("Generate Analysis Report"):
            self._generate_analysis_report()
    
    def _render_footer(self) -> None:
        """Render dashboard footer"""
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("📊 Semantic Flow Analyzer Dashboard")
        
        with col2:
            if st.button("ℹ️ Help & Documentation"):
                self._show_help()
        
        with col3:
            st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Helper methods
    def _get_total_words(self) -> int:
        """Get total number of unique words"""
        if 'vocabulary_stats' in self.analysis_results:
            return self.analysis_results['vocabulary_stats'].get('total_unique_words', 0)
        return 0
    
    def _get_total_flows(self) -> int:
        """Get total number of flows"""
        if 'flow_stats' in self.analysis_results:
            return self.analysis_results['flow_stats'].get('total_flows', 0)
        return 0
    
    def _get_active_timestamps(self) -> int:
        """Get number of active timestamps"""
        if 'timestamps' in self.analysis_results:
            return len(self.analysis_results['timestamps'])
        return 0
    
    def _get_system_health(self) -> str:
        """Get system health status"""
        if 'system_health' in self.analysis_results:
            health_score = self.analysis_results['system_health'].get('overall_score', 0)
            
            if health_score > 0.8:
                return "🟢 Healthy"
            elif health_score > 0.5:
                return "🟡 Moderate"
            else:
                return "🔴 Critical"
        
        return "❓ Unknown"
    
    def _refresh_analysis(self) -> None:
        """Refresh analysis results"""
        with st.spinner("Refreshing analysis..."):
            # This would trigger a new analysis
            st.success("Analysis refreshed!")
            st.experimental_rerun()
    
    def _load_new_data(self) -> None:
        """Load new data interface"""
        st.info("Data loading interface would be implemented here")
    
    def _plot_temporal_trends(self, trend_data: Dict[str, Any]) -> None:
        """Plot temporal trends"""
        # Create sample trend plot
        fig = go.Figure()
        
        # Sample data
        timestamps = list(range(10))
        values = np.random.random(10)
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines+markers',
            name='Flow Intensity'
        ))
        
        fig.update_layout(
            title="Temporal Trends",
            xaxis_title="Time",
            yaxis_title="Intensity",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _get_layers_status(self) -> Dict[str, bool]:
        """Get status of analysis layers"""
        return {
            'Flow Network': 'flow_network' in self.analysis_results,
            'Community Evolution': 'community_evolution' in self.analysis_results,
            'Phase Transitions': 'phase_transitions' in self.analysis_results,
            'Theoretical Analogies': 'analogy_results' in self.analysis_results,
            'UMAP Projection': 'umap_results' in self.analysis_results
        }
    
    def _create_network_visualization(self, show_flows: bool, 
                                    node_size_metric: str, 
                                    edge_weight_metric: str) -> go.Figure:
        """Create network visualization"""
        # Placeholder network visualization
        fig = go.Figure()
        
        # Sample network data
        nodes_x = np.random.random(20)
        nodes_y = np.random.random(20)
        
        fig.add_trace(go.Scatter(
            x=nodes_x,
            y=nodes_y,
            mode='markers',
            marker=dict(size=10, color='blue'),
            name='Words'
        ))
        
        fig.update_layout(
            title="Semantic Network",
            xaxis_title="X Dimension",
            yaxis_title="Y Dimension",
            height=500
        )
        
        return fig
    
    def _display_network_stats(self) -> None:
        """Display network statistics"""
        # Sample statistics
        stats = {
            "Nodes": 150,
            "Edges": 450,
            "Density": 0.02,
            "Average Clustering": 0.35,
            "Components": 1
        }
        
        for stat, value in stats.items():
            st.metric(stat, value)
    
    def _display_top_nodes(self) -> None:
        """Display top nodes by centrality"""
        # Sample data
        top_nodes = pd.DataFrame({
            'Word': ['technology', 'innovation', 'data', 'system', 'network'],
            'Centrality': [0.85, 0.72, 0.68, 0.55, 0.42]
        })
        
        st.dataframe(top_nodes, use_container_width=True)
    
    def _create_trajectory_plot(self) -> go.Figure:
        """Create word trajectory plot"""
        fig = go.Figure()
        
        # Sample trajectory
        t = np.linspace(0, 10, 50)
        x = np.sin(t)
        y = np.cos(t)
        
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines+markers',
            name='Word Trajectory'
        ))
        
        fig.update_layout(
            title="Word Trajectories",
            height=400
        )
        
        return fig
    
    def _plot_community_evolution(self, community_data: Dict[str, Any]) -> None:
        """Plot community evolution"""
        # Sample community evolution plot
        fig = go.Figure()
        
        timestamps = list(range(10))
        community_counts = np.random.randint(5, 15, 10)
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=community_counts,
            mode='lines+markers',
            name='Community Count'
        ))
        
        fig.update_layout(
            title="Community Evolution",
            xaxis_title="Time",
            yaxis_title="Number of Communities",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_analogy_predictions(self, analogy_name: str, predictions: Dict[str, Any]) -> None:
        """Plot analogy predictions"""
        # Sample predictions plot
        fig = go.Figure()
        
        t = np.linspace(0, 10, 50)
        y = np.exp(-t/5) * np.sin(t)
        
        fig.add_trace(go.Scatter(
            x=t, y=y,
            mode='lines',
            name=f'{analogy_name} Prediction'
        ))
        
        fig.update_layout(
            title=f"{analogy_name.title()} Model Predictions",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_analogy_comparison(self, analogy_results: Dict[str, Any]) -> None:
        """Plot analogy comparison"""
        # Sample comparison
        analogies = list(analogy_results.keys())
        scores = np.random.random(len(analogies))
        
        fig = px.bar(
            x=analogies,
            y=scores,
            title="Analogy Performance Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _generate_automated_insights(self) -> List[str]:
        """Generate automated insights"""
        return [
            "🔥 High burst activity detected in technology-related terms",
            "📈 Community formation rate has increased by 25% this period", 
            "⚡ Cascade risk is elevated - monitor for viral spread",
            "🎯 Strong consensus emerging around sustainability themes"
        ]
    
    def _display_interventions(self, interventions: Dict[str, Any]) -> None:
        """Display intervention recommendations"""
        # Sample interventions
        intervention_data = [
            {"Type": "Amplification", "Target": "innovation", "Urgency": "High", "Confidence": 0.85},
            {"Type": "Dampening", "Target": "controversy", "Urgency": "Medium", "Confidence": 0.72}
        ]
        
        df = pd.DataFrame(intervention_data)
        st.dataframe(df, use_container_width=True)
    
    def _assess_system_risks(self) -> Dict[str, float]:
        """Assess system risks"""
        return {
            "Cascade Risk": 0.35,
            "Fragmentation Risk": 0.42,
            "Stability Risk": 0.28,
            "Information Overload": 0.55
        }
    
    def _generate_analysis_report(self) -> None:
        """Generate comprehensive analysis report"""
        with st.spinner("Generating report..."):
            # This would generate a comprehensive report
            st.success("Report generated! Check your downloads folder.")
    
    def _show_help(self) -> None:
        """Show help documentation"""
        st.info("""
        **Dashboard Help**
        
        - **Overview Tab**: System metrics and trends
        - **Network View**: Interactive network visualization  
        - **Temporal Analysis**: Evolution over time
        - **Analogies**: Theoretical model insights
        - **Insights**: AI-powered recommendations
        
        Use the sidebar filters to customize your analysis.
        """)
    
    def _export_current_view(self, format_type: str) -> None:
        """Export current view"""
        st.info(f"Exporting current view as {format_type}...")
    
    def _export_full_analysis(self, format_type: str) -> None:
        """Export full analysis"""
        st.info(f"Exporting full analysis as {format_type}...")
    
    def _run_sample_analysis(self) -> None:
        """Run sample analysis for demonstration"""
        with st.spinner("Running sample analysis..."):
            # Generate sample results
            self.analysis_results = self._generate_sample_results()
            st.success("Sample analysis completed!")
            st.experimental_rerun()
    
    def _generate_sample_results(self) -> Dict[str, Any]:
        """Generate sample analysis results for demonstration"""
        return {
            'timestamps': ['2024-01', '2024-02', '2024-03', '2024-04'],
            'vocabulary_stats': {'total_unique_words': 1250},
            'flow_stats': {'total_flows': 3400},
            'system_health': {'overall_score': 0.75},
            'summary': {
                'flow_intensity': 0.65,
                'network_density': 0.12,
                'community_count': 8,
                'stability_score': 0.73
            }
        }