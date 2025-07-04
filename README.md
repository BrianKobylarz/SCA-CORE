# Semantic Flow Analyzer

A computational framework that bridges natural language processing with complexity science to analyze how word meanings flow, evolve, and cascade through communities over time.

## Overview

The Semantic Flow Analyzer (SFA) transforms diachronic embeddings into living maps of semantic evolution. Built specifically for sparse, event-driven data (like Reddit), it reveals patterns from individual word trajectories to system-wide phase transitions.

## Core Features

### 🌊 Flow-Based Analysis
- **Semantic Flows**: Track how meanings flow through semantic space
- **Flow Coherence**: Measure directional vs. dispersed semantic change
- **Flow Cascades**: Detect viral spread of meaning changes

### 🔬 Complexity Science Metrics
- **Cascade Risk (R₀)**: Predict semantic contagion potential
- **Path Dependency**: Measure momentum in semantic drift
- **Stochasticity**: Quantify volatility in word stability
- **Phase Transitions**: Detect system-wide reorganizations

### 📊 Multi-Scale Visualization
- **Animated Flow Networks**: 3D visualization of semantic flows
- **Four-Layer Analysis**: Word → Community → System → Meta-system
- **Interactive Dashboards**: Real-time exploration of semantic dynamics

### 💡 Theoretical Analogies
- **Epidemic Models**: Contagion dynamics of semantic change
- **Ferromagnetic Systems**: Phase transitions in meaning communities
- **Evolutionary Dynamics**: Selection and drift in semantic space
- **Opinion Dynamics**: Bounded confidence and polarization

## Installation

```bash
# Clone the repository
git clone https://github.com/semanticflow/semantic-flow-analyzer.git
cd semantic-flow-analyzer

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

```python
from sfa import SemanticFlowAnalyzer
from sfa.io import RedditLoader

# Load Reddit data
loader = RedditLoader(subreddit="politics", timespan="2020-2024")
embeddings = loader.load_diachronic_embeddings()

# Initialize analyzer
analyzer = SemanticFlowAnalyzer(embeddings)

# Analyze semantic flows
flows = analyzer.track_word_flows("democracy")
bursts = analyzer.detect_semantic_bursts()
cascades = analyzer.analyze_cascade_potential()

# Generate insights
report = analyzer.generate_report("democracy")
print(report.summary)
```

## Architecture

```
semantic_flow_analyzer/
├── config/          # Configuration management
├── sfa/
│   ├── core/        # Base classes and types
│   ├── metrics/     # Flow and complexity metrics
│   ├── dynamics/    # Flow tracking and event detection
│   ├── sparse/      # Sparse data handling
│   ├── analogies/   # Scientific analogies
│   ├── visualization/ # Interactive visualizations
│   ├── io/          # Data loading and export
│   └── analysis/    # High-level analysis orchestration
```

## Key Concepts

### Semantic Flow
Rather than treating semantic change as discrete shifts, SFA models meaning as flowing through high-dimensional space, creating currents, eddies, and cascades.

### Sparse Data Resilience
Designed for Reddit's bursty nature through:
- Adaptive temporal windows
- Dynamic neighborhood expansion
- Confidence-aware interpolation

### Multi-Scale Analysis
Four analytical layers:
1. **Word Level**: Individual trajectories and flows
2. **Community Level**: Local semantic neighborhoods
3. **System Level**: Global network properties
4. **Meta-System Level**: Cross-temporal patterns

## Scientific Analogies

### Epidemic Models
- **R₀ Calculation**: Predict semantic contagion
- **Superspreader Detection**: Identify influential words
- **Intervention Strategies**: Manage semantic change

### Phase Transitions
- **Order Parameters**: Measure system coherence
- **Critical Points**: Detect reorganization events
- **Universality Classes**: Compare across domains

### Evolutionary Dynamics
- **Semantic Selection**: Track meaning competition
- **Drift Patterns**: Quantify random change
- **Fitness Landscapes**: Map semantic advantages

## Research Applications

- **Political Discourse**: Track polarization dynamics
- **Cultural Evolution**: Monitor meme propagation
- **Crisis Communication**: Analyze meaning stability
- **Market Intelligence**: Predict semantic trends
- **Social Movements**: Understand narrative evolution

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use SFA in your research, please cite:

```bibtex
@software{semantic_flow_analyzer,
  title={Semantic Flow Analyzer: A Complexity Science Approach to Meaning Evolution},
  author={Semantic Flow Research Team},
  year={2024},
  url={https://github.com/semanticflow/semantic-flow-analyzer}
}
```

## Support

- **Documentation**: [https://sfa-docs.readthedocs.io](https://sfa-docs.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/semanticflow/semantic-flow-analyzer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/semanticflow/semantic-flow-analyzer/discussions)

---

*Bridging the gap between computational linguistics and complex systems science.*