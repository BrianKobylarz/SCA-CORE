"""Comprehensive export manager for semantic flow analysis results."""

import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import h5py
from datetime import datetime
import logging

from ..core.types import Word, Timestamp, SemanticFlow, FlowEvent
from ..core.base import BaseAnalyzer

class ExportManager(BaseAnalyzer):
    """Manages export of analysis results to various formats"""
    
    def __init__(self, config):
        super().__init__("ExportManager")
        self.config = config
        self.supported_formats = {
            'json': self._export_json,
            'csv': self._export_csv,
            'parquet': self._export_parquet,
            'hdf5': self._export_hdf5,
            'pickle': self._export_pickle,
            'excel': self._export_excel,
            'latex': self._export_latex,
            'html': self._export_html
        }
        
        # Export configuration
        self.export_config = {
            'timestamp_format': '%Y-%m-%d_%H-%M-%S',
            'include_metadata': True,
            'compress': True,
            'precision': 6,
            'index_columns': ['timestamp', 'word'],
            'chunk_size': 10000,
            'export_raw_data': True,
            'export_processed_data': True,
            'export_visualizations': True,
            'export_reports': True
        }
    
    def analyze(self) -> Dict[str, Any]:
        """Standard analyze method for BaseAnalyzer compatibility"""
        return {
            'supported_formats': list(self.supported_formats.keys()),
            'export_config': self.export_config,
            'export_history': getattr(self, 'export_history', [])
        }
    
    def export_analysis_results(self, results: Dict[str, Any], 
                              output_dir: str,
                              formats: List[str] = None,
                              prefix: str = "semantic_flow_analysis") -> Dict[str, str]:
        """Export complete analysis results in specified formats"""
        
        if formats is None:
            formats = ['json', 'csv', 'hdf5']
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime(self.export_config['timestamp_format'])
        export_info = {
            'timestamp': timestamp,
            'formats': formats,
            'files': {}
        }
        
        for format_name in formats:
            if format_name in self.supported_formats:
                try:
                    filename = f"{prefix}_{timestamp}.{format_name}"
                    filepath = output_path / filename
                    
                    export_func = self.supported_formats[format_name]
                    export_func(results, str(filepath))
                    
                    export_info['files'][format_name] = str(filepath)
                    logging.info(f"Exported results to {filepath}")
                    
                except Exception as e:
                    logging.error(f"Failed to export to {format_name}: {str(e)}")
                    export_info['files'][format_name] = f"Error: {str(e)}"
            else:
                logging.warning(f"Unsupported format: {format_name}")
        
        # Save export metadata
        metadata_file = output_path / f"{prefix}_{timestamp}_metadata.json"
        self._save_export_metadata(results, str(metadata_file), export_info)
        
        return export_info
    
    def _export_json(self, results: Dict[str, Any], filepath: str) -> None:
        """Export results to JSON format"""
        # Convert numpy arrays and other non-serializable objects
        serializable_results = self._make_json_serializable(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    def _make_json_serializable(self, obj: Any, seen=None) -> Any:
        """Convert object to JSON-serializable format"""
        if seen is None:
            seen = set()
        
        # Check for circular references
        obj_id = id(obj)
        if obj_id in seen:
            return str(obj)  # Return string representation to break cycle
        
        if isinstance(obj, dict):
            seen.add(obj_id)
            result = {}
            for key, value in obj.items():
                try:
                    result[key] = self._make_json_serializable(value, seen)
                except RecursionError:
                    result[key] = str(value)
            seen.discard(obj_id)
            return result
        elif isinstance(obj, list):
            return [self._make_json_serializable(item, seen) for item in obj]
        elif isinstance(obj, tuple):
            return list(self._make_json_serializable(item, seen) for item in obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, '__dict__') and not isinstance(obj, type):
            # Handle custom objects, but avoid circular references
            seen.add(obj_id)
            try:
                result = {
                    '_type': obj.__class__.__name__,
                    '_data': self._make_json_serializable(obj.__dict__, seen)
                }
            except RecursionError:
                result = {'_type': obj.__class__.__name__, '_str': str(obj)}
            seen.discard(obj_id)
            return result
        else:
            return str(obj) if not isinstance(obj, (str, int, float, bool, type(None))) else obj
    
    def _export_csv(self, results: Dict[str, Any], filepath: str) -> None:
        """Export results to CSV format"""
        # Extract tabular data from results
        tabular_data = self._extract_tabular_data(results)
        
        if tabular_data:
            # Export main data
            main_df = tabular_data.get('main', pd.DataFrame())
            if not main_df.empty:
                main_df.to_csv(filepath, index=False)
            
            # Export additional tables as separate files
            base_path = Path(filepath)
            for table_name, df in tabular_data.items():
                if table_name != 'main' and not df.empty:
                    table_filepath = base_path.parent / f"{base_path.stem}_{table_name}.csv"
                    df.to_csv(table_filepath, index=False)
        else:
            # Fallback: create simple summary CSV
            summary_data = self._create_summary_table(results)
            summary_data.to_csv(filepath, index=False)
    
    def _extract_tabular_data(self, results: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Extract tabular data from results for CSV export"""
        tabular_data = {}
        
        # Flow data
        if 'flows' in results:
            flows = results['flows']
            if isinstance(flows, list):
                flow_records = []
                for flow in flows:
                    if hasattr(flow, '__dict__'):
                        record = flow.__dict__.copy()
                        # Convert non-scalar fields
                        for key, value in record.items():
                            if isinstance(value, (list, tuple, np.ndarray)):
                                record[key] = str(value)
                        flow_records.append(record)
                
                if flow_records:
                    tabular_data['flows'] = pd.DataFrame(flow_records)
        
        # Metrics data
        if 'metrics' in results:
            metrics = results['metrics']
            if isinstance(metrics, dict):
                metrics_records = []
                for timestamp, timestamp_metrics in metrics.items():
                    if isinstance(timestamp_metrics, dict):
                        record = {'timestamp': timestamp}
                        record.update(timestamp_metrics)
                        metrics_records.append(record)
                
                if metrics_records:
                    tabular_data['metrics'] = pd.DataFrame(metrics_records)
        
        # Community evolution data
        if 'community_evolution' in results:
            community_data = results['community_evolution']
            if isinstance(community_data, dict):
                # Extract lineages
                if 'lineages' in community_data:
                    lineage_records = []
                    for lineage_id, lineage in community_data['lineages'].items():
                        record = {'lineage_id': lineage_id}
                        record.update(lineage)
                        # Convert lists to strings
                        for key, value in record.items():
                            if isinstance(value, (list, set)):
                                record[key] = ';'.join(map(str, value))
                        lineage_records.append(record)
                    
                    if lineage_records:
                        tabular_data['lineages'] = pd.DataFrame(lineage_records)
        
        # Analogy results
        if 'analogy_results' in results:
            analogy_data = results['analogy_results']
            if isinstance(analogy_data, dict):
                analogy_records = []
                for analogy_name, analogy_result in analogy_data.items():
                    if isinstance(analogy_result, dict):
                        record = {'analogy': analogy_name}
                        
                        # Extract key metrics
                        if 'validation_metrics' in analogy_result:
                            record.update(analogy_result['validation_metrics'])
                        
                        if 'analogy_metrics' in analogy_result:
                            record.update(analogy_result['analogy_metrics'])
                        
                        analogy_records.append(record)
                
                if analogy_records:
                    tabular_data['analogies'] = pd.DataFrame(analogy_records)
        
        return tabular_data
    
    def _create_summary_table(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create summary table from results"""
        summary_records = []
        
        def extract_summary_info(data, prefix=''):
            if isinstance(data, dict):
                for key, value in data.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    
                    if isinstance(value, (int, float, str, bool)):
                        summary_records.append({
                            'metric': full_key,
                            'value': value,
                            'type': type(value).__name__
                        })
                    elif isinstance(value, (list, tuple)) and len(value) > 0:
                        if isinstance(value[0], (int, float)):
                            summary_records.append({
                                'metric': f"{full_key}.mean",
                                'value': np.mean(value),
                                'type': 'float'
                            })
                            summary_records.append({
                                'metric': f"{full_key}.std",
                                'value': np.std(value),
                                'type': 'float'
                            })
                    elif isinstance(value, dict):
                        extract_summary_info(value, full_key)
        
        extract_summary_info(results)
        
        return pd.DataFrame(summary_records)
    
    def _export_parquet(self, results: Dict[str, Any], filepath: str) -> None:
        """Export results to Parquet format"""
        tabular_data = self._extract_tabular_data(results)
        
        if tabular_data:
            # Export main table
            main_df = tabular_data.get('main', pd.DataFrame())
            if not main_df.empty:
                main_df.to_parquet(filepath, index=False)
            
            # Export additional tables
            base_path = Path(filepath)
            for table_name, df in tabular_data.items():
                if table_name != 'main' and not df.empty:
                    table_filepath = base_path.parent / f"{base_path.stem}_{table_name}.parquet"
                    df.to_parquet(table_filepath, index=False)
    
    def _export_hdf5(self, results: Dict[str, Any], filepath: str) -> None:
        """Export results to HDF5 format"""
        with h5py.File(filepath, 'w') as f:
            self._write_hdf5_group(f, results, '')
    
    def _write_hdf5_group(self, group, data: Any, path: str) -> None:
        """Recursively write data to HDF5 group"""
        if isinstance(data, dict):
            for key, value in data.items():
                safe_key = self._make_hdf5_safe_key(key)
                
                if isinstance(value, (dict, list, tuple)):
                    subgroup = group.create_group(safe_key)
                    self._write_hdf5_group(subgroup, value, f"{path}/{safe_key}")
                else:
                    try:
                        if isinstance(value, (int, float, bool, str)):
                            group.attrs[safe_key] = value
                        elif isinstance(value, np.ndarray):
                            group.create_dataset(safe_key, data=value)
                        elif value is not None:
                            # Convert to string for unsupported types
                            group.attrs[safe_key] = str(value)
                    except Exception as e:
                        logging.warning(f"Could not write {path}/{safe_key}: {e}")
        
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                item_key = f"item_{i:06d}"
                
                if isinstance(item, (dict, list, tuple)):
                    subgroup = group.create_group(item_key)
                    self._write_hdf5_group(subgroup, item, f"{path}/{item_key}")
                else:
                    try:
                        if isinstance(item, (int, float, bool, str)):
                            group.attrs[item_key] = item
                        elif isinstance(item, np.ndarray):
                            group.create_dataset(item_key, data=item)
                        elif item is not None:
                            group.attrs[item_key] = str(item)
                    except Exception as e:
                        logging.warning(f"Could not write {path}/{item_key}: {e}")
    
    def _make_hdf5_safe_key(self, key: str) -> str:
        """Make key safe for HDF5"""
        # Replace problematic characters
        safe_key = str(key).replace('/', '_').replace(' ', '_').replace('.', '_')
        
        # Ensure it starts with a letter or underscore
        if safe_key and not (safe_key[0].isalpha() or safe_key[0] == '_'):
            safe_key = f"key_{safe_key}"
        
        return safe_key or "empty_key"
    
    def _export_pickle(self, results: Dict[str, Any], filepath: str) -> None:
        """Export results to Pickle format"""
        with open(filepath, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _export_excel(self, results: Dict[str, Any], filepath: str) -> None:
        """Export results to Excel format"""
        tabular_data = self._extract_tabular_data(results)
        
        if tabular_data:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                for sheet_name, df in tabular_data.items():
                    if not df.empty:
                        # Truncate sheet name to Excel limit
                        safe_sheet_name = sheet_name[:31]
                        df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
        else:
            # Fallback to summary
            summary_df = self._create_summary_table(results)
            summary_df.to_excel(filepath, index=False)
    
    def _export_latex(self, results: Dict[str, Any], filepath: str) -> None:
        """Export results to LaTeX format"""
        latex_content = self._generate_latex_report(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(latex_content)
    
    def _generate_latex_report(self, results: Dict[str, Any]) -> str:
        """Generate LaTeX report from results"""
        latex = []
        
        # Document header
        latex.append(r"\documentclass{article}")
        latex.append(r"\usepackage{amsmath}")
        latex.append(r"\usepackage{amsfonts}")
        latex.append(r"\usepackage{booktabs}")
        latex.append(r"\usepackage{longtable}")
        latex.append(r"\usepackage{graphicx}")
        latex.append(r"\title{Semantic Flow Analysis Report}")
        latex.append(r"\date{\today}")
        latex.append(r"\begin{document}")
        latex.append(r"\maketitle")
        
        # Summary section
        latex.append(r"\section{Analysis Summary}")
        
        if 'summary' in results:
            summary = results['summary']
            if isinstance(summary, dict):
                latex.append(r"\begin{itemize}")
                for key, value in summary.items():
                    safe_key = str(key).replace('_', r'\_')
                    safe_value = str(value).replace('_', r'\_').replace('%', r'\%')
                    latex.append(f"\\item {safe_key}: {safe_value}")
                latex.append(r"\end{itemize}")
        
        # Metrics section
        if 'metrics' in results:
            latex.append(r"\section{Key Metrics}")
            latex.append(r"\begin{table}[h]")
            latex.append(r"\centering")
            latex.append(r"\begin{tabular}{ll}")
            latex.append(r"\toprule")
            latex.append(r"Metric & Value \\")
            latex.append(r"\midrule")
            
            metrics = results['metrics']
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        safe_key = str(key).replace('_', r'\_')
                        formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                        latex.append(f"{safe_key} & {formatted_value} \\\\")
            
            latex.append(r"\bottomrule")
            latex.append(r"\end{tabular}")
            latex.append(r"\caption{Analysis Metrics}")
            latex.append(r"\end{table}")
        
        # Analogy results section
        if 'analogy_results' in results:
            latex.append(r"\section{Theoretical Analogy Results}")
            
            analogy_results = results['analogy_results']
            if isinstance(analogy_results, dict):
                for analogy_name, analogy_data in analogy_results.items():
                    safe_name = analogy_name.replace('_', r'\_')
                    latex.append(f"\\subsection{{{safe_name}}}")
                    
                    if isinstance(analogy_data, dict) and 'regime' in analogy_data:
                        regime = str(analogy_data['regime']).replace('_', r'\_')
                        latex.append(f"Regime: {regime}")
        
        # Document footer
        latex.append(r"\end{document}")
        
        return '\n'.join(latex)
    
    def _export_html(self, results: Dict[str, Any], filepath: str) -> None:
        """Export results to HTML format"""
        html_content = self._generate_html_report(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report from results"""
        html = []
        
        # HTML header
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("<title>Semantic Flow Analysis Report</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 40px; }")
        html.append("table { border-collapse: collapse; width: 100%; }")
        html.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("th { background-color: #f2f2f2; }")
        html.append("h1, h2, h3 { color: #333; }")
        html.append(".metric-value { font-weight: bold; color: #007acc; }")
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        
        # Title
        html.append("<h1>Semantic Flow Analysis Report</h1>")
        html.append(f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        # Summary section
        if 'summary' in results:
            html.append("<h2>Analysis Summary</h2>")
            summary = results['summary']
            if isinstance(summary, dict):
                html.append("<ul>")
                for key, value in summary.items():
                    html.append(f"<li><strong>{key}:</strong> <span class='metric-value'>{value}</span></li>")
                html.append("</ul>")
        
        # Metrics section
        if 'metrics' in results:
            html.append("<h2>Key Metrics</h2>")
            html.append("<table>")
            html.append("<tr><th>Metric</th><th>Value</th></tr>")
            
            metrics = results['metrics']
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                        html.append(f"<tr><td>{key}</td><td class='metric-value'>{formatted_value}</td></tr>")
            
            html.append("</table>")
        
        # Tabular data
        tabular_data = self._extract_tabular_data(results)
        for table_name, df in tabular_data.items():
            if not df.empty:
                html.append(f"<h2>{table_name.title()} Data</h2>")
                html.append(df.to_html(index=False, classes="data-table"))
        
        # HTML footer
        html.append("</body>")
        html.append("</html>")
        
        return '\n'.join(html)
    
    def _save_export_metadata(self, results: Dict[str, Any], filepath: str, 
                            export_info: Dict[str, Any]) -> None:
        """Save metadata about the export"""
        metadata = {
            'export_info': export_info,
            'data_summary': {
                'total_size': len(str(results)),
                'main_sections': list(results.keys()) if isinstance(results, dict) else [],
                'export_timestamp': datetime.now().isoformat(),
                'config': self.export_config
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def export_flows(self, flows: List[SemanticFlow], filepath: str, 
                    format_type: str = 'csv') -> None:
        """Export semantic flows to specified format"""
        if format_type == 'csv':
            flow_records = []
            for flow in flows:
                record = {
                    'timestamp': flow.timestamp,
                    'source_word': flow.source_word,
                    'target_word': flow.target_word,
                    'magnitude': flow.magnitude,
                    'direction': flow.direction,
                    'confidence': flow.confidence
                }
                flow_records.append(record)
            
            df = pd.DataFrame(flow_records)
            df.to_csv(filepath, index=False)
        
        elif format_type == 'json':
            flow_data = []
            for flow in flows:
                flow_data.append({
                    'timestamp': flow.timestamp,
                    'source_word': flow.source_word,
                    'target_word': flow.target_word,
                    'magnitude': flow.magnitude,
                    'direction': flow.direction.tolist() if hasattr(flow.direction, 'tolist') else flow.direction,
                    'confidence': flow.confidence
                })
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(flow_data, f, indent=2)
    
    def export_metrics(self, metrics: Dict[str, Any], filepath: str, 
                      format_type: str = 'csv') -> None:
        """Export metrics to specified format"""
        if format_type == 'csv':
            metrics_records = []
            
            def flatten_metrics(data, prefix=''):
                for key, value in data.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    
                    if isinstance(value, dict):
                        flatten_metrics(value, full_key)
                    elif isinstance(value, (int, float, str, bool)):
                        metrics_records.append({
                            'metric': full_key,
                            'value': value,
                            'type': type(value).__name__
                        })
            
            flatten_metrics(metrics)
            
            df = pd.DataFrame(metrics_records)
            df.to_csv(filepath, index=False)
        
        elif format_type == 'json':
            serializable_metrics = self._make_json_serializable(metrics)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_metrics, f, indent=2)
    
    def export_visualizations(self, viz_data: Dict[str, Any], output_dir: str) -> Dict[str, str]:
        """Export visualization data and configurations"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        for viz_name, viz_config in viz_data.items():
            if isinstance(viz_config, dict):
                # Export configuration
                config_file = output_path / f"{viz_name}_config.json"
                serializable_config = self._make_json_serializable(viz_config)
                
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_config, f, indent=2)
                
                exported_files[f"{viz_name}_config"] = str(config_file)
                
                # Export data if present
                if 'data' in viz_config:
                    data_file = output_path / f"{viz_name}_data.json"
                    data = self._make_json_serializable(viz_config['data'])
                    
                    with open(data_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)
                    
                    exported_files[f"{viz_name}_data"] = str(data_file)
        
        return exported_files
    
    def import_results(self, filepath: str, format_type: str = None) -> Dict[str, Any]:
        """Import previously exported results"""
        if format_type is None:
            format_type = Path(filepath).suffix[1:]  # Remove the dot
        
        if format_type == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        elif format_type == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        
        elif format_type == 'hdf5':
            return self._import_hdf5(filepath)
        
        elif format_type in ['csv', 'parquet', 'excel']:
            return self._import_tabular(filepath, format_type)
        
        else:
            raise ValueError(f"Unsupported import format: {format_type}")
    
    def _import_hdf5(self, filepath: str) -> Dict[str, Any]:
        """Import data from HDF5 file"""
        results = {}
        
        with h5py.File(filepath, 'r') as f:
            results = self._read_hdf5_group(f)
        
        return results
    
    def _read_hdf5_group(self, group) -> Dict[str, Any]:
        """Recursively read HDF5 group"""
        data = {}
        
        # Read attributes
        for key, value in group.attrs.items():
            data[key] = value
        
        # Read datasets
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset):
                data[key] = item[()]
            elif isinstance(item, h5py.Group):
                data[key] = self._read_hdf5_group(item)
        
        return data
    
    def _import_tabular(self, filepath: str, format_type: str) -> Dict[str, Any]:
        """Import tabular data"""
        if format_type == 'csv':
            df = pd.read_csv(filepath)
        elif format_type == 'parquet':
            df = pd.read_parquet(filepath)
        elif format_type == 'excel':
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported tabular format: {format_type}")
        
        return {'imported_data': df.to_dict('records')}
    
    def get_export_summary(self) -> Dict[str, Any]:
        """Get summary of export capabilities"""
        return {
            'supported_formats': list(self.supported_formats.keys()),
            'export_config': self.export_config,
            'format_descriptions': {
                'json': 'Human-readable structured data',
                'csv': 'Comma-separated values for spreadsheets',
                'parquet': 'Efficient columnar storage',
                'hdf5': 'Hierarchical scientific data format',
                'pickle': 'Python binary serialization',
                'excel': 'Microsoft Excel workbook',
                'latex': 'LaTeX document for academic papers',
                'html': 'Web-ready HTML report'
            }
        }