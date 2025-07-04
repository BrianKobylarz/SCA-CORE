"""FastAPI application for Semantic Flow Analyzer"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime

from ..core.embeddings import TemporalEmbeddingStore
from ..analysis.flow_analyzer import SemanticFlowAnalyzer
from ..config.flow_config import FlowConfig

class AnalysisRequest(BaseModel):
    """Request model for analysis endpoint"""
    focus_words: Optional[List[str]] = None
    compute_umap: bool = True
    config: Optional[Dict[str, Any]] = None

class AnalysisResponse(BaseModel):
    """Response model for analysis results"""
    status: str
    timestamp: str
    results: Dict[str, Any]
    summary: Dict[str, Any]

def create_app():
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="Semantic Flow Analyzer API",
        description="Analyze semantic evolution and dynamics in textual data",
        version="1.0.0"
    )
    
    # Enable CORS for web frontends
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Store for active sessions
    sessions = {}
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "name": "Semantic Flow Analyzer API",
            "version": "1.0.0",
            "endpoints": {
                "POST /upload": "Upload embeddings data",
                "POST /analyze": "Run semantic flow analysis",
                "GET /results/{session_id}": "Get analysis results",
                "GET /visualize/{session_id}/{chart_type}": "Get visualization data"
            }
        }
    
    @app.post("/upload")
    async def upload_embeddings(file: UploadFile = File(...)):
        """Upload embeddings data"""
        # Implementation for handling file uploads
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Process uploaded file and create embedding store
        # This is a placeholder - implement actual file processing
        
        return {
            "session_id": session_id,
            "status": "success",
            "message": "Embeddings uploaded successfully"
        }
    
    @app.post("/analyze")
    async def analyze(request: AnalysisRequest):
        """Run semantic flow analysis"""
        try:
            # Create sample data for demo
            # In production, this would use uploaded data
            embedding_store = create_sample_embeddings()
            
            # Configure analysis
            config = FlowConfig()
            if request.config:
                for key, value in request.config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            
            # Run analysis
            analyzer = SemanticFlowAnalyzer(embedding_store, config)
            results = analyzer.analyze_complete_timeline(
                focus_words=request.focus_words,
                compute_umap=request.compute_umap,
                save_results=False
            )
            
            # Create response
            response = AnalysisResponse(
                status="success",
                timestamp=datetime.now().isoformat(),
                results=results,
                summary=results.get('summary', {})
            )
            
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/visualize/{session_id}/{chart_type}")
    async def get_visualization(session_id: str, chart_type: str):
        """Get visualization data for specific chart type"""
        # Return plot data in JSON format for frontend rendering
        visualization_data = {
            "chart_type": chart_type,
            "data": {},
            "layout": {},
            "config": {}
        }
        
        return JSONResponse(content=visualization_data)
    
    return app

def create_sample_embeddings():
    """Create sample embeddings for demo"""
    # Implementation from complete_analysis_example.py
    embedding_store = TemporalEmbeddingStore(embedding_dim=128)
    # ... (implement sample data creation)
    return embedding_store