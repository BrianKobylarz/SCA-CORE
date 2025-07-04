"""Adaptive temporal windowing for sparse Reddit data."""

import numpy as np
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from ..core.types import Timestamp, Word

class AdaptiveTemporalWindows:
    """Handles adaptive windowing for sparse Reddit data"""
    
    def __init__(self, embeddings_store, config):
        self.embeddings = embeddings_store
        self.config = config
        self.activity_cache = {}
    
    def compute_adaptive_windows(self, 
                               base_window_days: int = 7,
                               min_window_days: int = 1,
                               max_window_days: int = 30) -> List[Tuple[Timestamp, Timestamp]]:
        """Compute variable-sized windows based on semantic activity"""
        
        # First, measure activity levels
        activity_profile = self._compute_activity_profile()
        
        # Identify high-activity periods (events)
        event_periods = self._detect_event_periods(activity_profile)
        
        # Generate adaptive windows
        windows = []
        timestamps = self.embeddings.get_timestamps()
        
        i = 0
        while i < len(timestamps):
            # Check if we're in an event period
            current_time = timestamps[i]
            
            if any(start <= current_time <= end for start, end in event_periods):
                # Use minimum window during events
                window_size = min_window_days
            else:
                # Adjust window based on local activity
                local_activity = activity_profile.get(current_time, 0)
                
                if local_activity < self.config.activity_threshold:
                    # Low activity: use larger window
                    window_size = max_window_days
                else:
                    # Normal activity
                    window_size = base_window_days
            
            # Find window end
            j = i + 1
            while j < len(timestamps) and self._days_between(timestamps[i], timestamps[j]) < window_size:
                j += 1
            
            if j > i + 1:  # Valid window
                windows.append((timestamps[i], timestamps[j-1]))
            
            i = j - 1 if j > i + 1 else i + 1
        
        return windows
    
    def _compute_activity_profile(self) -> Dict[Timestamp, float]:
        """Compute semantic activity level at each timestamp"""
        activity = {}
        timestamps = self.embeddings.get_timestamps()
        
        for i in range(len(timestamps) - 1):
            t1, t2 = timestamps[i], timestamps[i+1]
            
            # Measure vocabulary change
            vocab_t1 = set(self.embeddings.get_vocabulary(t1))
            vocab_t2 = set(self.embeddings.get_vocabulary(t2))
            
            # New words appearing
            new_words = len(vocab_t2 - vocab_t1)
            
            # Words disappearing  
            lost_words = len(vocab_t1 - vocab_t2)
            
            # Average semantic shift
            common_words = vocab_t1 & vocab_t2
            
            if common_words:
                shifts = []
                for word in list(common_words)[:100]:  # Sample
                    emb1 = self.embeddings.get_embedding(word, t1)
                    emb2 = self.embeddings.get_embedding(word, t2)
                    
                    if emb1 is not None and emb2 is not None:
                        shift = np.linalg.norm(emb2 - emb1)
                        shifts.append(shift)
                
                avg_shift = np.mean(shifts) if shifts else 0
            else:
                avg_shift = 0
            
            # Combined activity score
            activity[t2] = (
                0.3 * (new_words / max(len(vocab_t2), 1)) +
                0.3 * (lost_words / max(len(vocab_t1), 1)) +
                0.4 * avg_shift
            )
        
        return activity
    
    def _detect_event_periods(self, 
                            activity_profile: Dict[Timestamp, float],
                            z_threshold: float = 2.0) -> List[Tuple[Timestamp, Timestamp]]:
        """Detect periods of unusually high activity"""
        activities = list(activity_profile.values())
        
        if len(activities) < 3:
            return []
        
        mean_activity = np.mean(activities)
        std_activity = np.std(activities)
        
        if std_activity == 0:
            return []
        
        event_periods = []
        timestamps = sorted(activity_profile.keys())
        
        in_event = False
        event_start = None
        
        for timestamp, activity in activity_profile.items():
            z_score = (activity - mean_activity) / std_activity
            
            if z_score > z_threshold and not in_event:
                # Event starts
                in_event = True
                event_start = timestamp
            elif z_score <= z_threshold and in_event:
                # Event ends
                in_event = False
                if event_start:
                    event_periods.append((event_start, timestamp))
                    event_start = None
        
        # Close final event if needed
        if in_event and event_start:
            event_periods.append((event_start, timestamps[-1]))
        
        return event_periods
    
    def interpolate_sparse_embeddings(self, 
                                    word: Word,
                                    target_timestamps: List[Timestamp]) -> Dict[Timestamp, np.ndarray]:
        """Interpolate embeddings for sparse data"""
        available_embeddings = {}
        
        # Collect available embeddings
        for t in self.embeddings.get_timestamps():
            emb = self.embeddings.get_embedding(word, t)
            if emb is not None:
                available_embeddings[t] = emb
        
        if len(available_embeddings) < 2:
            return available_embeddings
        
        interpolated = {}
        sorted_available = sorted(available_embeddings.items())
        
        for target_t in target_timestamps:
            if target_t in available_embeddings:
                interpolated[target_t] = available_embeddings[target_t]
            else:
                # Find surrounding timestamps
                before = None
                after = None
                
                for t, emb in sorted_available:
                    if t < target_t:
                        before = (t, emb)
                    elif t > target_t and after is None:
                        after = (t, emb)
                        break
                
                if before and after:
                    # Linear interpolation
                    t1, emb1 = before
                    t2, emb2 = after
                    
                    # Compute interpolation weight
                    total_gap = self._timestamp_diff(t1, t2)
                    partial_gap = self._timestamp_diff(t1, target_t)
                    
                    if total_gap > 0:
                        alpha = partial_gap / total_gap
                        interpolated[target_t] = (1 - alpha) * emb1 + alpha * emb2
                elif before:
                    # Extrapolate from last known
                    interpolated[target_t] = before[1]
        
        return interpolated
    
    def compute_confidence_bands(self, 
                               word: Word,
                               window: Tuple[Timestamp, Timestamp]) -> Dict[str, float]:
        """Compute confidence bands for sparse data"""
        # Count data points in window
        data_points = 0
        for t in self.embeddings.get_timestamps():
            if window[0] <= t <= window[1]:
                if self.embeddings.get_embedding(word, t) is not None:
                    data_points += 1
        
        # Compute window size
        window_days = self._days_between(window[0], window[1])
        
        # Confidence based on data density
        density = data_points / max(window_days, 1)
        
        # Confidence levels
        if density > 0.7:
            confidence = 0.95
            band_width = 0.1
        elif density > 0.3:
            confidence = 0.80
            band_width = 0.2
        else:
            confidence = 0.60
            band_width = 0.3
        
        return {
            'confidence_level': confidence,
            'band_width': band_width,
            'data_density': density,
            'data_points': data_points
        }
    
    def _days_between(self, t1: Timestamp, t2: Timestamp) -> int:
        """Calculate days between timestamps"""
        # Assuming timestamps are in YYYY-MM-DD format
        try:
            date1 = datetime.strptime(t1, "%Y-%m-%d")
            date2 = datetime.strptime(t2, "%Y-%m-%d")
            return abs((date2 - date1).days)
        except:
            # Fallback: assume sequential indices
            return 1
    
    def _timestamp_diff(self, t1: Timestamp, t2: Timestamp) -> float:
        """Compute numeric difference between timestamps"""
        try:
            date1 = datetime.strptime(t1, "%Y-%m-%d")
            date2 = datetime.strptime(t2, "%Y-%m-%d")
            return (date2 - date1).total_seconds()
        except:
            # Fallback
            return 1.0