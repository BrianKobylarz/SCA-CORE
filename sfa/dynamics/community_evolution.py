"""Community evolution tracking for semantic networks."""

import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

from ..core.types import Word, Timestamp, CommunityLineage
from ..core.base import BaseAnalyzer

class CommunityEvolutionTracker(BaseAnalyzer):
    """Tracks evolution and dynamics of semantic communities"""
    
    def __init__(self, embeddings_store, config):
        super().__init__("CommunityEvolutionTracker")
        self.embeddings = embeddings_store
        self.config = config
        
        # Community tracking
        self.communities_history = {}  # timestamp -> communities
        self.lineages = {}  # lineage_id -> CommunityLineage
        self.lineage_counter = 0
    
    def analyze(self, method: str = 'louvain') -> Dict[str, any]:
        """Analyze community evolution across all timestamps"""
        # Detect communities at each timestamp
        self._detect_communities_timeline(method)
        
        # Track community lineages
        self._track_community_lineages()
        
        # Analyze evolution patterns
        evolution_patterns = self._analyze_evolution_patterns()
        
        return {
            'communities_timeline': self.communities_history,
            'lineages': {lid: self._serialize_lineage(lineage) 
                        for lid, lineage in self.lineages.items()},
            'evolution_patterns': evolution_patterns,
            'summary_statistics': self._compute_summary_statistics()
        }
    
    def _detect_communities_timeline(self, method: str) -> None:
        """Detect communities at each timestamp"""
        timestamps = self.embeddings.get_timestamps()
        
        for timestamp in timestamps:
            communities = self._detect_communities_at_time(timestamp, method)
            self.communities_history[timestamp] = communities
    
    def _detect_communities_at_time(self, timestamp: Timestamp, 
                                   method: str) -> List[Set[Word]]:
        """Detect communities at a specific timestamp"""
        # Build semantic network
        network = self._build_semantic_network(timestamp)
        
        if network.number_of_nodes() < 2:
            return []
        
        # Apply community detection method
        if method == 'louvain':
            communities = nx.community.louvain_communities(
                network, weight='weight', resolution=1.0
            )
        elif method == 'leiden':
            try:
                import leidenalg
                communities = nx.community.leiden_communities(
                    network, weight='weight'
                )
            except ImportError:
                # Fallback to Louvain
                communities = nx.community.louvain_communities(
                    network, weight='weight'
                )
        elif method == 'girvan_newman':
            communities = list(nx.community.girvan_newman(network))
            if communities:
                communities = communities[0]  # Take first level
        else:
            raise ValueError(f"Unknown community detection method: {method}")
        
        # Filter small communities
        min_size = getattr(self.config, 'min_community_size', 3)
        filtered_communities = [
            set(c) for c in communities 
            if len(c) >= min_size
        ]
        
        return filtered_communities
    
    def _build_semantic_network(self, timestamp: Timestamp) -> nx.Graph:
        """Build semantic network for community detection"""
        from scipy.spatial.distance import cosine
        
        G = nx.Graph()
        vocabulary = list(self.embeddings.get_vocabulary(timestamp))
        
        # Add nodes
        G.add_nodes_from(vocabulary)
        
        # Add edges based on semantic similarity
        threshold = 0.3  # Similarity threshold
        
        for i, w1 in enumerate(vocabulary):
            emb1 = self.embeddings.get_embedding(w1, timestamp)
            if emb1 is None:
                continue
            
            for w2 in vocabulary[i+1:]:
                emb2 = self.embeddings.get_embedding(w2, timestamp)
                if emb2 is not None:
                    similarity = 1 - cosine(emb1, emb2)
                    
                    if similarity > threshold:
                        G.add_edge(w1, w2, weight=similarity)
        
        return G
    
    def _track_community_lineages(self) -> None:
        """Track community lineages across time"""
        timestamps = sorted(self.communities_history.keys())
        
        if len(timestamps) < 2:
            return
        
        # Initialize lineages from first timestamp
        for community in self.communities_history[timestamps[0]]:
            lineage_id = f"lineage_{self.lineage_counter}"
            self.lineage_counter += 1
            
            self.lineages[lineage_id] = CommunityLineage(
                lineage_id=lineage_id,
                birth_time=timestamps[0],
                death_time=None,
                peak_size=len(community),
                total_persistence=1,
                splits=[],
                merges=[],
                core_vocabulary=community.copy()
            )
        
        # Track evolution through subsequent timestamps
        for i in range(1, len(timestamps)):
            prev_timestamp = timestamps[i-1]
            curr_timestamp = timestamps[i]
            
            prev_communities = self.communities_history[prev_timestamp]
            curr_communities = self.communities_history[curr_timestamp]
            
            self._match_communities(prev_communities, curr_communities, 
                                  prev_timestamp, curr_timestamp)
    
    def _match_communities(self, prev_communities: List[Set[Word]],
                          curr_communities: List[Set[Word]],
                          prev_time: Timestamp, curr_time: Timestamp) -> None:
        """Match communities between consecutive timestamps"""
        # Compute overlap matrix
        overlap_matrix = np.zeros((len(prev_communities), len(curr_communities)))
        
        for i, prev_comm in enumerate(prev_communities):
            for j, curr_comm in enumerate(curr_communities):
                overlap = len(prev_comm & curr_comm)
                overlap_matrix[i, j] = overlap
        
        # Track lineage evolution events
        used_curr = set()
        
        for i, prev_comm in enumerate(prev_communities):
            # Find best matches for this previous community
            best_matches = []
            for j, overlap in enumerate(overlap_matrix[i]):
                if overlap > 0 and j not in used_curr:
                    best_matches.append((j, overlap))
            
            best_matches.sort(key=lambda x: x[1], reverse=True)
            
            if not best_matches:
                # Community disappeared
                self._handle_community_death(prev_comm, curr_time)
            elif len(best_matches) == 1:
                # Simple continuation
                j, overlap = best_matches[0]
                curr_comm = curr_communities[j]
                self._handle_community_continuation(
                    prev_comm, curr_comm, prev_time, curr_time
                )
                used_curr.add(j)
            else:
                # Community split
                split_communities = [curr_communities[j] for j, _ in best_matches]
                self._handle_community_split(
                    prev_comm, split_communities, curr_time
                )
                for j, _ in best_matches:
                    used_curr.add(j)
        
        # Handle new communities (births) and merges
        for j, curr_comm in enumerate(curr_communities):
            if j not in used_curr:
                # Check if it's a merge
                contributing_prev = []
                for i, prev_comm in enumerate(prev_communities):
                    if overlap_matrix[i, j] > 0:
                        contributing_prev.append(prev_comm)
                
                if len(contributing_prev) > 1:
                    # Merge event
                    self._handle_community_merge(
                        contributing_prev, curr_comm, curr_time
                    )
                else:
                    # New community birth
                    self._handle_community_birth(curr_comm, curr_time)
    
    def _handle_community_continuation(self, prev_comm: Set[Word],
                                     curr_comm: Set[Word],
                                     prev_time: Timestamp,
                                     curr_time: Timestamp) -> None:
        """Handle community continuation"""
        # Find lineage that matches previous community
        for lineage in self.lineages.values():
            if lineage.death_time is None:  # Active lineage
                # Check overlap with core vocabulary
                overlap = len(lineage.core_vocabulary & prev_comm)
                if overlap > len(prev_comm) * 0.5:  # 50% overlap threshold
                    # Update lineage
                    lineage.total_persistence += 1
                    lineage.peak_size = max(lineage.peak_size, len(curr_comm))
                    lineage.core_vocabulary &= curr_comm  # Keep core
                    break
    
    def _handle_community_split(self, prev_comm: Set[Word],
                               split_communities: List[Set[Word]],
                               curr_time: Timestamp) -> None:
        """Handle community split event"""
        # Find parent lineage
        parent_lineage = None
        for lineage in self.lineages.values():
            if lineage.death_time is None:
                overlap = len(lineage.core_vocabulary & prev_comm)
                if overlap > len(prev_comm) * 0.5:
                    parent_lineage = lineage
                    break
        
        if parent_lineage:
            parent_lineage.splits.append(curr_time)
            
            # Create new lineages for split communities
            for split_comm in split_communities[1:]:  # Keep first in parent
                new_lineage_id = f"lineage_{self.lineage_counter}"
                self.lineage_counter += 1
                
                self.lineages[new_lineage_id] = CommunityLineage(
                    lineage_id=new_lineage_id,
                    birth_time=curr_time,
                    death_time=None,
                    peak_size=len(split_comm),
                    total_persistence=1,
                    splits=[],
                    merges=[],
                    core_vocabulary=split_comm.copy()
                )
            
            # Update parent lineage with largest split
            largest_split = max(split_communities, key=len)
            parent_lineage.core_vocabulary &= largest_split
    
    def _handle_community_merge(self, contributing_communities: List[Set[Word]],
                               merged_comm: Set[Word],
                               curr_time: Timestamp) -> None:
        """Handle community merge event"""
        # Find contributing lineages
        contributing_lineages = []
        for contrib_comm in contributing_communities:
            for lineage in self.lineages.values():
                if lineage.death_time is None:
                    overlap = len(lineage.core_vocabulary & contrib_comm)
                    if overlap > len(contrib_comm) * 0.3:
                        contributing_lineages.append(lineage)
                        break
        
        if contributing_lineages:
            # Keep largest lineage as survivor
            survivor = max(contributing_lineages, 
                         key=lambda l: l.peak_size)
            survivor.merges.append(curr_time)
            survivor.core_vocabulary = merged_comm.copy()
            
            # Mark others as dead
            for lineage in contributing_lineages:
                if lineage != survivor:
                    lineage.death_time = curr_time
    
    def _handle_community_birth(self, new_comm: Set[Word],
                               curr_time: Timestamp) -> None:
        """Handle new community birth"""
        lineage_id = f"lineage_{self.lineage_counter}"
        self.lineage_counter += 1
        
        self.lineages[lineage_id] = CommunityLineage(
            lineage_id=lineage_id,
            birth_time=curr_time,
            death_time=None,
            peak_size=len(new_comm),
            total_persistence=1,
            splits=[],
            merges=[],
            core_vocabulary=new_comm.copy()
        )
    
    def _handle_community_death(self, dead_comm: Set[Word],
                               curr_time: Timestamp) -> None:
        """Handle community death"""
        for lineage in self.lineages.values():
            if lineage.death_time is None:
                overlap = len(lineage.core_vocabulary & dead_comm)
                if overlap > len(dead_comm) * 0.5:
                    lineage.death_time = curr_time
                    break
    
    def _analyze_evolution_patterns(self) -> Dict[str, any]:
        """Analyze patterns in community evolution"""
        timestamps = sorted(self.communities_history.keys())
        
        # Community count over time
        community_counts = [
            len(self.communities_history[t]) for t in timestamps
        ]
        
        # Birth and death rates
        birth_events = defaultdict(int)
        death_events = defaultdict(int)
        split_events = defaultdict(int)
        merge_events = defaultdict(int)
        
        for lineage in self.lineages.values():
            if lineage.birth_time in timestamps:
                birth_events[lineage.birth_time] += 1
            if lineage.death_time in timestamps:
                death_events[lineage.death_time] += 1
            
            for split_time in lineage.splits:
                split_events[split_time] += 1
            for merge_time in lineage.merges:
                merge_events[merge_time] += 1
        
        return {
            'community_count_timeline': dict(zip(timestamps, community_counts)),
            'birth_events': dict(birth_events),
            'death_events': dict(death_events),
            'split_events': dict(split_events),
            'merge_events': dict(merge_events),
            'average_community_size': self._compute_average_community_size(),
            'community_stability': self._compute_community_stability()
        }
    
    def _compute_average_community_size(self) -> Dict[Timestamp, float]:
        """Compute average community size over time"""
        avg_sizes = {}
        
        for timestamp, communities in self.communities_history.items():
            if communities:
                sizes = [len(c) for c in communities]
                avg_sizes[timestamp] = np.mean(sizes)
            else:
                avg_sizes[timestamp] = 0.0
        
        return avg_sizes
    
    def _compute_community_stability(self) -> float:
        """Compute overall community stability"""
        active_lineages = [l for l in self.lineages.values() 
                          if l.death_time is None]
        
        if not active_lineages:
            return 0.0
        
        # Average persistence of active lineages
        avg_persistence = np.mean([l.total_persistence for l in active_lineages])
        
        # Normalize by timeline length
        timeline_length = len(self.communities_history)
        
        return avg_persistence / timeline_length if timeline_length > 0 else 0.0
    
    def _compute_summary_statistics(self) -> Dict[str, any]:
        """Compute summary statistics"""
        total_lineages = len(self.lineages)
        active_lineages = len([l for l in self.lineages.values() 
                              if l.death_time is None])
        
        if self.lineages:
            avg_lifespan = np.mean([
                l.total_persistence for l in self.lineages.values()
            ])
            max_lifespan = max(l.total_persistence for l in self.lineages.values())
        else:
            avg_lifespan = 0.0
            max_lifespan = 0
        
        return {
            'total_lineages': total_lineages,
            'active_lineages': active_lineages,
            'average_lifespan': avg_lifespan,
            'max_lifespan': max_lifespan,
            'stability_score': self._compute_community_stability()
        }
    
    def _serialize_lineage(self, lineage: CommunityLineage) -> Dict:
        """Serialize lineage for export"""
        return {
            'lineage_id': lineage.lineage_id,
            'birth_time': lineage.birth_time,
            'death_time': lineage.death_time,
            'peak_size': lineage.peak_size,
            'total_persistence': lineage.total_persistence,
            'splits': lineage.splits,
            'merges': lineage.merges,
            'core_vocabulary': list(lineage.core_vocabulary)
        }