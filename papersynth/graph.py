"""
Knowledge graph builder — constructs a citation-methodology graph.

Nodes: papers (with cluster membership, methodology metadata)
Edges: citations between papers, methodology similarity within clusters
"""

from __future__ import annotations
import logging
from collections import defaultdict
from itertools import combinations

import networkx as nx
import numpy as np

from .models import Paper, Cluster

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Citation-methodology knowledge graph."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.paper_nodes: dict[str, Paper] = {}
        self.clusters: dict[int, Cluster] = {}
    
    def build(self, papers: list[Paper], clusters: list[Cluster]) -> nx.DiGraph:
        """Build the full knowledge graph."""
        logger.info(f"Building knowledge graph from {len(papers)} papers and {len(clusters)} clusters...")
        
        self.paper_nodes = {p.paper_id: p for p in papers}
        self.clusters = {c.cluster_id: c for c in clusters}
        
        # Add paper nodes
        for paper in papers:
            self.graph.add_node(
                paper.paper_id,
                type="paper",
                title=paper.title,
                year=paper.year,
                cluster_id=paper.cluster_id,
                citation_count=paper.citation_count,
                keywords=paper.methodology_keywords,
            )
        
        # Add cluster nodes
        for cluster in clusters:
            self.graph.add_node(
                f"cluster_{cluster.cluster_id}",
                type="cluster",
                label=cluster.label,
                size=len(cluster.papers),
                density=cluster.density,
            )
            
            # Connect papers to their cluster
            for pid in cluster.papers:
                if pid in self.paper_nodes:
                    self.graph.add_edge(
                        pid, f"cluster_{cluster.cluster_id}",
                        type="membership",
                    )
        
        # Add citation edges
        for paper in papers:
            for ref_id in paper.references:
                if ref_id in self.paper_nodes:
                    self.graph.add_edge(
                        paper.paper_id, ref_id,
                        type="citation",
                    )
            for cit_id in paper.citations:
                if cit_id in self.paper_nodes:
                    self.graph.add_edge(
                        cit_id, paper.paper_id,
                        type="citation",
                    )
        
        # Add methodology similarity edges within clusters
        for cluster in clusters:
            cluster_papers = [self.paper_nodes[pid] for pid in cluster.papers if pid in self.paper_nodes]
            for p1, p2 in combinations(cluster_papers, 2):
                shared_kw = set(p1.methodology_keywords) & set(p2.methodology_keywords)
                if shared_kw:
                    self.graph.add_edge(
                        p1.paper_id, p2.paper_id,
                        type="methodology_sim",
                        shared_keywords=list(shared_kw),
                        weight=len(shared_kw),
                    )
        
        logger.info(f"Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph
    
    def compute_metrics(self) -> dict:
        """Compute graph-level metrics for gap analysis."""
        logger.info("Computing graph metrics...")
        
        # Get paper-only subgraph for citation analysis
        paper_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "paper"]
        paper_sub = self.graph.subgraph(paper_nodes).to_undirected()
        
        metrics = {
            "total_papers": len(paper_nodes),
            "total_edges": self.graph.number_of_edges(),
            "citation_edges": sum(1 for _, _, d in self.graph.edges(data=True) if d.get("type") == "citation"),
            "methodology_edges": sum(1 for _, _, d in self.graph.edges(data=True) if d.get("type") == "methodology_sim"),
        }
        
        # Per-cluster metrics
        for cid, cluster in self.clusters.items():
            cluster_papers = [pid for pid in cluster.papers if pid in self.graph]
            if not cluster_papers:
                continue
            
            subgraph = self.graph.subgraph(cluster_papers)
            
            metrics[f"cluster_{cid}_size"] = len(cluster_papers)
            metrics[f"cluster_{cid}_internal_edges"] = subgraph.number_of_edges()
            metrics[f"cluster_{cid}_density"] = (
                nx.density(subgraph) if len(cluster_papers) > 1 else 0
            )
        
        # Bridge papers (papers that connect different clusters)
        bridges = self._find_bridge_papers()
        metrics["bridge_papers"] = len(bridges)
        
        return metrics
    
    def _find_bridge_papers(self) -> list[dict]:
        """Find papers that bridge between different methodology clusters."""
        bridges = []
        
        for paper in self.paper_nodes.values():
            if paper.cluster_id is None or paper.cluster_id == -1:
                continue
            
            connected_clusters = set()
            for ref_id in paper.references:
                if ref_id in self.paper_nodes:
                    other = self.paper_nodes[ref_id]
                    if other.cluster_id is not None and other.cluster_id != -1 and other.cluster_id != paper.cluster_id:
                        connected_clusters.add(other.cluster_id)
            
            for cit_id in paper.citations:
                if cit_id in self.paper_nodes:
                    other = self.paper_nodes[cit_id]
                    if other.cluster_id is not None and other.cluster_id != -1 and other.cluster_id != paper.cluster_id:
                        connected_clusters.add(other.cluster_id)
            
            if connected_clusters:
                bridges.append({
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "home_cluster": paper.cluster_id,
                    "bridges_to": sorted(connected_clusters),
                    "bridge_count": len(connected_clusters),
                })
        
        return sorted(bridges, key=lambda x: x["bridge_count"], reverse=True)
    
    def get_cluster_pair_connections(self) -> dict[tuple[int, int], int]:
        """Count citation connections between each pair of clusters."""
        pair_counts = defaultdict(int)
        
        for paper in self.paper_nodes.values():
            if paper.cluster_id is None or paper.cluster_id == -1:
                continue
            
            connected_ids = set(paper.references) | set(paper.citations)
            for other_id in connected_ids:
                if other_id in self.paper_nodes:
                    other = self.paper_nodes[other_id]
                    if other.cluster_id is not None and other.cluster_id != -1 and other.cluster_id != paper.cluster_id:
                        pair = tuple(sorted([paper.cluster_id, other.cluster_id]))
                        pair_counts[pair] += 1
        
        return dict(pair_counts)
    
    def get_cluster_keyword_profile(self) -> dict[int, dict[str, float]]:
        """Get normalized keyword frequency profile for each cluster."""
        profiles = {}
        
        for cid, cluster in self.clusters.items():
            keyword_counts = defaultdict(float)
            cluster_papers = [self.paper_nodes[pid] for pid in cluster.papers if pid in self.paper_nodes]
            
            for paper in cluster_papers:
                for kw in paper.methodology_keywords:
                    keyword_counts[kw] += 1.0
            
            # Normalize
            total = sum(keyword_counts.values())
            if total > 0:
                profiles[cid] = {kw: count / total for kw, count in keyword_counts.items()}
            else:
                profiles[cid] = {}
        
        return profiles
