"""
Gap detection engine — identifies research gaps from the knowledge graph.

Gap types:
1. Missing bridges: Clusters that should have cross-citations but don't
2. Under-explored combinations: Methodology pairs rarely combined
3. Isolated clusters: Research threads with few external connections
4. Methodology voids: Topics that appear in the query domain but are under-studied
"""

from __future__ import annotations
import logging
import uuid
from collections import defaultdict
from itertools import combinations

import numpy as np

from .models import Paper, Cluster, Gap
from .graph import KnowledgeGraph
from .config import Config

logger = logging.getLogger(__name__)


class GapDetector:
    """Detects research gaps from a knowledge graph."""
    
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.clusters = graph.clusters
    
    def detect_all(self, papers: list[Paper]) -> list[Gap]:
        """Run all gap detection strategies and return merged, ranked gaps."""
        logger.info("Running gap detection...")
        
        gaps = []
        
        # Strategy 1: Missing bridges between clusters
        gaps.extend(self._detect_missing_bridges())
        
        # Strategy 2: Under-explored methodology combinations
        gaps.extend(self._detect_under_explored_combos())
        
        # Strategy 3: Isolated clusters
        gaps.extend(self._detect_isolated_clusters())
        
        # Strategy 4: Keyword methodology voids
        gaps.extend(self._detect_methodology_voids(papers))
        
        # Score and rank
        gaps = self._score_gaps(gaps)
        gaps.sort(key=lambda g: g.composite_score, reverse=True)
        
        top_gaps = gaps[:Config.TOP_GAPS]
        logger.info(f"Detected {len(gaps)} total gaps, returning top {len(top_gaps)}")
        return top_gaps
    
    def _detect_missing_bridges(self) -> list[Gap]:
        """
        Find pairs of clusters that share methodology keywords but have few/no 
        cross-citations — these represent natural connections that haven't been made.
        """
        gaps = []
        
        keyword_profiles = self.graph.get_cluster_keyword_profile()
        pair_connections = self.graph.get_cluster_pair_connections()
        
        # Compare each pair of clusters
        cluster_ids = list(self.clusters.keys())
        
        for c1_id, c2_id in combinations(cluster_ids, 2):
            # Compute keyword overlap
            profile1 = keyword_profiles.get(c1_id, {})
            profile2 = keyword_profiles.get(c2_id, {})
            
            shared_keywords = set(profile1.keys()) & set(profile2.keys())
            if not shared_keywords:
                continue
            
            # How much overlap?
            overlap_strength = sum(
                min(profile1.get(kw, 0), profile2.get(kw, 0))
                for kw in shared_keywords
            )
            
            # How many actual cross-citations?
            actual_connections = pair_connections.get(
                tuple(sorted([c1_id, c2_id])), 0
            )
            
            # Expected connections based on keyword overlap
            c1_size = len(self.clusters[c1_id].papers)
            c2_size = len(self.clusters[c2_id].papers)
            expected_connections = overlap_strength * min(c1_size, c2_size) * 0.3
            
            if actual_connections < expected_connections and overlap_strength > 0.15:
                c1_label = self.clusters[c1_id].label
                c2_label = self.clusters[c2_id].label
                
                gaps.append(Gap(
                    gap_id=str(uuid.uuid4())[:8],
                    gap_type="missing_bridge",
                    description=(
                        f"Clusters '{c1_label}' and '{c2_label}' share methodology keywords "
                        f"({', '.join(sorted(shared_keywords)[:5])}) but have only {actual_connections} "
                        f"cross-citations. Expected ~{int(expected_connections)}. "
                        f"This suggests an under-explored connection between these research threads."
                    ),
                    clusters_involved=[c1_id, c2_id],
                    evidence=f"Shared keywords: {', '.join(sorted(shared_keywords))}. "
                             f"Actual connections: {actual_connections}, expected: ~{int(expected_connections)}.",
                    novelty_score=min(overlap_strength * 2, 1.0),
                    significance_score=min(expected_connections / max(actual_connections, 1) / 5, 1.0),
                ))
        
        return gaps
    
    def _detect_under_explored_combos(self) -> list[Gap]:
        """
        Find methodology keyword pairs that rarely co-occur in papers,
        even though they appear individually in the corpus.
        """
        gaps = []
        
        keyword_profiles = self.graph.get_cluster_keyword_profile()
        
        # Aggregate all keywords across clusters
        all_keywords: dict[str, int] = defaultdict(int)
        for cid, profile in keyword_profiles.items():
            cluster_size = len(self.clusters[cid].papers)
            for kw, freq in profile.items():
                all_keywords[kw] += int(freq * cluster_size)
        
        # Find keywords that appear in multiple clusters but never together in a paper
        cluster_ids = list(self.clusters.keys())
        
        keyword_to_clusters: dict[str, set[int]] = defaultdict(set)
        for cid, profile in keyword_profiles.items():
            for kw in profile:
                keyword_to_clusters[kw].add(cid)
        
        # Look at keyword pairs that span different clusters
        kw_pairs_checked = set()
        for kw1, kw2 in combinations(sorted(all_keywords.keys()), 2):
            if (kw1, kw2) in kw_pairs_checked:
                continue
            kw_pairs_checked.add((kw1, kw2))
            
            clusters_kw1 = keyword_to_clusters.get(kw1, set())
            clusters_kw2 = keyword_to_clusters.get(kw2, set())
            
            # They appear in different clusters
            if clusters_kw1 and clusters_kw2 and not (clusters_kw1 & clusters_kw2):
                # Check if any paper in the corpus uses both
                co_occurrence = 0
                for paper in self.graph.paper_nodes.values():
                    if kw1 in paper.methodology_keywords and kw2 in paper.methodology_keywords:
                        co_occurrence += 1
                
                if co_occurrence == 0 and all_keywords[kw1] >= 3 and all_keywords[kw2] >= 3:
                    gaps.append(Gap(
                        gap_id=str(uuid.uuid4())[:8],
                        gap_type="under_explored_combo",
                        description=(
                            f"Methodologies '{kw1}' (appears in {all_keywords[kw1]} papers) and "
                            f"'{kw2}' (appears in {all_keywords[kw2]} papers) are never combined "
                            f"in the retrieved corpus, despite both being well-represented. "
                            f"This could indicate an unexplored methodological fusion."
                        ),
                        clusters_involved=sorted(clusters_kw1 | clusters_kw2),
                        evidence=f"'{kw1}' appears in clusters {sorted(clusters_kw1)}, "
                                 f"'{kw2}' appears in clusters {sorted(clusters_kw2)}. "
                                 f"Zero co-occurrence across {len(self.graph.paper_nodes)} papers.",
                        novelty_score=0.8,
                        significance_score=min(
                            (all_keywords[kw1] + all_keywords[kw2]) / len(self.graph.paper_nodes) / 2, 
                            1.0
                        ),
                    ))
        
        return gaps[:10]  # Limit to avoid explosion
    
    def _detect_isolated_clusters(self) -> list[Gap]:
        """Find clusters with very few external connections — isolated research threads."""
        gaps = []
        
        for cid, cluster in self.clusters.items():
            if len(cluster.papers) < 3:
                continue
            
            # Low density means few internal connections
            # Low external citations means few cross-cluster connections
            total_possible = len(cluster.papers) * (len(cluster.papers) - 1) / 2
            
            pair_connections = self.graph.get_cluster_pair_connections()
            external = sum(
                count for (c1, c2), count in pair_connections.items()
                if c1 == cid or c2 == cid
            )
            
            if external < 3 and cluster.density > 0.3:
                # Dense internally but isolated externally
                gaps.append(Gap(
                    gap_id=str(uuid.uuid4())[:8],
                    gap_type="isolated_cluster",
                    description=(
                        f"Cluster '{cluster.label}' ({len(cluster.papers)} papers) is internally cohesive "
                        f"(density: {cluster.density:.2f}) but has only {external} external connections. "
                        f"This research thread may be developing in isolation — potential for cross-pollination "
                        f"with other clusters."
                    ),
                    clusters_involved=[cid],
                    evidence=f"Internal density: {cluster.density:.2f}, external connections: {external}.",
                    novelty_score=0.5,
                    significance_score=min(external / 3 + 0.3, 1.0),
                ))
        
        return gaps
    
    def _detect_methodology_voids(self, papers: list[Paper]) -> list[Gap]:
        """
        Identify methodology keywords that are expected in the research domain 
        (based on fields of study) but absent from the corpus.
        """
        gaps = []
        
        # Collect all fields of study and keywords present
        all_fields = set()
        present_keywords = set()
        for paper in papers:
            all_fields.update(f.lower() for f in paper.fields_of_study)
            present_keywords.update(paper.methodology_keywords)
        
        # Common methodology expectations by field
        field_expected_methods = {
            "computer science": {"transformer", "attention", "reinforcement learning", "graph neural", "contrastive", "prompt", "fine-tuning", "few-shot"},
            "biology": {"crispr", "rna-seq", "single-cell", "gene therapy", "transcriptomics"},
            "medicine": {"randomized", "clinical trial", "meta-analysis", "cohort", "placebo"},
            "mathematics": {"bayesian", "optimization", "convergence", "variational"},
            "engineering": {"optimization", "simulation", "control", "signal processing"},
        }
        
        for field in all_fields:
            for known_field, expected_kw in field_expected_methods.items():
                if known_field in field:
                    missing = expected_kw - present_keywords
                    if missing and len(missing) >= 2:
                        gaps.append(Gap(
                            gap_id=str(uuid.uuid4())[:8],
                            gap_type="methodology_void",
                            description=(
                                f"For a {field} research topic, methodologies [{', '.join(sorted(missing)[:5])}] "
                                f"are commonly relevant but absent from the retrieved corpus. "
                                f"These represent potential methodological directions not yet explored in this specific problem space."
                            ),
                            evidence=f"Field: {field}. Expected but absent: {', '.join(sorted(missing))}.",
                            novelty_score=0.7,
                            significance_score=0.6,
                        ))
        
        return gaps
    
    def _score_gaps(self, gaps: list[Gap]) -> list[Gap]:
        """Compute composite score for ranking."""
        for gap in gaps:
            # Weighted composite
            gap.composite_score = (
                0.4 * gap.novelty_score +
                0.4 * gap.significance_score +
                0.2 * (1.0 if gap.gap_type == "missing_bridge" else 0.7)
            )
        return gaps
