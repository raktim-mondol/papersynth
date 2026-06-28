"""Data models for the PaperSynth pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Paper:
    """A single academic paper with metadata."""
    paper_id: str
    title: str
    abstract: str
    year: Optional[int] = None
    venue: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    fields_of_study: list[str] = field(default_factory=list)
    citation_count: int = 0
    url: Optional[str] = None
    
    # Derived fields (populated during pipeline)
    methodology_keywords: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None
    cluster_id: Optional[int] = None
    
    # Graph relationships
    references: list[str] = field(default_factory=list)   # paper_ids this cites
    citations: list[str] = field(default_factory=list)     # paper_ids that cite this


@dataclass
class Cluster:
    """A methodology cluster of related papers."""
    cluster_id: int
    label: str                          # Auto-generated from methodology keywords
    methodology_keywords: list[str] = field(default_factory=list)
    papers: list[str] = field(default_factory=list)  # paper_ids
    centroid: Optional[list[float]] = None
    description: str = ""
    
    # Graph metrics
    internal_citations: int = 0
    external_citations: int = 0
    density: float = 0.0


@dataclass
class Gap:
    """A detected research gap."""
    gap_id: str
    gap_type: str  # "missing_bridge", "under_explored_combo", "isolated_cluster", "methodology_void"
    description: str
    clusters_involved: list[int] = field(default_factory=list)
    papers_involved: list[str] = field(default_factory=list)
    evidence: str = ""
    novelty_score: float = 0.0       # 0-1, how novel this gap is
    significance_score: float = 0.0  # 0-1, how important to address
    composite_score: float = 0.0     # combined ranking score


@dataclass
class Hypothesis:
    """A generated research hypothesis."""
    hypothesis_id: str
    title: str
    statement: str                    # The actual hypothesis
    rationale: str                    # Why this is worth investigating
    methodology_suggestion: str       # How to test it
    evidence_from_gaps: list[str] = field(default_factory=list)
    feasibility_score: float = 0.0    # 0-1
    novelty_score: float = 0.0        # 0-1
    impact_score: float = 0.0         # 0-1
    overall_score: float = 0.0        # weighted composite
    related_papers: list[str] = field(default_factory=list)
    related_clusters: list[int] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Complete result of a PaperSynth run."""
    query: str
    papers: list[Paper] = field(default_factory=list)
    clusters: list[Cluster] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    
    # Pipeline stats
    papers_found: int = 0
    clusters_found: int = 0
    gaps_found: int = 0
    hypotheses_generated: int = 0
    
    def summary(self) -> str:
        return (
            f"PaperSynth Results for: '{self.query}'\n"
            f"  Papers fetched:    {self.papers_found}\n"
            f"  Clusters formed:   {self.clusters_found}\n"
            f"  Gaps detected:     {self.gaps_found}\n"
            f"  Hypotheses:        {self.hypotheses_generated}"
        )
