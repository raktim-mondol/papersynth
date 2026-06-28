"""
PaperSynth Agent — Literature-to-Hypothesis Pipeline

Autonomous research agent that:
1. Fetches papers on a topic from Semantic Scholar + arXiv
2. Clusters them by methodology using embeddings
3. Builds a citation-methodology knowledge graph
4. Detects research gaps (missing links, under-explored combinations)
5. Generates novel hypotheses ranked by feasibility
"""

__version__ = "0.1.0"
