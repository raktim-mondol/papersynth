"""
Embedding and clustering module.

Uses sentence-transformers for paper embeddings and HDBSCAN for methodology clustering.
Optionally applies UMAP for dimensionality reduction before clustering.
"""

from __future__ import annotations
import logging
import re
from typing import Optional
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Config
from .models import Paper, Cluster

logger = logging.getLogger(__name__)


def _extract_methodology_keywords_from_abstract(abstract: str, title: str = "") -> list[str]:
    """
    Extract methodology-related keywords from abstract using regex heuristics.
    Returns only keywords that actually appear in the text (no false positives).
    """
    # Combine title + abstract for matching
    text = f"{title} {abstract}".lower()

    # Common methodology terms organized by domain
    methodology_patterns = [
        # CS/ML methods
        r'\b(?:transformer|attention|CNN|RNN|LSTM|GRU|GAN|VAE|autoencoder)\b',
        r'\b(?:reinforcement\s+learning|supervised|unsupervised|semi-supervised|self-supervised)\b',
        r'\b(?:fine-tuning|pre-training|transfer\s+learning|few-shot|zero-shot|meta-learning)\b',
        r'\b(?:graph\s+neural|GNN|GCN|GAT|message\s+passing|graph\s+convolution)\b',
        r'\b(?:clustering|k-means|DBSCAN|hierarchical|density-based)\b',
        r'\b(?:Bayesian|Monte\s+Carlo|variational|posterior|prior)\b',
        r'\b(?:gradient|Adam|SGD|learning\s+rate|convergence)\b',
        r'\b(?:regularization|dropout|batch\s+normalization|layer\s+normalization)\b',
        r'\b(?:multi-head|self-attention|cross-attention|causal)\b',
        r'\b(?:diffusion|score-based|denoising|noise\s+schedule)\b',
        r'\b(?:contrastive|triplet|metric\s+learning|representation)\b',
        r'\b(?:prompt|instruction|RLHF|alignment|in-context|chain-of-thought)\b',
        r'\b(?:federated|distributed|decentralized|privacy-preserving)\b',
        r'\b(?:ablation|benchmark|evaluation|baseline)\b',
        # Bio/CRISPR methods
        r'\b(?:CRISPR|Cas9|Cas12|Cas13|base\s+editing|prime\s+editing)\b',
        r'\b(?:gene\s+therapy|gene\s+editing|genome\s+editing)\b',
        r'\b(?:delivery\s+vector|viral\s+vector|non-viral|lipid\s+nanoparticle|LNPs?)\b',
        r'\b(?:AAV|adeno-associated|lentivirus|adenovirus|plasmid)\b',
        r'\b(?:electroporation|microinjection|nanoparticle|polymer)\b',
        r'\b(?:single-cell|RNA-seq|scRNA|transcriptomics|proteomics|genomics)\b',
        r'\b(?:in\s+vivo|in\s+vitro|ex\s+vivo|clinical\s+trial)\b',
        r'\b(?:off-target|on-target|guide\s+RNA|sgRNA|PAM)\b',
        # CL/continual learning
        r'\b(?:continual|lifelong|incremental|catastrophic\s+forgetting)\b',
        r'\b(?:elastic\s+weight|knowledge\s+distillation|replay|rehearsal)\b',
    ]

    keywords = set()
    for pattern in methodology_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        keywords.update(m.strip().lower() for m in matches)

    # Filter out overly generic single words that don't add value
    generic = {"supervised", "unsupervised", "evaluation", "prior", "distributed", "baseline"}
    keywords -= generic

    return sorted(keywords)


def _build_paper_text(paper: Paper) -> str:
    """Build text representation for embedding."""
    parts = [paper.abstract]
    if paper.methodology_keywords:
        parts.append("Methods: " + ", ".join(paper.methodology_keywords))
    if paper.fields_of_study:
        parts.append("Fields: " + ", ".join(paper.fields_of_study))
    return " ".join(parts)


def _cluster_keyword_overlap(c1_keywords: list[str], c2_keywords: list[str]) -> float:
    """
    Compute overlap between two clusters' keyword sets.
    Uses overlap coefficient (intersection / min size) instead of Jaccard,
    which works better for domains where all clusters share core terms.
    Also considers differentiating keywords — clusters are only merged if
    they share >70% of their keywords AND have few unique terms.
    """
    set1 = set(c1_keywords)
    set2 = set(c2_keywords)
    if not set1 or not set2:
        return 0.0
    intersection = set1 & set2
    # Differentiating keywords (unique to each cluster)
    only_in_1 = set1 - set2
    only_in_2 = set2 - set1
    differentiating = len(only_in_1) + len(only_in_2)

    # If clusters have ANY unique keywords, they're distinct topics
    if differentiating >= 2:
        return 0.0

    # Overlap coefficient: how much of the smaller set is shared
    return len(intersection) / min(len(set1), len(set2))


class PaperEmbedder:
    """Embeds papers and clusters them by methodology."""

    def __init__(self, model_name: str = None):
        model_name = model_name or Config.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def extract_keywords(self, papers: list[Paper]) -> list[Paper]:
        """Extract methodology keywords from all paper abstracts."""
        logger.info(f"Extracting methodology keywords for {len(papers)} papers...")
        for paper in papers:
            paper.methodology_keywords = _extract_methodology_keywords_from_abstract(
                paper.abstract, paper.title
            )
        return papers

    def embed(self, papers: list[Paper]) -> list[Paper]:
        """Generate embeddings for all papers."""
        logger.info(f"Generating embeddings for {len(papers)} papers...")

        texts = [_build_paper_text(p) for p in papers]
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)

        # Normalize for cosine similarity
        embeddings = normalize(embeddings)

        for paper, emb in zip(papers, embeddings):
            paper.embedding = emb.tolist()

        return papers

    def cluster(self, papers: list[Paper], query: str = "") -> tuple[list[Paper], list[Cluster]]:
        """
        Cluster papers by methodology using HDBSCAN.
        Returns updated papers (with cluster_id) and Cluster objects.
        """
        import hdbscan

        embeddings = np.array([p.embedding for p in papers])

        # Optional UMAP reduction for better clustering
        try:
            import umap
            logger.info(f"Applying UMAP reduction to {Config.UMAP_N_COMPONENTS}D...")
            reducer = umap.UMAP(
                n_components=Config.UMAP_N_COMPONENTS,
                n_neighbors=Config.UMAP_N_NEIGHBORS,
                min_dist=Config.UMAP_MIN_DIST,
                metric='cosine',
                random_state=42,
            )
            embeddings_reduced = reducer.fit_transform(embeddings)
        except ImportError:
            logger.warning("UMAP not available, clustering on raw embeddings")
            embeddings_reduced = embeddings

        logger.info("Running HDBSCAN clustering...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=Config.CLUSTER_MIN_SIZE,
            min_samples=Config.CLUSTER_MIN_SAMPLES,
            metric='euclidean',
            cluster_selection_method='eom',
        )
        labels = clusterer.fit_predict(embeddings_reduced)

        # Assign cluster IDs to papers
        for paper, label in zip(papers, labels):
            paper.cluster_id = int(label)

        # Build Cluster objects
        clusters = []
        unique_labels = set(labels)

        for label in sorted(unique_labels):
            if label == -1:
                continue  # Skip noise

            cluster_papers = [p for p in papers if p.cluster_id == label]

            # Collect all methodology keywords in this cluster
            all_keywords = []
            for p in cluster_papers:
                all_keywords.extend(p.methodology_keywords)

            keyword_counts = Counter(all_keywords)
            top_keywords = [kw for kw, _ in keyword_counts.most_common(10)]

            # Compute centroid
            cluster_embeddings = np.array([p.embedding for p in cluster_papers])
            centroid = cluster_embeddings.mean(axis=0).tolist()

            # Generate label from top keywords
            cluster_label = " + ".join(top_keywords[:3]) if top_keywords else f"Cluster {label}"

            # Compute internal vs external citations
            cluster_ids = {p.paper_id for p in cluster_papers}
            internal = 0
            external = 0
            for p in cluster_papers:
                for ref in p.references:
                    if ref in cluster_ids:
                        internal += 1
                    else:
                        external += 1

            clusters.append(Cluster(
                cluster_id=int(label),
                label=cluster_label,
                methodology_keywords=top_keywords,
                papers=[p.paper_id for p in cluster_papers],
                centroid=centroid,
                description=f"Cluster of {len(cluster_papers)} papers focusing on: {cluster_label}",
                internal_citations=internal,
                external_citations=external,
                density=internal / max(internal + external, 1),
            ))

        # Merge near-duplicate clusters
        clusters = self._merge_duplicate_clusters(clusters, papers)

        # Remove off-topic clusters (dominant keywords don't match query)
        if query:
            clusters = self._filter_offtopic_clusters(clusters, papers, query)

        noise_count = sum(1 for l in labels if l == -1)
        logger.info(f"Found {len(clusters)} clusters ({noise_count} noise papers)")

        return papers, clusters

    def _merge_duplicate_clusters(self, clusters: list[Cluster], papers: list[Paper]) -> list[Cluster]:
        """
        Merge clusters with >60% keyword overlap.
        This prevents the same topic from being split into multiple tiny clusters.
        """
        if len(clusters) <= 1:
            return clusters

        merged = True
        while merged:
            merged = False
            for i in range(len(clusters)):
                if clusters[i] is None:
                    continue
                for j in range(i + 1, len(clusters)):
                    if clusters[j] is None:
                        continue

                    overlap = _cluster_keyword_overlap(
                        clusters[i].methodology_keywords,
                        clusters[j].methodology_keywords
                    )

                    if overlap > 0.45:
                        logger.info(
                            f"Merging cluster {clusters[j].cluster_id} into {clusters[i].cluster_id} "
                            f"(keyword overlap: {overlap:.2f})"
                        )
                        # Merge j into i
                        clusters[i].papers.extend(clusters[j].papers)
                        clusters[i].internal_citations += clusters[j].internal_citations
                        clusters[i].external_citations += clusters[j].external_citations

                        # Recompute keywords
                        all_kw = []
                        for pid in clusters[i].papers:
                            for p in papers:
                                if p.paper_id == pid:
                                    all_kw.extend(p.methodology_keywords)
                                    break
                        kw_counts = Counter(all_kw)
                        clusters[i].methodology_keywords = [kw for kw, _ in kw_counts.most_common(10)]
                        clusters[i].label = " + ".join(clusters[i].methodology_keywords[:3])
                        clusters[i].description = (
                            f"Cluster of {len(clusters[i].papers)} papers focusing on: {clusters[i].label}"
                        )
                        clusters[i].density = clusters[i].internal_citations / max(
                            clusters[i].internal_citations + clusters[i].external_citations, 1
                        )

                        # Update paper cluster_ids
                        for pid in clusters[j].papers:
                            for p in papers:
                                if p.paper_id == pid:
                                    p.cluster_id = clusters[i].cluster_id
                                    break

                        clusters[j] = None  # Mark as merged
                        merged = True

        # Remove merged clusters and re-index
        clusters = [c for c in clusters if c is not None]
        for i, c in enumerate(clusters):
            c.cluster_id = i

        return clusters

    def _filter_offtopic_clusters(
        self, clusters: list[Cluster], papers: list[Paper], query: str
    ) -> list[Cluster]:
        """
        Remove clusters whose top keywords don't overlap with query terms.
        This catches papers that S2 returned as loosely related (e.g., generic ML papers
        that mention CRISPR in passing but aren't about CRISPR research).
        """
        # Extract query-specific terms
        from .retriever import _extract_query_terms
        query_terms = _extract_query_terms(query)
        generic = {"mechanisms", "systems", "approaches", "techniques", "methods",
                    "frameworks", "strategies", "applications", "tools", "review"}
        query_specific = {t for t in query_terms if t not in generic}

        if not query_specific:
            return clusters

        filtered = []
        removed_cluster_ids = set()

        for cluster in clusters:
            # Check if any of the cluster's top keywords match query terms
            cluster_kw = set(kw.lower() for kw in cluster.methodology_keywords)
            overlap = cluster_kw & query_specific

            # Also check if query terms appear in the cluster label
            label_lower = cluster.label.lower()
            label_match = any(term in label_lower for term in query_specific)

            if overlap or label_match:
                filtered.append(cluster)
            else:
                logger.info(
                    f"Removed off-topic cluster {cluster.cluster_id} "
                    f"('{cluster.label}', {len(cluster.papers)} papers) "
                    f"— no overlap with query terms {query_specific}"
                )
                removed_cluster_ids.add(cluster.cluster_id)
                # Mark papers in removed cluster as noise
                for pid in cluster.papers:
                    for p in papers:
                        if p.paper_id == pid:
                            p.cluster_id = -1
                            break

        if removed_cluster_ids:
            # Re-index remaining clusters
            for i, c in enumerate(filtered):
                c.cluster_id = i

        return filtered
