#!/usr/bin/env python3
"""Diagnose why certain topics produce 0 clusters."""
import sys, asyncio, logging
sys.path.insert(0, '/home/raktim/papersynth')
logging.basicConfig(level=logging.WARNING)

from papersynth.retriever import PaperRetriever, _extract_query_terms
from papersynth.embedder import PaperEmbedder, _extract_methodology_keywords_from_abstract
from collections import Counter
import numpy as np

async def diagnose(topic):
    print(f"\n{'='*60}")
    print(f"TOPIC: {topic}")
    print(f"Query terms: {_extract_query_terms(topic)}")
    
    r = PaperRetriever()
    papers = await r.retrieve(topic, max_papers=50)
    await r.close()
    
    if not papers:
        print(f"  NO PAPERS RETURNED")
        return
    
    print(f"Papers after filter: {len(papers)}")
    
    # Check keyword extraction
    all_kw = Counter()
    zero_kw = 0
    for p in papers:
        kw = _extract_methodology_keywords_from_abstract(p.abstract, p.title)
        all_kw.update(kw)
        if not kw:
            zero_kw += 1
    
    print(f"Papers with 0 keywords: {zero_kw}/{len(papers)}")
    print(f"Top keywords: {all_kw.most_common(8)}")
    
    # Check clustering
    e = PaperEmbedder()
    papers = e.extract_keywords(papers)
    papers = e.embed(papers)
    
    import hdbscan, umap
    embeddings = np.array([p.embedding for p in papers])
    reducer = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.0, metric='cosine', random_state=42)
    reduced = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2, metric='euclidean')
    labels = clusterer.fit_predict(reduced)
    
    counts = Counter(labels)
    n_clusters = len([l for l in counts if l != -1])
    noise = counts.get(-1, 0)
    
    print(f"HDBSCAN: {n_clusters} clusters, {noise} noise papers")
    
    if n_clusters > 0:
        for label in sorted(counts):
            if label == -1:
                continue
            cps = [p for p, l in zip(papers, labels) if l == label]
            ckw = Counter()
            for p in cps:
                ckw.update(p.methodology_keywords)
            print(f"  Cluster {label}: {len(cps)} papers, kw={[k for k,_ in ckw.most_common(3)]}")

async def main():
    topics = [
        "large language model reasoning",
        "protein structure prediction", 
        "immunotherapy checkpoint inhibitors",
        "telemedicine rural healthcare access",
        "cybersecurity intrusion detection deep learning",
        "synthetic biology metabolic engineering",
    ]
    for t in topics:
        await diagnose(t)

asyncio.run(main())
