#!/usr/bin/env python3
"""
PaperSynth Benchmark — 50 topics across 10+ domains.
Runs the pipeline without LLM hypothesis generation.
Collects metrics: papers, clusters, gaps, gap types, off-topic removals.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from papersynth.config import Config
from papersynth.retriever import PaperRetriever
from papersynth.embedder import PaperEmbedder
from papersynth.graph import KnowledgeGraph
from papersynth.gap_detector import GapDetector

TOPICS = [
    # Computer Science (10)
    ("transformer attention mechanism", "CS"),
    ("federated learning privacy", "CS"),
    ("graph neural networks combinatorial optimization", "CS"),
    ("large language model reasoning", "CS"),
    ("diffusion model image generation", "CS"),
    ("reinforcement learning robotics", "CS"),
    ("software engineering code review automation", "CS"),
    ("cybersecurity intrusion detection deep learning", "CS"),
    ("natural language processing sentiment analysis", "CS"),
    ("computer vision object detection autonomous driving", "CS"),

    # Biology (8)
    ("CRISPR delivery mechanisms", "Biology"),
    ("single cell RNA sequencing", "Biology"),
    ("protein structure prediction", "Biology"),
    ("gene therapy viral vectors", "Biology"),
    ("microbiome gut brain axis", "Biology"),
    ("epigenetics DNA methylation cancer", "Biology"),
    ("synthetic biology metabolic engineering", "Biology"),
    ("stem cell differentiation regenerative medicine", "Biology"),

    # Medicine (6)
    ("immunotherapy checkpoint inhibitors", "Medicine"),
    ("mRNA vaccine lipid nanoparticle delivery", "Medicine"),
    ("artificial intelligence radiology diagnosis", "Medicine"),
    ("clinical trial adaptive design", "Medicine"),
    ("drug resistance tuberculosis", "Medicine"),
    ("telemedicine rural healthcare access", "Medicine"),

    # Mathematics (4)
    ("neural network convergence theory", "Math"),
    ("bayesian optimization hyperparameter tuning", "Math"),
    ("topological data analysis persistent homology", "Math"),
    ("partial differential equations numerical methods", "Math"),

    # Physics (4)
    ("quantum error correction", "Physics"),
    ("dark matter detection experiments", "Physics"),
    ("gravitational wave astronomy", "Physics"),
    ("topological insulators", "Physics"),

    # Chemistry (4)
    ("metal organic framework catalysis", "Chemistry"),
    ("perovskite solar cell stability", "Chemistry"),
    ("electrochemistry lithium sulfur batteries", "Chemistry"),
    ("computational chemistry molecular dynamics", "Chemistry"),

    # Economics/Business (5)
    ("cryptocurrency market volatility", "Economics"),
    ("supply chain resilience disruption", "Business"),
    ("behavioral economics nudge theory", "Economics"),
    ("corporate governance ESG investing", "Business"),
    ("machine learning financial forecasting", "Economics"),

    # Engineering (4)
    ("additive manufacturing 3D printing metals", "Engineering"),
    ("renewable energy grid integration", "Engineering"),
    ("autonomous vehicle sensor fusion", "Engineering"),
    ("structural health monitoring bridges", "Engineering"),

    # Social Sciences (3)
    ("misinformation social media detection", "Social"),
    ("educational technology adaptive learning", "Social"),
    ("climate change public opinion", "Social"),

    # Environmental Science (2)
    ("carbon capture storage", "Environment"),
    ("biodiversity conservation remote sensing", "Environment"),
]


async def run_one(topic: str, domain: str, idx: int) -> dict:
    """Run pipeline for one topic and return metrics."""
    start = time.time()
    result = {
        "idx": idx,
        "topic": topic,
        "domain": domain,
        "papers_found": 0,
        "papers_after_filter": 0,
        "clusters": 0,
        "cluster_labels": [],
        "gaps": 0,
        "gap_types": {},
        "noise_papers": 0,
        "elapsed_s": 0,
        "error": None,
    }

    try:
        retriever = PaperRetriever()
        papers = await retriever.retrieve(topic, max_papers=100)
        await retriever.close()
        result["papers_after_filter"] = len(papers)

        if not papers:
            result["error"] = "no papers"
            return result

        embedder = PaperEmbedder()
        papers = embedder.extract_keywords(papers)
        papers = embedder.embed(papers)
        papers, clusters = embedder.cluster(papers, query=topic)

        result["papers_found"] = len(papers)
        result["clusters"] = len(clusters)
        result["cluster_labels"] = [c.label for c in clusters]
        result["noise_papers"] = sum(1 for p in papers if p.cluster_id == -1)

        if clusters:
            kg = KnowledgeGraph()
            kg.build(papers, clusters)
            kg.compute_metrics()

            detector = GapDetector(kg)
            gaps = detector.detect_all(papers)

            result["gaps"] = len(gaps)
            for g in gaps:
                result["gap_types"][g.gap_type] = result["gap_types"].get(g.gap_type, 0) + 1

    except Exception as e:
        result["error"] = str(e)[:200]

    result["elapsed_s"] = round(time.time() - start, 1)
    return result


async def main():
    results = []
    total = len(TOPICS)

    for i, (topic, domain) in enumerate(TOPICS):
        print(f"[{i+1}/{total}] {domain}: {topic}...", end=" ", flush=True)
        r = await run_one(topic, domain, i)
        results.append(r)
        if r["error"]:
            print(f"ERROR: {r['error']}")
        else:
            print(f"{r['papers_after_filter']}p → {r['clusters']}c, {r['gaps']}g [{r['elapsed_s']}s]")
        # Pause between queries to avoid rate limiting (S2 API: 1 req/sec)
        await asyncio.sleep(3)

    # Save raw results
    out_path = Path(__file__).parent.parent / "output" / "benchmark_50.json"
    out_path.write_text(json.dumps(results, indent=2))

    # Print summary
    print(f"\n{'='*70}")
    print(f"BENCHMARK SUMMARY — {total} topics")
    print(f"{'='*70}")

    success = [r for r in results if not r["error"]]
    errors = [r for r in results if r["error"]]

    print(f"Success: {len(success)}, Errors: {len(errors)}")

    if success:
        avg_papers = sum(r["papers_after_filter"] for r in success) / len(success)
        avg_clusters = sum(r["clusters"] for r in success) / len(success)
        avg_gaps = sum(r["gaps"] for r in success) / len(success)
        avg_time = sum(r["elapsed_s"] for r in success) / len(success)
        single_cluster = sum(1 for r in success if r["clusters"] <= 1)
        no_gaps = sum(1 for r in success if r["gaps"] == 0)
        only_missing_bridge = sum(1 for r in success if r["gap_types"] == {"missing_bridge": r["gaps"]})

        print(f"Avg papers: {avg_papers:.0f}")
        print(f"Avg clusters: {avg_clusters:.1f}")
        print(f"Avg gaps: {avg_gaps:.1f}")
        print(f"Avg time: {avg_time:.1f}s")
        print(f"Single/no cluster: {single_cluster} ({single_cluster/len(success)*100:.0f}%)")
        print(f"No gaps detected: {no_gaps} ({no_gaps/len(success)*100:.0f}%)")
        print(f"Only missing_bridge gaps: {only_missing_bridge} ({only_missing_bridge/len(success)*100:.0f}%)")

        # Per-domain breakdown
        print(f"\n{'Domain':<15} {'Count':>5} {'Papers':>7} {'Clust':>6} {'Gaps':>5} {'1-clust':>8}")
        print("-" * 50)
        domains = sorted(set(r["domain"] for r in success))
        for d in domains:
            dr = [r for r in success if r["domain"] == d]
            avg_p = sum(r["papers_after_filter"] for r in dr) / len(dr)
            avg_c = sum(r["clusters"] for r in dr) / len(dr)
            avg_g = sum(r["gaps"] for r in dr) / len(dr)
            sc = sum(1 for r in dr if r["clusters"] <= 1)
            print(f"{d:<15} {len(dr):>5} {avg_p:>7.0f} {avg_c:>6.1f} {avg_g:>5.1f} {sc:>8}")

        # Problem cases
        print(f"\nPROBLEM CASES (1 cluster or 0 gaps):")
        for r in success:
            if r["clusters"] <= 1 or r["gaps"] == 0:
                print(f"  [{r['domain']}] {r['topic']}: {r['clusters']}c, {r['gaps']}g")

    if errors:
        print(f"\nERRORS:")
        for r in errors:
            print(f"  [{r['domain']}] {r['topic']}: {r['error']}")


if __name__ == "__main__":
    asyncio.run(main())
