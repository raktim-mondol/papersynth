"""
CLI entry point for PaperSynth Agent.

Usage:
    papersynth "continual learning for neural networks"
    papersynth "CRISPR delivery mechanisms" --max-papers 100 --top-hypotheses 5
"""

from __future__ import annotations
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich import box

from .config import Config
from .models import PipelineResult
from .retriever import PaperRetriever
from .embedder import PaperEmbedder
from .graph import KnowledgeGraph
from .gap_detector import GapDetector
from .hypothesizer import HypothesisGenerator

console = Console()


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


async def run_pipeline(
    query: str,
    max_papers: int,
    top_hypotheses: int,
    skip_hypotheses: bool,
) -> PipelineResult:
    """Execute the full PaperSynth pipeline."""
    result = PipelineResult(query=query)
    start_time = time.time()
    
    # Step 1: Retrieve papers
    with console.status("[bold cyan]Searching Semantic Scholar...", spinner="dots"):
        retriever = PaperRetriever()
        try:
            papers = await retriever.retrieve(query, max_papers=max_papers)
        finally:
            await retriever.close()
    
    if not papers:
        console.print("[red]No papers found. Try a different query.[/red]")
        return result
    
    result.papers = papers
    result.papers_found = len(papers)
    console.print(f"[green]✓[/green] Fetched {len(papers)} papers")
    
    # Step 2: Extract keywords + embed
    with console.status("[bold cyan]Extracting methodology keywords...", spinner="dots"):
        embedder = PaperEmbedder()
        papers = embedder.extract_keywords(papers)
    console.print(f"[green]✓[/green] Extracted methodology keywords")
    
    with console.status("[bold cyan]Generating embeddings...", spinner="dots"):
        papers = embedder.embed(papers)
    console.print(f"[green]✓[/green] Generated embeddings")
    
    # Step 3: Cluster
    with console.status("[bold cyan]Clustering papers by methodology...", spinner="dots"):
        papers, clusters = embedder.cluster(papers)
    
    result.papers = papers
    result.clusters = clusters
    result.clusters_found = len(clusters)
    console.print(f"[green]✓[/green] Found {len(clusters)} methodology clusters")
    
    # Step 4: Build knowledge graph
    with console.status("[bold cyan]Building knowledge graph...", spinner="dots"):
        kg = KnowledgeGraph()
        graph = kg.build(papers, clusters)
        metrics = kg.compute_metrics()
    console.print(f"[green]✓[/green] Built graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Step 5: Detect gaps
    with console.status("[bold cyan]Detecting research gaps...", spinner="dots"):
        detector = GapDetector(kg)
        gaps = detector.detect_all(papers)
    
    result.gaps = gaps
    result.gaps_found = len(gaps)
    console.print(f"[green]✓[/green] Detected {len(gaps)} research gaps")
    
    # Step 6: Generate hypotheses
    if not skip_hypotheses and gaps:
        if not Config.DEEPSEEK_API_KEY:
            console.print("[yellow]⚠ Skipping hypothesis generation (no DEEPSEEK_API_KEY)[/yellow]")
        else:
            with console.status("[bold cyan]Generating hypotheses via DeepSeek...", spinner="dots"):
                gen = HypothesisGenerator()
                hypotheses = await gen.generate(
                    query=query,
                    papers=papers,
                    clusters=clusters,
                    gaps=gaps,
                    num_hypotheses=top_hypotheses,
                )
            
            result.hypotheses = hypotheses
            result.hypotheses_generated = len(hypotheses)
            console.print(f"[green]✓[/green] Generated {len(hypotheses)} hypotheses")
    elif skip_hypotheses:
        console.print("[dim]Skipped hypothesis generation (--no-hypotheses)[/dim]")
    
    elapsed = time.time() - start_time
    console.print(f"\n[dim]Pipeline completed in {elapsed:.1f}s[/dim]")
    
    return result


def display_results(result: PipelineResult):
    """Display results in a rich formatted output."""
    
    # === CLUSTERS ===
    console.print("\n")
    console.rule("[bold cyan]Methodology Clusters")
    
    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("ID", style="bold", width=4)
    table.add_column("Label", style="cyan", min_width=30)
    table.add_column("Papers", justify="right", width=6)
    table.add_column("Density", justify="right", width=8)
    table.add_column("Top Keywords", min_width=40)
    
    for cluster in result.clusters:
        kw = ", ".join(cluster.methodology_keywords[:6])
        density_str = f"{cluster.density:.2f}"
        table.add_row(
            str(cluster.cluster_id),
            cluster.label,
            str(len(cluster.papers)),
            density_str,
            kw,
        )
    
    console.print(table)
    
    # === GAPS ===
    console.print("\n")
    console.rule("[bold magenta]Research Gaps Detected")
    
    for i, gap in enumerate(result.gaps[:10], 1):
        type_colors = {
            "missing_bridge": "yellow",
            "under_explored_combo": "cyan",
            "isolated_cluster": "blue",
            "methodology_void": "red",
        }
        color = type_colors.get(gap.gap_type, "white")
        
        console.print(Panel(
            f"[bold]{gap.description}[/bold]\n\n"
            f"[dim]Evidence:[/dim] {gap.evidence}\n"
            f"[dim]Scores:[/dim] novelty={gap.novelty_score:.2f} | "
            f"significance={gap.significance_score:.2f} | "
            f"composite={gap.composite_score:.2f}",
            title=f"[{color}]Gap {i}: {gap.gap_type}[/{color}]",
            border_style=color,
        ))
    
    # === HYPOTHESES ===
    if result.hypotheses:
        console.print("\n")
        console.rule("[bold green]Generated Research Hypotheses")
        
        for i, hyp in enumerate(result.hypotheses, 1):
            console.print(Panel(
                f"[bold cyan]Statement:[/bold cyan] {hyp.statement}\n\n"
                f"[bold]Rationale:[/bold] {hyp.rationale}\n\n"
                f"[bold]Methodology:[/bold] {hyp.methodology_suggestion}\n\n"
                f"[dim]Scores:[/dim] feasibility={hyp.feasibility_score:.2f} | "
                f"novelty={hyp.novelty_score:.2f} | "
                f"impact={hyp.impact_score:.2f} | "
                f"[bold]overall={hyp.overall_score:.2f}[/bold]",
                title=f"[bold green]Hypothesis {i}: {hyp.title}[/bold green]",
                border_style="green",
            ))
    
    # === SUMMARY ===
    console.print("\n")
    console.print(Panel(
        f"[bold]{result.summary()}[/bold]",
        title="[bold white]PaperSynth Summary[/bold white]",
        border_style="white",
    ))


def save_results(result: PipelineResult, output_dir: Path = None):
    """Save results to JSON."""
    output_dir = output_dir or Config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Serialize to dict
    data = {
        "query": result.query,
        "summary": result.summary(),
        "papers": [
            {
                "paper_id": p.paper_id,
                "title": p.title,
                "abstract": p.abstract[:500],
                "year": p.year,
                "venue": p.venue,
                "authors": p.authors[:5],
                "fields_of_study": p.fields_of_study,
                "citation_count": p.citation_count,
                "methodology_keywords": p.methodology_keywords,
                "cluster_id": p.cluster_id,
            }
            for p in result.papers
        ],
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "label": c.label,
                "methodology_keywords": c.methodology_keywords,
                "paper_count": len(c.papers),
                "density": c.density,
                "description": c.description,
            }
            for c in result.clusters
        ],
        "gaps": [
            {
                "gap_id": g.gap_id,
                "gap_type": g.gap_type,
                "description": g.description,
                "evidence": g.evidence,
                "clusters_involved": g.clusters_involved,
                "novelty_score": g.novelty_score,
                "significance_score": g.significance_score,
                "composite_score": g.composite_score,
            }
            for g in result.gaps
        ],
        "hypotheses": [
            {
                "hypothesis_id": h.hypothesis_id,
                "title": h.title,
                "statement": h.statement,
                "rationale": h.rationale,
                "methodology_suggestion": h.methodology_suggestion,
                "feasibility_score": h.feasibility_score,
                "novelty_score": h.novelty_score,
                "impact_score": h.impact_score,
                "overall_score": h.overall_score,
                "related_clusters": h.related_clusters,
            }
            for h in result.hypotheses
        ],
    }
    
    safe_name = result.query.replace(" ", "_")[:50].lower()
    out_path = output_dir / f"papersynth_{safe_name}.json"
    out_path.write_text(json.dumps(data, indent=2, default=str))
    
    console.print(f"\n[dim]Results saved to: {out_path}[/dim]")
    return out_path


@click.command()
@click.argument("query")
@click.option("--max-papers", "-n", default=200, help="Maximum papers to fetch")
@click.option("--top-hypotheses", "-h", default=5, help="Number of hypotheses to generate")
@click.option("--no-hypotheses", is_flag=True, help="Skip hypothesis generation")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory for results")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def main(query: str, max_papers: int, top_hypotheses: int, no_hypotheses: bool, 
         output_dir: str, verbose: bool):
    """
    PaperSynth Agent — Literature-to-Hypothesis Pipeline.
    
    Search academic literature, cluster by methodology, detect research gaps,
    and generate novel hypotheses.
    
    Example:
        papersynth "continual learning for neural networks"
        papersynth "CRISPR delivery mechanisms" --max-papers 100
    """
    setup_logging(verbose)
    
    # Validate config
    errors = Config.validate()
    if errors and not no_hypotheses:
        console.print("[yellow]Warnings:[/yellow]")
        for err in errors:
            console.print(f"  ⚠ {err}")
    
    console.print(Panel(
        f"[bold cyan]{query}[/bold cyan]",
        title="[bold]PaperSynth Agent[/bold]",
        subtitle="Literature → Clusters → Gaps → Hypotheses",
        border_style="cyan",
    ))
    
    # Run pipeline
    result = asyncio.run(run_pipeline(
        query=query,
        max_papers=max_papers,
        top_hypotheses=top_hypotheses,
        skip_hypotheses=no_hypotheses,
    ))
    
    # Display results
    display_results(result)
    
    # Save results
    out_dir = Path(output_dir) if output_dir else Config.OUTPUT_DIR
    save_results(result, out_dir)


if __name__ == "__main__":
    main()
