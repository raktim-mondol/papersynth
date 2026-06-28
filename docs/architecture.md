# PaperSynth Architecture Guide

## Overview

PaperSynth is a 5-stage autonomous research pipeline that transforms a natural language query into ranked research hypotheses. Each stage is modular and can be used independently.

## System Architecture

```mermaid
flowchart TB
    subgraph Input["📥 Input"]
        QUERY["Research Query<br/><i>e.g., 'continual learning'</i>"]
    end

    subgraph Retrieval["🔍 Stage 1: Paper Retrieval"]
        direction TB
        S2_SEARCH["Semantic Scholar<br/>Graph API"]
        ARXIV_SEARCH["arXiv<br/>Atom API"]
        MERGE["Dedup & Merge"]
        EXPAND["Citation Expansion<br/>(top papers by citation count)"]
        S2_SEARCH --> MERGE
        ARXIV_SEARCH --> MERGE
        MERGE --> EXPAND
    end

    subgraph Processing["⚙️ Stage 2: Embedding & Clustering"]
        direction TB
        KW_EXT["Keyword Extraction<br/><i>Regex pattern matching</i>"]
        EMBED["Sentence Embedding<br/><i>all-MiniLM-L6-v2</i>"]
        UMAP_R["UMAP Reduction<br/><i>384D → 10D</i>"]
        CLUSTER["HDBSCAN Clustering<br/><i>Density-based</i>"]
        KW_EXT --> EMBED --> UMAP_R --> CLUSTER
    end

    subgraph Graph["🕸️ Stage 3: Knowledge Graph"]
        direction TB
        PAPER_NODES["Paper Nodes"]
        CLUSTER_NODES["Cluster Nodes"]
        CITE_EDGES["Citation Edges"]
        METH_EDGES["Methodology Edges"]
        MEMB_EDGES["Membership Edges"]
        PAPER_NODES --- CITE_EDGES
        PAPER_NODES --- METH_EDGES
        PAPER_NODES --- MEMB_EDGES
        CLUSTER_NODES --- MEMB_EDGES
    end

    subgraph Gaps["🔎 Stage 4: Gap Detection"]
        direction TB
        BRIDGE["Missing Bridges"]
        COMBO["Under-explored<br/>Combinations"]
        ISOLATED["Isolated Clusters"]
        VOID["Methodology Voids"]
        RANK["Score & Rank<br/><i>novelty × significance</i>"]
        BRIDGE --> RANK
        COMBO --> RANK
        ISOLATED --> RANK
        VOID --> RANK
    end

    subgraph Output["📤 Stage 5: Hypothesis Generation"]
        direction TB
        LLM["DeepSeek LLM<br/><i>Structured prompt</i>"]
        PARSE["JSON Parsing"]
        SCORE["Multi-factor Scoring<br/><i>feasibility × novelty × impact</i>"]
        LLM --> PARSE --> SCORE
    end

    QUERY --> Retrieval
    EXPAND --> Processing
    CLUSTER --> Graph
    Graph --> Gaps
    RANK --> Output

    style Input fill:#1a1a2e,stroke:#e94560,color:#fff
    style Retrieval fill:#16213e,stroke:#0f3460,color:#fff
    style Processing fill:#0f3460,stroke:#533483,color:#fff
    style Graph fill:#533483,stroke:#e94560,color:#fff
    style Gaps fill:#e94560,stroke:#1a1a2e,color:#fff
    style Output fill:#0f3460,stroke:#16213e,color:#fff
```

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI as CLI (cli.py)
    participant Ret as Retriever
    participant S2 as Semantic Scholar
    participant AX as arXiv
    participant Emb as Embedder
    participant Gr as Graph
    participant Gap as Gap Detector
    participant Hyp as Hypothesizer
    participant DS as DeepSeek API

    User->>CLI: papersynth "topic"
    CLI->>Ret: retrieve("topic")
    
    par Fetch papers
        Ret->>S2: search(query)
        S2-->>Ret: papers[]
    and
        Ret->>AX: search_arxiv(query)
        AX-->>Ret: papers[]
    end
    
    Ret->>Ret: deduplicate
    Ret->>S2: expand_citations(top_papers)
    S2-->>Ret: expanded_papers[]
    Ret-->>CLI: papers[]
    
    CLI->>Emb: extract_keywords(papers)
    Emb-->>CLI: papers with keywords
    
    CLI->>Emb: embed(papers)
    Emb-->>CLI: papers with embeddings
    
    CLI->>Emb: cluster(papers)
    Emb-->>CLI: papers[], clusters[]
    
    CLI->>Gr: build(papers, clusters)
    Gr-->>CLI: knowledge_graph
    
    CLI->>Gap: detect_all(papers)
    Gap-->>CLI: gaps[]
    
    CLI->>Hyp: generate(query, papers, clusters, gaps)
    Hyp->>DS: chat.completions.create(...)
    DS-->>Hyp: response
    Hyp-->>CLI: hypotheses[]
    
    CLI->>User: display_results() + save JSON
```

## Module Dependencies

```mermaid
graph TD
    cli["cli.py<br/><i>Orchestrator</i>"]
    config["config.py<br/><i>Settings</i>"]
    models["models.py<br/><i>Data structures</i>"]
    retriever["retriever.py<br/><i>Paper fetching</i>"]
    embedder["embedder.py<br/><i>Embedding + Clustering</i>"]
    graph["graph.py<br/><i>Knowledge graph</i>"]
    gap_detector["gap_detector.py<br/><i>Gap analysis</i>"]
    hypothesizer["hypothesizer.py<br/><i>LLM generation</i>"]

    cli --> retriever
    cli --> embedder
    cli --> graph
    cli --> gap_detector
    cli --> hypothesizer
    cli --> config
    cli --> models

    retriever --> config
    retriever --> models
    
    embedder --> config
    embedder --> models
    
    graph --> models
    
    gap_detector --> models
    gap_detector --> graph
    gap_detector --> config
    
    hypothesizer --> config
    hypothesizer --> models

    style cli fill:#e94560,stroke:#1a1a2e,color:#fff
    style config fill:#533483,stroke:#16213e,color:#fff
    style models fill:#0f3460,stroke:#16213e,color:#fff
```

## Design Decisions

### Why Sentence-Transformers over LLM Embeddings?
- **Speed**: `all-MiniLM-L6-v2` embeds 50 papers in <1 second
- **Cost**: Zero API calls for embedding
- **Quality**: 384-dim embeddings capture semantic similarity well enough for clustering
- **Offline**: Works without internet after model download

### Why HDBSCAN over K-Means?
- **No predefined K**: We don't know how many methodology clusters exist
- **Noise handling**: Papers that don't fit any cluster are labeled as noise (-1)
- **Density-based**: Naturally finds clusters of varying sizes and shapes
- **Robust**: Works well with UMAP-reduced embeddings

### Why UMAP before Clustering?
- Reduces 384D embeddings to 10D, removing noise while preserving structure
- Makes HDBSCAN's Euclidean metric meaningful for high-dimensional data
- Speeds up clustering significantly

### Why NetworkX for the Graph?
- Pure Python — no compiled dependencies
- Rich algorithm library (centrality, community detection, shortest paths)
- Easy serialization and visualization
- Sufficient for academic corpus sizes (100–1000 papers)

### Why DeepSeek for Hypothesis Generation?
- Excellent reasoning at low cost (~$0.01 per hypothesis generation)
- OpenAI-compatible API (easy integration via `openai` SDK)
- Good at structured JSON output
- Strong academic/technical vocabulary

## Performance Characteristics

| Stage | Time (50 papers) | Time (200 papers) | Bottleneck |
|-------|------------------|-------------------|------------|
| S2 Search | 2–5s | 5–15s | API rate limits |
| arXiv Search | 2–5s | 5–15s | XML parsing |
| Citation Expansion | 10–30s | 30–120s | S2 rate limits |
| Keyword Extraction | <1s | <2s | CPU |
| Embedding | 1–3s | 5–10s | Model inference |
| UMAP + HDBSCAN | 5–15s | 15–30s | UMAP fitting |
| Graph Building | <1s | 1–3s | CPU |
| Gap Detection | 1–3s | 3–10s | CPU |
| Hypothesis Gen | 8–15s | 8–15s | LLM latency |
| **Total** | **30–90s** | **60–240s** | S2 rate limits |

## Extending PaperSynth

### Adding a New Data Source
1. Add a new search method to `retriever.py` (e.g., `search_pubmed()`)
2. Call it in `retrieve()` alongside S2 and arXiv
3. Papers are automatically deduplicated by `paper_id`

### Adding a New Gap Detection Strategy
1. Add a new method to `GapDetector` in `gap_detector.py`
2. Return a list of `Gap` objects with the new `gap_type`
3. Call it in `detect_all()`
4. The scoring and ranking handles new types automatically

### Using a Different LLM
1. Change `DEEPSEEK_BASE_URL` and `DEEPSEEK_MODEL` in config
2. Any OpenAI-compatible API works (OpenAI, Anthropic via proxy, local vLLM)
3. The prompt templates in `hypothesizer.py` are provider-agnostic
