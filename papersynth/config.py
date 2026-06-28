"""Configuration management for PaperSynth."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")


class Config:
    """Central configuration."""
    
    # DeepSeek API
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"
    
    # Semantic Scholar
    S2_API_KEY: str = os.getenv("S2_API_KEY", "")  # Optional, higher rate limits
    S2_BASE_URL: str = "https://api.semanticscholar.org/graph/v1"
    S2_RATE_LIMIT_DELAY: float = 1.0  # seconds between requests (3/sec without key, 10/sec with)
    
    # Pipeline parameters
    MAX_PAPERS: int = 200             # Max papers to fetch
    CLUSTER_MIN_SIZE: int = 3         # Min papers per cluster
    CLUSTER_MIN_SAMPLES: int = 2      # HDBSCAN min_samples
    TOP_GAPS: int = 10                # Number of top gaps to report
    TOP_HYPOTHESES: int = 5           # Number of hypotheses to generate

    # S2 search filters (set via CLI or env)
    YEAR_MIN: Optional[int] = int(os.getenv("PAPERSYNTH_YEAR_MIN", "0")) or None
    YEAR_MAX: Optional[int] = int(os.getenv("PAPERSYNTH_YEAR_MAX", "0")) or None
    VENUE_FILTER: str = os.getenv("PAPERSYNTH_VENUE", "")
    OPEN_ACCESS_ONLY: bool = os.getenv("PAPERSYNTH_OPEN_ACCESS", "").lower() in ("1", "true", "yes")
    
    # Embedding
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Fast + good quality
    UMAP_N_COMPONENTS: int = 10       # Dimensionality after UMAP
    UMAP_N_NEIGHBORS: int = 15
    UMAP_MIN_DIST: float = 0.0
    
    # Output
    OUTPUT_DIR: Path = _project_root / "output"
    
    @classmethod
    def validate(cls) -> list[str]:
        """Return list of validation errors."""
        errors = []
        if not cls.DEEPSEEK_API_KEY:
            errors.append("DEEPSEEK_API_KEY not set (needed for hypothesis generation)")
        return errors
    
    @classmethod
    def ensure_output_dir(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
