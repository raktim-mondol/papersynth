"""
Paper retriever — fetches papers from Semantic Scholar and arXiv.

Uses Semantic Scholar's Graph API **bulk search** endpoint for efficient retrieval
with token-based pagination. Falls back to arXiv API for diversity and coverage.
"""

from __future__ import annotations
import asyncio
import re
import time
import logging
import xml.etree.ElementTree as ET
from typing import Optional

import httpx

from .config import Config
from .models import Paper

logger = logging.getLogger(__name__)

# Stop words to strip from queries when extracting key terms
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "using", "use", "used", "based",
    "approach", "method", "methods", "analysis", "study", "research",
    "new", "novel", "paper", "proposed", "via", "through", "between",
}


def _extract_query_terms(query: str) -> list[str]:
    """Extract meaningful terms from a search query, stripping stop words."""
    words = re.findall(r'\b[a-z]+\b', query.lower())
    terms = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
    # Also extract multi-word terms (bigrams)
    bigrams = []
    for i in range(len(words) - 1):
        if words[i] not in _STOP_WORDS and words[i + 1] not in _STOP_WORDS:
            bigrams.append(f"{words[i]} {words[i + 1]}")
    return terms + bigrams

S2_FIELDS = "paperId,title,abstract,year,venue,authors,fieldsOfStudy,citationCount,referenceCount,externalIds,url,openAccessPdf"
S2_CITATION_FIELDS = "paperId,title,abstract,year,venue,authors,fieldsOfStudy,citationCount,url"
S2_REF_FIELDS = S2_CITATION_FIELDS


class PaperRetriever:
    """Fetches and expands a corpus of papers on a given topic."""

    def __init__(self):
        self._last_request_time = 0.0
        headers = {"User-Agent": "PaperSynth/0.1 (academic research agent)"}
        if Config.S2_API_KEY:
            headers["x-api-key"] = Config.S2_API_KEY
        self.client = httpx.AsyncClient(
            base_url=Config.S2_BASE_URL,
            headers=headers,
            timeout=60.0,
            follow_redirects=True,
        )
        self.arxiv_client = httpx.AsyncClient(
            base_url="https://export.arxiv.org/api",
            headers={"User-Agent": "PaperSynth/0.1"},
            timeout=60.0,
            follow_redirects=True,
        )
        self._seen_ids: set[str] = set()
        self._arxiv_id_map: dict[str, str] = {}  # arxiv_id -> s2_paperId
        self._rate_delay = max(Config.S2_RATE_LIMIT_DELAY, 1.0)
        self._consecutive_failures = 0

    async def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        delay = self._rate_delay + (self._consecutive_failures * 2)
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)
        self._last_request_time = time.time()

    async def _get_with_retry(self, path: str, params: dict, max_retries: int = 3) -> Optional[dict]:
        for attempt in range(max_retries):
            await self._rate_limit()
            try:
                resp = await self.client.get(path, params=params)
                if resp.status_code == 429:
                    wait = (attempt + 1) * 8
                    logger.warning(f"Rate limited (attempt {attempt+1}/{max_retries}), waiting {wait}s...")
                    self._consecutive_failures += 1
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                self._consecutive_failures = 0
                return resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait = (attempt + 1) * 8
                    logger.warning(f"Rate limited (attempt {attempt+1}/{max_retries}), waiting {wait}s...")
                    self._consecutive_failures += 1
                    await asyncio.sleep(wait)
                    continue
                logger.warning(f"S2 API error {e.response.status_code} for {path}")
                self._consecutive_failures += 1
                return None
            except Exception as e:
                logger.warning(f"Request failed for {path}: {e}")
                self._consecutive_failures += 1
                if attempt < max_retries - 1:
                    await asyncio.sleep(3)
                    continue
                return None
        return None

    def _to_paper(self, data: dict) -> Optional[Paper]:
        if not data.get("paperId") or not data.get("abstract"):
            return None
        paper_id = data["paperId"]
        if paper_id in self._seen_ids:
            return None
        self._seen_ids.add(paper_id)

        authors = [a.get("name", "") for a in (data.get("authors") or [])]

        # Track arXiv ID mapping for dedup
        ext_ids = data.get("externalIds") or {}
        arxiv_id = ext_ids.get("ArXiv")
        if arxiv_id:
            self._arxiv_id_map[arxiv_id] = paper_id

        return Paper(
            paper_id=paper_id,
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            year=data.get("year"),
            venue=data.get("venue", ""),
            authors=authors,
            fields_of_study=data.get("fieldsOfStudy") or [],
            citation_count=data.get("citationCount") or 0,
            url=data.get("url"),
        )

    def _arxiv_to_paper(self, entry, ns) -> Optional[Paper]:
        title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
        abstract = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")
        arxiv_url = entry.findtext("atom:id", "", ns).strip()
        if not abstract or not arxiv_url:
            return None
        # Clean arxiv ID
        raw_id = arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else arxiv_url.split("/")[-1]
        # Strip version suffix (v1, v2, etc.) for dedup
        base_id = raw_id.rsplit("v", 1)[0] if "v" in raw_id else raw_id

        # Check if S2 already has this paper (via externalIds.ArXiv mapping)
        s2_id = self._arxiv_id_map.get(base_id) or self._arxiv_id_map.get(raw_id)
        if s2_id:
            logger.debug(f"Skipping arXiv {raw_id} — already in S2 as {s2_id}")
            return None

        paper_id = f"arxiv:{raw_id}"
        if paper_id in self._seen_ids:
            return None
        self._seen_ids.add(paper_id)
        authors = [a.findtext("atom:name", "", ns) for a in entry.findall("atom:author", ns)]
        year_str = entry.findtext("atom:published", "", ns)[:4]
        year = int(year_str) if year_str.isdigit() else None
        return Paper(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            year=year,
            venue="arXiv",
            authors=authors,
            fields_of_study=[],
            citation_count=0,
            url=arxiv_url,
        )

    async def search_arxiv(self, query: str, limit: int = 50) -> list[Paper]:
        """Search arXiv API."""
        logger.info(f"Searching arXiv for: '{query}' (limit={limit})")
        papers = []
        try:
            resp = await self.arxiv_client.get("/query", params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
                "sortOrder": "descending",
            })
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                paper = self._arxiv_to_paper(entry, ns)
                if paper:
                    papers.append(paper)
            logger.info(f"Found {len(papers)} papers from arXiv")
        except Exception as e:
            logger.warning(f"arXiv search failed: {e}")
        return papers

    async def search_bulk(self, query: str, limit: int = 100) -> list[Paper]:
        """
        Search Semantic Scholar using the bulk search endpoint.

        Uses /paper/search/bulk with token-based pagination (recommended by S2 docs).
        Supports year range, venue, and open-access filtering.
        """
        logger.info(f"Searching S2 (bulk) for: '{query}' (limit={limit})")
        papers = []

        # Build query parameters for bulk search
        params = {
            "query": query,
            "fields": S2_FIELDS,
        }

        # Add optional filters
        if Config.YEAR_MIN or Config.YEAR_MAX:
            year_min = str(Config.YEAR_MIN) if Config.YEAR_MIN else ""
            year_max = str(Config.YEAR_MAX) if Config.YEAR_MAX else ""
            params["year"] = f"{year_min}-{year_max}"

        if Config.VENUE_FILTER:
            params["venue"] = Config.VENUE_FILTER

        if Config.OPEN_ACCESS_ONLY:
            params["openAccessPdf"] = ""

        # Fetch pages using token-based pagination
        token = None
        while len(papers) < limit:
            if token:
                params["token"] = token

            data = await self._get_with_retry("/paper/search/bulk", params)
            if not data or not data.get("data"):
                break

            batch_papers = 0
            for item in data["data"]:
                paper = self._to_paper(item)
                if paper:
                    papers.append(paper)
                    batch_papers += 1

            logger.debug(f"  Batch: got {len(data['data'])} raw, {batch_papers} usable (total: {len(papers)})")

            # Token-based pagination: "token" field absent = no more results
            token = data.get("token")
            if not token:
                break

            # Remove query from params for subsequent requests (S2 only needs token)
            params.pop("query", None)

        logger.info(f"Found {len(papers)} papers from S2 bulk search")
        return papers[:limit]

    async def expand_citations(self, papers: list[Paper], max_per_paper: int = 3) -> list[Paper]:
        """Expand corpus by fetching references of top papers."""
        logger.info(f"Expanding citations for {len(papers)} papers...")
        top_papers = sorted(papers, key=lambda p: p.citation_count, reverse=True)[:8]
        new_papers = []
        for paper in top_papers:
            refs_data = await self._get_with_retry(f"/paper/{paper.paper_id}/references", {
                "fields": S2_REF_FIELDS,
                "limit": max_per_paper,
            })
            if refs_data and refs_data.get("data"):
                for ref in refs_data["data"]:
                    cited = ref.get("citedPaper", {})
                    if cited and cited.get("paperId"):
                        p = self._to_paper(cited)
                        if p:
                            new_papers.append(p)
                            paper.references.append(p.paper_id)
        logger.info(f"Expanded with {len(new_papers)} additional papers")
        return papers + new_papers

    async def retrieve(self, query: str, max_papers: int = None) -> list[Paper]:
        """Full retrieval pipeline: search S2 bulk + arXiv, filter, expand."""
        max_papers = max_papers or Config.MAX_PAPERS

        # Step 1: Search S2 using bulk endpoint
        papers = await self.search_bulk(query, limit=50)

        # Step 2: Always also search arXiv for diversity
        # arXiv dedup happens via _arxiv_id_map built during S2 search
        arxiv_papers = await self.search_arxiv(query, limit=50)

        # Merge: S2 papers first, then arXiv (dedup handled by _seen_ids + _arxiv_id_map)
        papers.extend(arxiv_papers)

        # Step 3: Filter irrelevant papers (don't match query terms)
        papers = self._filter_relevant(papers, query)

        # Step 4: Expand with citations (only S2 papers have citations)
        s2_papers = [p for p in papers if not p.paper_id.startswith("arxiv:")]
        if s2_papers and len(papers) < max_papers:
            papers = await self.expand_citations(papers)

        logger.info(f"Total unique papers: {len(papers)}")
        return papers[:max_papers]

    def _filter_relevant(self, papers: list[Paper], query: str) -> list[Paper]:
        """
        Remove papers that don't contain enough query key terms in title or abstract.
        Requires at least 1 query-specific term (excluding generic words like 'mechanisms').
        """
        query_terms = _extract_query_terms(query)
        if not query_terms:
            return papers

        # Separate domain-specific terms from generic ones
        generic_terms = {"mechanisms", "systems", "approaches", "techniques", "methods",
                         "frameworks", "strategies", "applications", "tools", "review"}
        specific_terms = [t for t in query_terms if t not in generic_terms and " " not in t]
        if not specific_terms:
            specific_terms = query_terms  # fallback to all terms

        logger.info(f"Relevance filter: specific terms={specific_terms}")

        relevant = []
        removed = 0
        for paper in papers:
            text = f"{paper.title} {paper.abstract}".lower()
            # Paper must contain at least one domain-specific term
            matches = sum(1 for term in specific_terms if term in text)
            if matches >= 1:
                relevant.append(paper)
            else:
                removed += 1
                logger.debug(f"Filtered out ({matches} matches): {paper.title[:60]}...")

        logger.info(f"Relevance filter: kept {len(relevant)}, removed {removed}")
        return relevant

    async def close(self):
        await self.client.aclose()
        await self.arxiv_client.aclose()
