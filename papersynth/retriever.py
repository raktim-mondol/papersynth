"""
Paper retriever — fetches papers from Semantic Scholar and arXiv.

Uses Semantic Scholar's Graph API for paper search + citation/reference expansion.
Falls back to arXiv API if S2 is rate-limited.
"""

from __future__ import annotations
import asyncio
import time
import logging
import xml.etree.ElementTree as ET
from typing import Optional

import httpx

from .config import Config
from .models import Paper

logger = logging.getLogger(__name__)

S2_FIELDS = "paperId,title,abstract,year,venue,authors,fieldsOfStudy,citationCount,referenceCount,externalIds,url"
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
        self._rate_delay = max(Config.S2_RATE_LIMIT_DELAY, 4.0)  # Conservative for no-key
        self._consecutive_failures = 0
    
    async def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        delay = self._rate_delay + (self._consecutive_failures * 2)  # Back off on failures
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)
        self._last_request_time = time.time()
    
    async def _get_with_retry(self, path: str, params: dict, max_retries: int = 3) -> Optional[dict]:
        for attempt in range(max_retries):
            await self._rate_limit()
            try:
                resp = await self.client.get(path, params=params)
                if resp.status_code == 429:
                    wait = (attempt + 1) * 8  # 8s, 16s, 24s
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
    
    async def search(self, query: str, limit: int = 100) -> list[Paper]:
        """Search Semantic Scholar for papers."""
        logger.info(f"Searching S2 for: '{query}' (limit={limit})")
        papers = []
        offset = 0
        batch_size = min(limit, 100)
        while len(papers) < limit and offset < limit:
            data = await self._get_with_retry("/paper/search", {
                "query": query,
                "offset": offset,
                "limit": batch_size,
                "fields": S2_FIELDS,
            })
            if not data or not data.get("data"):
                break
            for item in data["data"]:
                paper = self._to_paper(item)
                if paper:
                    papers.append(paper)
            if not data.get("next"):
                break
            offset += batch_size
        logger.info(f"Found {len(papers)} papers from S2 search")
        return papers
    
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
        """Full retrieval pipeline: search S2 + arXiv, then expand."""
        max_papers = max_papers or Config.MAX_PAPERS
        
        # Step 1: Search S2
        papers = await self.search(query, limit=min(max_papers, 100))
        
        # Step 2: Always also search arXiv for diversity
        arxiv_papers = await self.search_arxiv(query, limit=max_papers)
        
        # Merge: S2 papers first, then arXiv (dedup handled by _seen_ids)
        papers.extend(arxiv_papers)
        
        # Step 3: Expand with citations (only S2 papers have citations)
        s2_papers = [p for p in papers if not p.paper_id.startswith("arxiv:")]
        if s2_papers and len(papers) < max_papers:
            papers = await self.expand_citations(papers)
        
        logger.info(f"Total unique papers: {len(papers)}")
        return papers[:max_papers]
    
    async def close(self):
        await self.client.aclose()
        await self.arxiv_client.aclose()
