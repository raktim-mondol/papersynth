"""
Hypothesis generator — uses DeepSeek LLM to synthesize research hypotheses from detected gaps.

Takes the gap analysis output and generates novel, testable research hypotheses
with feasibility assessments and methodology suggestions.
"""

from __future__ import annotations
import json
import logging
import uuid

from openai import AsyncOpenAI

from .config import Config
from .models import Paper, Cluster, Gap, Hypothesis

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert research scientist and methodologist. Your task is to analyze research gaps in a body of academic literature and generate novel, testable research hypotheses.

You will receive:
1. The original research query
2. A summary of methodology clusters found in the literature
3. Identified research gaps (missing connections, under-explored combinations, isolated threads)

For each gap, generate a research hypothesis that:
- Directly addresses the gap
- Is specific and testable
- Suggests a concrete methodology to investigate it
- Considers feasibility and potential impact

Respond in valid JSON format with an array of hypotheses."""


HYPOTHESIS_PROMPT_TEMPLATE = """## Research Query
{query}

## Literature Overview
- Total papers analyzed: {paper_count}
- Methodology clusters found: {cluster_count}

### Cluster Descriptions:
{cluster_descriptions}

## Detected Research Gaps

{gap_descriptions}

## Task

Based on these gaps, generate up to {num_hypotheses} novel research hypotheses. For each hypothesis, provide:

1. **title**: A concise, descriptive title
2. **statement**: The formal hypothesis statement (1-3 sentences)
3. **rationale**: Why this is worth investigating (2-3 sentences)
4. **methodology_suggestion**: Concrete steps to test this hypothesis
5. **feasibility_score**: 0-1 (1 = highly feasible with existing tools/data)
6. **novelty_score**: 0-1 (1 = completely novel direction)
7. **impact_score**: 0-1 (1 = potential for significant scientific impact)
8. **related_gaps**: Which gap indices (0-based) this addresses

Respond ONLY with valid JSON in this format:
```json
{{
  "hypotheses": [
    {{
      "title": "...",
      "statement": "...",
      "rationale": "...",
      "methodology_suggestion": "...",
      "feasibility_score": 0.8,
      "novelty_score": 0.7,
      "impact_score": 0.9,
      "related_gaps": [0, 2]
    }}
  ]
}}
```"""


class HypothesisGenerator:
    """Generates research hypotheses from detected gaps using DeepSeek."""
    
    def __init__(self):
        if not Config.DEEPSEEK_API_KEY:
            raise ValueError(
                "DEEPSEEK_API_KEY not set. Cannot generate hypotheses. "
                "Set it in .env or environment."
            )
        
        self.client = AsyncOpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL,
        )
    
    async def generate(
        self,
        query: str,
        papers: list[Paper],
        clusters: list[Cluster],
        gaps: list[Gap],
        num_hypotheses: int = None,
    ) -> list[Hypothesis]:
        """Generate hypotheses from the analysis results."""
        num_hypotheses = num_hypotheses or Config.TOP_HYPOTHESES
        
        logger.info(f"Generating {num_hypotheses} hypotheses via DeepSeek...")
        
        # Build prompt
        cluster_descriptions = self._format_clusters(clusters)
        gap_descriptions = self._format_gaps(gaps)
        
        prompt = HYPOTHESIS_PROMPT_TEMPLATE.format(
            query=query,
            paper_count=len(papers),
            cluster_count=len(clusters),
            cluster_descriptions=cluster_descriptions,
            gap_descriptions=gap_descriptions,
            num_hypotheses=num_hypotheses,
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=Config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=4000,
            )
            
            content = response.choices[0].message.content
            hypotheses = self._parse_response(content, gaps)
            
            logger.info(f"Generated {len(hypotheses)} hypotheses")
            return hypotheses
            
        except Exception as e:
            logger.error(f"Hypothesis generation failed: {e}")
            return []
    
    def _format_clusters(self, clusters: list[Cluster]) -> str:
        lines = []
        for c in clusters:
            kw_str = ", ".join(c.methodology_keywords[:8])
            lines.append(
                f"- **Cluster {c.cluster_id}** ({c.label}): {len(c.papers)} papers, "
                f"density={c.density:.2f}, keywords=[{kw_str}]"
            )
        return "\n".join(lines)
    
    def _format_gaps(self, gaps: list[Gap]) -> str:
        lines = []
        for i, gap in enumerate(gaps):
            lines.append(
                f"**Gap {i}** (type: {gap.gap_type}, score: {gap.composite_score:.2f}):\n"
                f"{gap.description}\n"
                f"Evidence: {gap.evidence}\n"
            )
        return "\n".join(lines)
    
    def _parse_response(self, content: str, gaps: list[Gap]) -> list[Hypothesis]:
        """Parse LLM response into Hypothesis objects."""
        # Extract JSON from response (handle markdown code blocks)
        json_str = content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0]
        
        try:
            data = json.loads(json_str.strip())
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response as JSON: {content[:500]}")
            return []
        
        hypotheses = []
        for item in data.get("hypotheses", []):
            related_gap_indices = item.get("related_gaps", [])
            evidence_from_gaps = []
            clusters_involved = []
            
            for idx in related_gap_indices:
                if 0 <= idx < len(gaps):
                    evidence_from_gaps.append(gaps[idx].description)
                    clusters_involved.extend(gaps[idx].clusters_involved)
            
            overall = (
                0.35 * item.get("feasibility_score", 0.5) +
                0.35 * item.get("novelty_score", 0.5) +
                0.30 * item.get("impact_score", 0.5)
            )
            
            hypotheses.append(Hypothesis(
                hypothesis_id=str(uuid.uuid4())[:8],
                title=item.get("title", "Untitled Hypothesis"),
                statement=item.get("statement", ""),
                rationale=item.get("rationale", ""),
                methodology_suggestion=item.get("methodology_suggestion", ""),
                evidence_from_gaps=evidence_from_gaps,
                feasibility_score=item.get("feasibility_score", 0.5),
                novelty_score=item.get("novelty_score", 0.5),
                impact_score=item.get("impact_score", 0.5),
                overall_score=overall,
                related_clusters=sorted(set(clusters_involved)),
            ))
        
        # Sort by overall score
        hypotheses.sort(key=lambda h: h.overall_score, reverse=True)
        return hypotheses
