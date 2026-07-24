import json
import logging
import time
from collections import deque
from typing import Optional, Any, Dict
import aiohttp
import asyncio

from tenacity import retry, wait_exponential, stop_after_attempt

from config.settings import settings
from core.delta_protocol import DeltaProtocol
from core.fact_validator import FactValidator
from llm.router import LLMRouter
from redis import Redis

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple async sliding-window rate limiter (stdlib only).

    Tracks call timestamps in a deque; before each call, drops timestamps
    older than the window and — if at capacity — sleeps until the oldest
    call ages out. Shared across scrape + search since both count against
    the same Firecrawl account limit.
    """

    def __init__(self, max_calls: int, period_seconds: float = 60.0):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            while self._calls and now - self._calls[0] >= self.period_seconds:
                self._calls.popleft()
            if len(self._calls) >= self.max_calls:
                wait_time = self.period_seconds - (now - self._calls[0])
                if wait_time > 0:
                    logger.info(f"Firecrawl rate limit reached, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.period_seconds:
                    self._calls.popleft()
            self._calls.append(time.monotonic())


class ResearchEngine:
    """
    Automated research and truth-filtering pipeline using Firecrawl.
    """
    def __init__(self, redis_conn: Optional[Redis] = None):
        self.firecrawl_key = settings.firecrawl_api_key
        self.router = LLMRouter()
        self.validator = FactValidator()
        self.protocol = DeltaProtocol()
        self.redis_conn = redis_conn
        self._rate_limiter = RateLimiter(max_calls=settings.firecrawl_rate_limit_per_min)

        if not self.firecrawl_key and settings.is_production:
            raise RuntimeError(
                "FIRECRAWL_API_KEY is not set. Refusing to serve mocked research "
                "data in production — set the key or run with ENV=development."
            )

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    async def _scrape_url(self, url: str) -> dict:
        """Crawl the URL using Firecrawl with retry handling."""
        if not self.firecrawl_key:
            logger.warning("FIRECRAWL_API_KEY is not set. Using mocked response.")
            return {
                "data": {
                    "markdown": f"Mocked content for {url}. The market is showing positive momentum.",
                    "metadata": {"sourceURL": url, "mocked": True}
                }
            }

        await self._rate_limiter.acquire()

        headers = {
            "Authorization": f"Bearer {self.firecrawl_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "url": url,
            "formats": ["markdown"]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.firecrawl.dev/v1/scrape", headers=headers, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    async def _search_web(self, query: str) -> dict:
        """Search the web using Firecrawl with retry handling."""
        if not self.firecrawl_key:
            logger.warning("FIRECRAWL_API_KEY is not set. Using mocked search response.")
            return {
                "data": [
                    {
                        "markdown": f"Expert opinion on {query}: The market is currently volatile but showing long-term strength.",
                        "metadata": {"title": f"Expert Analysis on {query}", "sourceURL": "https://example.com/analysis"}
                    },
                    {
                        "markdown": f"Recent report on {query}: Volumes have increased by 20% week-over-week.",
                        "metadata": {"title": f"Market Report: {query}", "sourceURL": "https://example.com/report"}
                    }
                ]
            }

        await self._rate_limiter.acquire()

        headers = {
            "Authorization": f"Bearer {self.firecrawl_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "query": query,
            "limit": 3,
            "scrapeOptions": {"formats": ["markdown"]}
        }

        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.firecrawl.dev/v1/search", headers=headers, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def run_research(self, url: str, domain_profile: Optional[str] = None) -> dict:
        """
        Executes the research flow:
        1. Crawl source
        2. Extract candidate facts
        3. Remove noise
        4. Verify/cross-check claims
        5. Assign confidence tiers
        6. Calculate conflict score
        7. Return structured facts
        """
        cache_key = f"research:{url}"
        cache_status = "miss"
        
        if self.redis_conn:
            try:
                cached = self.redis_conn.get(cache_key)
                if cached:
                    cache_status = "hit"
                    res = json.loads(cached)
                    res["cache_status"] = cache_status
                    return res
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")

        # 1. Crawl source
        try:
            crawl_data = await self._scrape_url(url)
            markdown_content = crawl_data.get("data", {}).get("markdown", "")
            crawl_metadata = crawl_data.get("data", {}).get("metadata", {})
        except Exception as e:
            logger.error(f"Firecrawl scrape failed: {e}")
            raise RuntimeError(f"Failed to crawl {url}: {e}")

        # 2. Extract candidate facts
        try:
            ext_prompt = self.protocol.render(
                agent_name="fact_extractor",
                question="Extract facts",
                fact_set=[markdown_content[:20000]], # Truncate to avoid context limit
                domain_profile=domain_profile
            )
            
            ext_response = self.router.call_model_direct(
                model="deepseek-ai/DeepSeek-V3",
                system_prompt=ext_prompt.system,
                user_prompt=ext_prompt.user,
                temperature=0.0,
                timeout=120
            )
        except Exception as e:
            logger.error(f"Failed to render extractor prompt: {e}")
            ext_response = None
        
        candidate_facts = []
        if ext_response and ext_response.success:
            try:
                raw_text = ext_response.content.strip()
                if raw_text.startswith("```"):
                    lines = raw_text.split("\n")
                    raw_text = "\n".join(line for line in lines if not line.strip().startswith("```")).strip()
                ext_json = json.loads(raw_text)
                candidate_facts = ext_json.get("extracted_facts", [])
            except Exception as e:
                logger.error(f"Failed to parse extractor JSON: {e}")
                
        if not candidate_facts:
             candidate_facts = ["Data unavailable or extraction failed."]

        # 3. & 4. & 5. Remove noise & verify claims via FactValidator
        try:
            val_result = self.validator.validate(candidate_facts)
            clean_facts = val_result.to_plain_list()
            tiers = [{"fact": f.original, "tier": f.tier.value} for f in val_result.facts]
        except Exception as e:
            logger.warning(f"FactValidator rejected some or all facts: {e}")
            clean_facts = candidate_facts
            tiers = [{"fact": f, "tier": "uncertain"} for f in candidate_facts]

        # 6. Truth filter for contradictions
        try:
            tf_prompt = self.protocol.render(
                agent_name="truth_filter",
                question="Check for contradictions",
                fact_set=clean_facts,
                domain_profile=domain_profile
            )
            
            tf_response = self.router.call_model_direct(
                model="deepseek-ai/DeepSeek-V3",
                system_prompt=tf_prompt.system,
                user_prompt=tf_prompt.user,
                temperature=0.0,
                timeout=120
            )
        except Exception as e:
            logger.error(f"Failed to render truth filter prompt: {e}")
            tf_response = None
        
        conflict_score = 0.0
        contradictions = []
        if tf_response and tf_response.success:
            try:
                raw_text = tf_response.content.strip()
                if raw_text.startswith("```"):
                    lines = raw_text.split("\n")
                    raw_text = "\n".join(line for line in lines if not line.strip().startswith("```")).strip()
                tf_json = json.loads(raw_text)
                conflict_score = float(tf_json.get("conflict_score", 0.0))
                contradictions = tf_json.get("contradictions_found", [])
            except Exception as e:
                logger.error(f"Failed to parse truth filter JSON: {e}")

        result = {
            "extracted_facts": clean_facts,
            "confidence_tiers": tiers,
            "contradictions": contradictions,
            "conflict_score": conflict_score,
            "cache_status": cache_status,
            "crawl_metadata": crawl_metadata
        }
        
        if self.redis_conn:
            try:
                self.redis_conn.setex(cache_key, 86400, json.dumps(result))
            except Exception as e:
                logger.warning(f"Redis cache write failed: {e}")
                
        return result

    async def run_topic_research(self, topic: str, domain_profile: Optional[str] = None) -> dict:
        """
        Performs research on a broad topic by searching the web and aggregating results.
        """
        cache_key = f"research:topic:{topic}"
        if self.redis_conn:
            try:
                cached = self.redis_conn.get(cache_key)
                if cached:
                    res = json.loads(cached)
                    res["cache_status"] = "hit"
                    return res
            except Exception: pass

        # 1. Search web
        logger.debug(f"Searching web for topic: {topic}")
        try:
            search_data = await self._search_web(topic)
            results = search_data.get("data", [])
            logger.debug(f"Found {len(results)} search results")
        except Exception as e:
            logger.error(f"Firecrawl search failed: {e}")
            raise RuntimeError(f"Failed to search for topic '{topic}': {e}")

        if not results:
            raise ValueError(f"No search results found for topic: {topic}")

        # 2. Aggregate content
        aggregated_markdown = ""
        sources = []
        for i, res in enumerate(results, 1):
            title = res.get("metadata", {}).get("title", f"Result {i}")
            url = res.get("metadata", {}).get("sourceURL", "")
            markdown = res.get("markdown", "")
            aggregated_markdown += f"\n--- SOURCE {i}: {title} ({url}) ---\n{markdown[:5000]}\n"
            sources.append({"title": title, "url": url})

        # 3. Extract and filter (reuse run_research logic but with aggregated content)
        # We'll pass the aggregated content as a single block to the research flow
        # To avoid duplicating logic, we can refactor or just call the core steps.
        
        # We trick run_research by passing a "virtual URL" or just refactor.
        # Let's just run the extraction directly here for now to keep it clean.
        
        # [Step: Extraction]
        ext_prompt = self.protocol.render(
            agent_name="fact_extractor",
            question=f"Extract facts about {topic}",
            fact_set=[aggregated_markdown[:15000]],
            domain_profile=domain_profile
        )
        logger.debug(f"Calling DeepSeek for fact extraction (chars={len(aggregated_markdown)})")
        ext_response = self.router.call_model_direct(
            model="deepseek-ai/DeepSeek-V3",
            system_prompt=ext_prompt.system,
            user_prompt=ext_prompt.user,
            temperature=0.0,
            timeout=120
        )

        candidate_facts = []
        if ext_response and ext_response.success:
            try:
                raw_text = ext_response.content.strip()
                if raw_text.startswith("```"):
                    lines = raw_text.split("\n")
                    raw_text = "\n".join(line for line in lines if not line.strip().startswith("```")).strip()
                ext_json = json.loads(raw_text)
                candidate_facts = ext_json.get("extracted_facts", [])
                logger.debug(f"Extracted {len(candidate_facts)} candidate facts")
            except Exception as e:
                logger.error(f"Failed to parse extractor JSON: {e}")
        else:
            err = ext_response.error if ext_response else "No response"
            logger.error(f"DeepSeek extraction failed: {err}")
            
        if not candidate_facts:
            candidate_facts = [f"No specific facts could be extracted for '{topic}'."]

        # [Step: Validation & Tiers]
        try:
            val_result = self.validator.validate(candidate_facts)
            clean_facts = val_result.to_plain_list()
            tiers = [{"fact": f.original, "tier": f.tier.value} for f in val_result.facts]
        except Exception:
            clean_facts = candidate_facts
            tiers = [{"fact": f, "tier": "uncertain"} for f in candidate_facts]

        # [Step: Truth Filtering]
        tf_prompt = self.protocol.render(
            agent_name="truth_filter",
            question=f"Contradiction check for {topic}",
            fact_set=clean_facts,
            domain_profile=domain_profile
        )
        tf_response = self.router.call_model_direct(
            model="deepseek-ai/DeepSeek-V3",
            system_prompt=tf_prompt.system,
            user_prompt=tf_prompt.user,
            temperature=0.0,
            timeout=120
        )
        
        conflict_score = 0.0
        contradictions = []
        if tf_response and tf_response.success:
            try:
                raw_text = tf_response.content.strip()
                if raw_text.startswith("```"):
                    lines = raw_text.split("\n")
                    raw_text = "\n".join(line for line in lines if not line.strip().startswith("```")).strip()
                tf_json = json.loads(raw_text)
                conflict_score = float(tf_json.get("conflict_score", 0.0))
                contradictions = tf_json.get("contradictions_found", [])
            except Exception: pass

        result = {
            "topic": topic,
            "extracted_facts": clean_facts,
            "confidence_tiers": tiers,
            "contradictions": contradictions,
            "conflict_score": conflict_score,
            "cache_status": "miss",
            "sources": sources,
            "crawl_metadata": {"results_count": len(results)}
        }
        
        if self.redis_conn:
            try:
                self.redis_conn.setex(cache_key, 86400, json.dumps(result))
            except Exception: pass
            
        return result
