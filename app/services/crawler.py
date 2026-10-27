import asyncio
import time
import json
import logging
from typing import List, Dict, Any, Tuple
import urllib.robotparser
import httpx
from app.config import settings
from app.services.parser import parse_placidway_html, chunk_document

logger = logging.getLogger(__name__)

async def fetch_url(client: httpx.AsyncClient, url: str) -> Tuple[str, str, int]:
    """Fetches a single URL with custom headers and error handling."""
    try:
        response = await client.get(url, timeout=25.0)
        return url, response.text, response.status_code
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return url, "", 500

async def crawl_all_target_urls() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """
    Crawls strictly the 7 target PlacidWay URLs at ~1 req/sec with polite User-Agent,
    respecting robots.txt rules.
    Returns:
        parsed_pages: List of cleaned page documents
        all_chunks: List of chunk dictionaries
        errors: List of error messages if any
    """
    headers = {
        "User-Agent": settings.CRAWLER_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    parsed_pages = []
    all_chunks = []
    errors = []
    
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        # Check and parse robots.txt for compliance
        rp = urllib.robotparser.RobotFileParser()
        try:
            _, robots_txt, r_status = await fetch_url(client, "https://www.placidway.com/robots.txt")
            if r_status == 200 and robots_txt:
                rp.parse(robots_txt.splitlines())
                logger.info("Successfully loaded and parsed PlacidWay robots.txt rules.")
            else:
                rp.allow_all = True
        except Exception as e:
            logger.warning(f"Could not load robots.txt, defaulting to polite crawl: {e}")
            rp.allow_all = True

        for idx, url in enumerate(settings.TARGET_URLS, 1):
            if not rp.can_fetch(settings.CRAWLER_USER_AGENT, url):
                logger.warning(f"Skipping {url} as disallowed by robots.txt")
                continue

            logger.info(f"Crawling [{idx}/{len(settings.TARGET_URLS)}]: {url}")
            url, html, status = await fetch_url(client, url)
            
            if status == 200 and html:
                try:
                    parsed = parse_placidway_html(html, url)
                    parsed_pages.append(parsed)
                    
                    # Save raw/clean cached page for offline audit
                    safe_filename = "".join(c if c.isalnum() else "_" for c in url.split("/")[-1])[:50]
                    cache_file = settings.SCRAPED_PAGES_DIR / f"{safe_filename}.json"
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(parsed, f, indent=2, ensure_ascii=False)
                        
                    chunks = chunk_document(parsed)
                    all_chunks.extend(chunks)
                    logger.info(f"Successfully processed {url}: {len(chunks)} chunks")
                except Exception as e:
                    err_msg = f"Failed to parse {url}: {str(e)}"
                    logger.error(err_msg)
                    errors.append(err_msg)
            else:
                err_msg = f"Failed to fetch {url} (HTTP {status})"
                logger.error(err_msg)
                errors.append(err_msg)
                
            # Respect rate limit of ~1 request per second
            if idx < len(settings.TARGET_URLS):
                await asyncio.sleep(settings.CRAWL_DELAY_SECONDS)
                
    return parsed_pages, all_chunks, errors
