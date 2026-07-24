import asyncio
import os
import json
from dotenv import load_dotenv

# Load env vars first
load_dotenv()

from core.research_engine import ResearchEngine

async def main():
    print("Initializing Research Engine...")
    engine = ResearchEngine(redis_conn=None) # Skip redis for direct test
    
    test_url = "https://www.firecrawl.dev/blog/announcing-v1"
    print(f"Starting research on: {test_url}")
    
    try:
        result = await engine.run_research(test_url, domain_profile="AI Scraping")
        
        print("\n" + "="*50)
        print("RESEARCH COMPLETED SUCCESSFULLY")
        print("="*50)
        
        print(f"\nExtracted Facts ({len(result['extracted_facts'])}):")
        for i, fact in enumerate(result['extracted_facts'][:5], 1):
            print(f"{i}. {fact}")
        if len(result['extracted_facts']) > 5:
            print("...")

        print(f"\nConflict Score: {result['conflict_score']}")
        print(f"Contradictions Found: {len(result['contradictions'])}")
        
        print("\nCrawl Metadata:")
        print(json.dumps(result['crawl_metadata'], indent=2))
        
    except Exception as e:
        print(f"\nERROR DURING RESEARCH: {e}")

if __name__ == "__main__":
    asyncio.run(main())
