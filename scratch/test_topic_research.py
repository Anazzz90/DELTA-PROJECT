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
    
    test_topic = "was the bengal election rigged"
    print(f"Starting topic research on: {test_topic}")
    
    try:
        result = await engine.run_topic_research(test_topic, domain_profile="politics")
        
        print("\n" + "="*50)
        print("RESEARCH COMPLETED")
        print("="*50)
        
        print(f"\nExtracted Facts ({len(result['extracted_facts'])}):")
        for i, fact in enumerate(result['extracted_facts'], 1):
            print(f"{i}. {fact}")

        print(f"\nConflict Score: {result['conflict_score']}")
        print(f"Sources Found: {len(result['sources'])}")
        for i, s in enumerate(result['sources'], 1):
            print(f" - {s['title']} ({s['url']})")
        
    except Exception as e:
        print(f"\nERROR DURING RESEARCH: {e}")

if __name__ == "__main__":
    asyncio.run(main())
