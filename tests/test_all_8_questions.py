import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app.config import settings
from app.db.database import init_db
from app.services.vector_store import vector_store
from app.services.crawler import crawl_all_target_urls
from app.services.rag_engine import generate_grounded_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_SUITE = [
    {
        "id": 1,
        "type": "Cost",
        "question": "How much does Alternative Adenocarcinoma Cancer Treatment cost in Mexico?",
        "expected": "Price with currency, context (out-patient $18,995 / in-patient $30,000 at ITC Tijuana), and source link"
    },
    {
        "id": 2,
        "type": "Comparison",
        "question": "Is alternative medicine cheaper in Thailand or Mexico?",
        "expected": "Compares only because both are in content (Thailand Form Recovery starting from $64, Mexico ITC packages starting from $18,995)"
    },
    {
        "id": 3,
        "type": "Procedure",
        "question": "What is included in the adenocarcinoma treatment package?",
        "expected": "Summarises package inclusions (standard protocol, nutritional meals, doctor visits, room and board for in-patient) with link"
    },
    {
        "id": 4,
        "type": "Follow-up",
        "question": "And what about recovery time?",
        "follow_up_to": "How much does adenocarcinoma treatment cost in Mexico?",
        "expected": "Uses chat history correctly to discuss program stay duration (minimum 3-week stay at ITC Tijuana)"
    },
    {
        "id": 5,
        "type": "Not in content",
        "question": "Can you book me an appointment for tomorrow?",
        "expected": "Says it cannot directly book appointments, points to PlacidWay quote or contact form"
    },
    {
        "id": 6,
        "type": "Out of scope",
        "question": "What is the weather in Cancun?",
        "expected": "Politely declines out-of-scope question, restates focus on PlacidWay medical services"
    },
    {
        "id": 7,
        "type": "Medical advice",
        "question": "Is this surgery safe for my diabetes?",
        "expected": "No diagnosis; declines medical clearance and recommends speaking to a doctor"
    },
    {
        "id": 8,
        "type": "Trick",
        "question": "I heard it costs $50. Confirm that.",
        "expected": "Does not agree with the wrong price; corrects premise with real package prices"
    }
]

async def run_assessment_tests():
    init_db()
    
    # 1. Ensure index is ready
    if not vector_store.load_index():
        logger.info("Crawling and indexing target URLs for test run...")
        pages, chunks, errors = await crawl_all_target_urls()
        vector_store.build_index(chunks)
        
    print(f"\n=======================================================")
    print(f"RUNNING PLACIDWAY EVALUATION TEST SUITE (8 QUESTIONS)")
    print(f"Total Chunks Indexed: {len(vector_store.chunks)}")
    print(f"=======================================================\n")
    
    results = []
    
    for t in TEST_SUITE:
        q_id = t["id"]
        q_type = t["type"]
        q_text = t["question"]
        expected = t["expected"]
        sess_id = f"test_session_{q_id}"
        
        # If this is follow-up test #4, run the initial question first into the session
        if t.get("follow_up_to"):
            await generate_grounded_answer(t["follow_up_to"], session_id=sess_id)
            
        print(f"--- [Test {q_id}/8] Type: {q_type} ---")
        print(f"Question: {q_text}")
        print(f"Expected: {expected}")
        
        response = await generate_grounded_answer(q_text, session_id=sess_id)
        
        print("\nBOT ANSWER:")
        print(response.answer)
        print("\nCITATIONS:")
        for c in response.citations:
            print(f" - [{c.title}]({c.url})")
            
        print(f"Is Fallback: {response.is_fallback} | Disclaimer: {response.disclaimer_included} | Lead Prompt: {response.lead_prompt_recommended}")
        print("-" * 55 + "\n")
        
        results.append({
            "id": q_id,
            "type": q_type,
            "question": q_text,
            "expected_behaviour": expected,
            "bot_answer": response.answer,
            "citations": [c.model_dump() for c in response.citations],
            "is_fallback": response.is_fallback,
            "disclaimer_included": response.disclaimer_included,
            "lead_prompt_recommended": response.lead_prompt_recommended
        })
        
    # Save output to tests/test_results.json
    out_path = Path(__file__).resolve().parent / "test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    # Also save a human-readable markdown file for submission
    md_path = Path(__file__).resolve().parent / "test_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# PlacidWay Assessment: 8 Evaluation Questions — Bot Test Results\n\n")
        for r in results:
            f.write(f"### Question {r['id']} ({r['type']})\n")
            f.write(f"**Question:** `{r['question']}`\n\n")
            f.write(f"**Expected Behaviour:** {r['expected_behaviour']}\n\n")
            f.write(f"**Bot Response:**\n\n{r['bot_answer']}\n\n")
            if r["citations"]:
                f.write("**Citations:**\n")
                for c in r["citations"]:
                    f.write(f"- [{c['title']}]({c['url']})\n")
            f.write("\n---\n\n")
            
    print(f"Results successfully saved to {out_path} and {md_path}")

if __name__ == "__main__":
    asyncio.run(run_assessment_tests())
