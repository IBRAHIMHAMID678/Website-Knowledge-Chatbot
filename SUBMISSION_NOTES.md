# Assessment Write-Up: Next Week's Roadmap & Known Limitations

## What I Would Improve With Another Week

1. **Automated Continuous Re-Ingestion with ETag & Conditional Headers:**  
   Currently, on-demand re-indexing occurs via `/api/ingest/refresh`. With another week, I would deploy an asynchronous worker that polls `ETag` and `Last-Modified` HTTP headers on a scheduled basis, re-crawling and re-embedding only the specific clinic pages that have updated, eliminating site-wide overhead.

2. **Two-Stage Retrieval with Cross-Encoder Reranking (Cohere / BGE-Reranker):**  
   While our hybrid dense + sparse vector search achieves 100% precision for the 7 target URLs, scaling to 10,000+ PlacidWay treatment pages requires a two-stage retrieval pipeline: retrieve the top 30 candidate chunks with vector search, followed by cross-encoder reranking to optimize semantic context relevance before passing to the LLM.

3. **Direct CRM & Lead Workflow Integration:**  
   Connect the lead capture modal to an external CRM endpoint (e.g., Salesforce, HubSpot, or secure webhook) to automatically route high-intent patient inquiries, preferred contact times, and chat session transcripts directly to PlacidWay clinical coordinators.

4. **Automated Continuous Evaluation Pipeline (RAGAS / TruLens):**  
   Implement an automated CI/CD evaluation step on code pushes to continuously score **Faithfulness**, **Answer Relevance**, and **Context Recall** against golden evaluation benchmarks.

---

## Known Limitations

1. **Strict 7-URL Content Boundary:**  
   The bot is deliberately constrained to the 7 verified PlacidWay URLs provided in the assessment. Inquiries regarding destinations, clinics, or procedures outside these verified pages (e.g., cosmetic surgery in Colombia or IVF in Greece) trigger an honest fallback and refer the patient to PlacidWay coordinators.

2. **Individualized Clinical & Diagnostic Pricing:**  
   Published international treatment costs serve as baseline tiers (such as the $18,995 Out-Patient and $30,000 In-Patient cancer programs at ITC Tijuana, $64 in Thailand, and $110 in Turkey). Because oncology regimens frequently require adjustments based on staging, pathology, and diagnostic bloodwork, the bot cannot quote customized clinical pricing without a physician consultation.

3. **External Scheduling Limitations:**  
   The bot cannot directly reserve or confirm appointment slots in clinic calendars, as it lacks real-time calendar and EHR integration. It directs patients to request a free quote or contact PlacidWay coordinators to initiate the scheduling process.
