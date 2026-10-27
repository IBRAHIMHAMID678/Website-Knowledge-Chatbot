# Assessment Write-Up: Next Week's Roadmap & Known Limitations

## What I Would Improve With Another Week

1. **Auto-updating data instead of manual scraping:**  
   Right now, the bot indexes the pages when triggered. If I had another week, I would build a lightweight background job that checks if PlacidWay pages changed (using `ETag` or `Last-Modified` headers) and automatically re-crawls only the pages that were updated, without re-crawling the entire site.

2. **Reranker for scaling to thousands of pages:**  
   Our hybrid search (keywords + embeddings) works accurately for the 7 target pages. But if we scale this to thousands of pages across PlacidWay, I would add a reranking step (like Cohere Rerank or BGE). That way, we can pull the top 20–30 relevant chunks and let the reranker pick the best context before passing it to the LLM.

3. **Connecting lead capture directly to CRM or email:**  
   The bot already collects patient contact info and saves leads to the database. With more time, I would connect this directly to PlacidWay’s CRM (like HubSpot or Salesforce) or trigger a webhook/email alert so patient coordinators receive the lead and chat summary immediately.

4. **Automated accuracy checks in CI/CD:**  
   I would set up automated testing (using tools like Ragas) in GitHub Actions to score the bot's factual accuracy, grounding, and relevance on every pull request.

---

## Known Limitations

1. **Strictly bounded to the 7 provided URLs:**  
   The bot is deliberately designed not to make things up or pull unverified info from the open web. It only knows what is on the 7 PlacidWay pages provided. If someone asks about procedures or countries outside this data (like dental in Colombia or IVF in Greece), the bot honestly says it does not have that information and directs them to PlacidWay coordinators.

2. **Custom cancer pricing requires doctor review:**  
   The bot quotes published package prices accurately ($18,995 Out-Patient and $30,000 In-Patient at ITC Tijuana, $64 in Thailand, and $110 in Turkey). However, real cancer treatments vary based on a patient's medical history and staging, so the bot cannot guess custom treatment costs and directs patients to request an official medical evaluation.

3. **No direct calendar booking:**  
   The bot does not have live access to clinic scheduling systems or doctor calendars, so it cannot book an appointment for tomorrow. Instead, it collects the user's details through the quote form so coordinators can arrange the booking.
