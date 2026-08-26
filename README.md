# AI Paper Tracker

A Streamlit app for keeping up with AI research without doom-scrolling arXiv. It opens straight into today's trending papers, lets you search a topic, and once you open a paper it downloads the PDF, summarizes it, and lets you ask it questions directly.

## How search works

arXiv is good at keyword matching but has no notion of what's actually worth reading. [Hugging Face Papers](https://huggingface.co/papers) has the opposite problem: it's a community-curated, upvoted feed, but it's not a general search index. So a topic search runs against both and merges them — anything upvoted on Hugging Face surfaces first, with the rest of the arXiv relevance ranking filling in behind it. That's also what's showing when the app first opens: today's top Hugging Face papers, no search needed.

There's no dedicated "search by lab" mode. Doing that properly would mean scraping each lab's own publications page — arXiv doesn't carry author affiliation as searchable metadata, so a lab name is only findable when a paper happens to mention it in its abstract or comments, which misses most papers. Not worth pretending that's a real filter.

## Quick start

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with a [DeepSeek](https://platform.deepseek.com) key (summarizing and chat need it; search works without one):

```
DEEPSEEK_API_KEY=your-deepseek-api-key
```

Run it — note the filename has a space, so quote it:

```bash
streamlit run "AI-paper Tracker.py"
```

## Under the hood

- Opening a paper downloads its arXiv PDF and runs it through [`pymupdf4llm`](https://pypi.org/project/pymupdf4llm/) to get clean markdown for the LLM.
- Summaries and chat both go through DeepSeek's OpenAI-compatible API (`deepseek-chat`); chat is answered strictly from the extracted paper text, not general knowledge.
- The `arxiv` package sets no HTTP timeout on its own requests, and arXiv's API is occasionally slow enough that this matters in practice — the app wraps every arXiv request with an explicit 15s timeout so a stalled request fails fast instead of hanging the UI.
