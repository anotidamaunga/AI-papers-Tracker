# AI Paper Tracker

A Streamlit app for discovering AI research papers, getting an LLM-generated summary, and chatting with the paper's content.

Opens straight into today's trending papers from [Hugging Face Papers](https://huggingface.co/papers), or search by topic keywords, by AI lab, or Hugging Face's own upvote-ranked index.

## Features

- **Trending on open** — the app auto-loads today's top Hugging Face Papers (ranked by upvotes) as soon as it starts, no search needed.
- **Search by topic** — free-text keyword search across arXiv's title/abstract/comments.
- **Search by AI lab** — matches a lab name (OpenAI, Google DeepMind, Anthropic, Meta AI, etc., or a custom name) against arXiv metadata; pair it with a topic keyword for much better precision, since arXiv has no institution field.
- **Search Hugging Face Papers** — searches HF's community-curated papers index by topic, ranked by upvotes, then fetched from arXiv for the full text.
- **Summarize** — downloads the paper's PDF, extracts it to markdown, and asks an LLM (DeepSeek) for a structured summary covering the core innovation, results, and limitations.
- **Chat** — ask follow-up questions about a specific paper; answers are grounded only in that paper's extracted text.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
DEEPSEEK_API_KEY=your-deepseek-api-key
```

Get a key at [platform.deepseek.com](https://platform.deepseek.com). Summarizing and chat won't work without it, but search still does.


## How it works

- **arXiv search** uses the [`arxiv`](https://pypi.org/project/arxiv/) Python package against arXiv's public API.
- **Hugging Face search** hits HF's `daily_papers` and `papers/search` endpoints directly, then cross-references the returned arXiv IDs to fetch full paper metadata — so all three search modes render through the same pipeline.
- **PDF extraction** uses [`pymupdf4llm`](https://pypi.org/project/pymupdf4llm/) to convert the paper's PDF into markdown text fed to the LLM.
- **Summaries and chat** use DeepSeek's OpenAI-compatible API (`deepseek-chat`).

## Notes

- arXiv's public API sets no request timeout on its own, and can occasionally be slow or unresponsive; this app wraps it with a 15-second timeout so a stalled request fails with a clear error instead of hanging.
- AI-lab search is best-effort: arXiv doesn't track author affiliation, so it only matches papers that mention the lab by name in the abstract or comments.
