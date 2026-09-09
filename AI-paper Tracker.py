import datetime
import os
import re
import tempfile

import arxiv
import pymupdf4llm
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)


ARXIV_REQUEST_TIMEOUT_SECONDS = 15


class _TimeoutSession(requests.Session):
    """arxiv.Client's session.get() call sets no timeout, so a slow/unresponsive
    arXiv API can hang indefinitely. This forces every request to time out."""

    def request(self, *args, **kwargs):
        kwargs.setdefault("timeout", ARXIV_REQUEST_TIMEOUT_SECONDS)
        return super().request(*args, **kwargs)


def _arxiv_client(num_retries=3, delay_seconds=3):
    client = arxiv.Client(num_retries=num_retries, delay_seconds=delay_seconds)
    client._session = _TimeoutSession()
    return client


DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

SUMMARY_SYSTEM_PROMPT = (
    "Summarise this research paper for an Applied AI university student. "
    "Explain the core innovation, experimental results and potential flaws or limitations. "
    "Use short headed sections and keep it under 300 words."
)
CHAT_SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert researcher discussing a specific paper with the user. "
    "Answer ONLY using the paper text below; if the answer isn't in the paper, say so.\n\n"
    "--- PAPER TEXT (truncated) ---\n{paper_text}"
)

CATEGORIES = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.CL": "Computation and Language",
    "cs.CV": "Computer Vision",
    "cs.NE": "Neural and Evolutionary Computing",
    "stat.ML": "Statistics - Machine Learning",
}

MAX_CHARS_FOR_LLM = 20000


@st.cache_resource(show_spinner=False)
def get_client(api_key):
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def call_deepseek(messages):
    client = get_client(DEEPSEEK_API_KEY)
    response = client.chat.completions.create(model="deepseek-chat", messages=messages)
    return response.choices[0].message.content


def summarize_paper(paper_text):
    return call_deepseek(
        [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": paper_text[:MAX_CHARS_FOR_LLM]},
        ]
    )


def chat_about_paper(paper_text, chat_history, question):
    system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(paper_text=paper_text[:MAX_CHARS_FOR_LLM])
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": question})
    return call_deepseek(messages)


def search_arxiv(query, max_results, sort_by, categories):
    if categories:
        cat_filter = " OR ".join(f"cat:{c}" for c in categories)
        query = f"({query}) AND ({cat_filter})"

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
        sort_order=arxiv.SortOrder.Descending,
    )
    return list(_arxiv_client().results(search))


HF_DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"
HF_PAPERS_SEARCH_URL = "https://huggingface.co/api/papers/search"


def _strip_arxiv_version(short_id):
    return re.sub(r"v\d+$", "", short_id)


def search_huggingface(topic_query, max_results, categories):
    """Pulls from huggingface.co/papers (community-curated, upvoted) rather than
    arXiv's own index, then fetches the matching papers' full metadata from arXiv
    so they go through the same render/summarize pipeline. Topic blank = this
    week's trending papers; otherwise searches HF's full papers index by keyword."""
    if topic_query:
        resp = requests.get(
            HF_PAPERS_SEARCH_URL, params={"q": topic_query}, timeout=ARXIV_REQUEST_TIMEOUT_SECONDS
        )
    else:
        # A single day's cohort is often too thin early in the day/week for a
        # meaningful "trending" ranking. The endpoint accepts an ISO week
        # (YYYY-Www) to scope to the current week instead.
        iso_year, iso_week, _ = datetime.datetime.now(datetime.timezone.utc).date().isocalendar()
        week = f"{iso_year}-W{iso_week:02d}"
        resp = requests.get(HF_DAILY_PAPERS_URL, params={"week": week}, timeout=ARXIV_REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    entries = resp.json()
    entries.sort(key=lambda e: e.get("paper", {}).get("upvotes", 0), reverse=True)

    arxiv_ids = []
    upvotes_by_id = {}
    for entry in entries:
        arxiv_id = entry.get("paper", {}).get("id")
        if not arxiv_id or arxiv_id in upvotes_by_id:
            continue
        arxiv_ids.append(arxiv_id)
        upvotes_by_id[arxiv_id] = entry.get("paper", {}).get("upvotes", 0)
        if len(arxiv_ids) >= max_results:
            break

    if not arxiv_ids:
        return [], {}

    results = list(_arxiv_client().results(arxiv.Search(id_list=arxiv_ids)))
    if categories:
        results = [r for r in results if any(c in r.categories for c in categories)]


    # upvotes are the relevance signal for this mode, and display order must reflect it.
    results.sort(key=lambda r: upvotes_by_id.get(_strip_arxiv_version(r.get_short_id()), 0), reverse=True)

    hf_meta = {}
    for r in results:
        base_id = _strip_arxiv_version(r.get_short_id())
        if base_id in upvotes_by_id:
            hf_meta[r.entry_id] = {
                "upvotes": upvotes_by_id[base_id],
                "hf_url": f"https://huggingface.co/papers/{base_id}",
            }
    return results, hf_meta


def search_combined(topic_query, max_results, categories):
    """Runs the same topic query against arXiv and Hugging Face Papers, merges
    the two result sets."""
    issues = []

    try:
        arxiv_results = search_arxiv(
            f"all:{topic_query}", max_results, arxiv.SortCriterion.Relevance, categories
        )
    except Exception as e:
        arxiv_results = []
        issues.append(f"arXiv search failed: {e}")

    try:
        hf_results, hf_meta = search_huggingface(topic_query, max_results, categories)
    except Exception as e:
        hf_results, hf_meta = [], {}
        issues.append(f"Hugging Face search failed: {e}")

    combined = {}
    order = []
    for r in hf_results + arxiv_results:
        base_id = _strip_arxiv_version(r.get_short_id())
        if base_id not in combined:
            combined[base_id] = r
            order.append(base_id)

    upvotes_by_base_id = {
        _strip_arxiv_version(r.get_short_id()): hf_meta[r.entry_id]
        for r in hf_results
        if r.entry_id in hf_meta
    }

    # Stable sort: upvoted papers first (highest upvotes first), everything
    # else keeps its original relevance-ranked order from arXiv.
    order.sort(key=lambda base_id: upvotes_by_base_id.get(base_id, {}).get("upvotes", -1), reverse=True)
    order = order[:max_results]

    results = [combined[base_id] for base_id in order]
    final_hf_meta = {
        combined[base_id].entry_id: upvotes_by_base_id[base_id]
        for base_id in order
        if base_id in upvotes_by_base_id
    }
    return results, final_hf_meta, issues


def extract_paper_text(result):
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = os.path.join(tmp_dir, "paper.pdf")
        resp = requests.get(result.pdf_url, timeout=ARXIV_REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        with open(pdf_path, "wb") as f:
            f.write(resp.content)
        return pymupdf4llm.to_markdown(pdf_path)


def inject_search_engine_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .search-hero h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            text-align: center;
            font-size: 3rem;
            margin-bottom: 0.1rem;
        }
        .search-hero p {
            text-align: center;
            color: #9a9aa8;
            margin-top: 0;
            margin-bottom: 1.6rem;
        }

        /* Pill-shaped text input, like a search engine */
        div[data-testid="stTextInput"] input {
            border-radius: 999px !important;
            padding: 0.7rem 1.2rem !important;
            border: 1px solid #33333f !important;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #FF4B4B !important;
            box-shadow: 0 0 0 1px #FF4B4B !important;
        }

        div[data-testid="stExpander"] {
            border-radius: 12px;
            border: 1px solid #2A2A3E;
        }

        /* Rounded result cards (st.container(border=True)) */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state():
    """Idempotent — safe to call at the top of every page render."""
    st.session_state.setdefault("paper_cache", {})
    st.session_state.setdefault("chat_histories", {})
    st.session_state.setdefault("results", [])
    st.session_state.setdefault("results_by_id", {})
    st.session_state.setdefault("hf_meta", {})
    st.session_state.setdefault("auto_loaded", False)
    st.session_state.setdefault("results_are_trending", False)
    st.session_state.setdefault("selected_paper_id", None)




def render_result_card(result, hf_info, rank):
    entry_id = result.entry_id
    st.session_state.results_by_id[entry_id] = result
    authors = ", ".join(a.name for a in result.authors)
    title_prefix = f"#{rank}  " if rank else ""

    with st.container(border=True):
        card_col, action_col = st.columns([6, 1])
        with card_col:
            st.markdown(f"**{title_prefix}{result.title}**")
            st.caption(f"{result.published.date()} · {authors}")
            if hf_info:
                st.caption(f" {hf_info['upvotes']} upvotes on Hugging Face Papers")
            abstract = result.summary.replace("\n", " ")
            preview = abstract if len(abstract) <= 280 else abstract[:280].rsplit(" ", 1)[0] + "…"
            st.write(preview)
        with action_col:
            st.write("")
            st.write("")
            if st.button("Open →", key=f"open_{entry_id}", use_container_width=True):
                st.session_state.selected_paper_id = entry_id
                st.switch_page(PAPER_PAGE)


def search_page():
    inject_search_engine_css()
    init_session_state()

    if not DEEPSEEK_API_KEY:
        st.error("DEEPSEEK_API_KEY is not set. Add it to your .env file to enable summaries and chat.")

    st.markdown(
        """
        <div class="search-hero">
            <h1>AI Paper Tracker</h1>
            <p>Search a topic — pulls matches from arXiv and surfaces community-upvoted picks from Hugging Face Papers.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_pad, center, right_pad = st.columns([1, 2, 1])
    with center:
        query_input = st.text_input(
            "Topic / keywords",
            placeholder="Search for a topic — e.g. retrieval augmented generation",
            label_visibility="collapsed",
        )
        search_clicked = st.button("Search", type="primary", use_container_width=True)

        with st.expander("Advanced options"):
            selected_categories = st.multiselect(
                "Restrict to categories (optional)",
                options=list(CATEGORIES.keys()),
                format_func=lambda c: f"{c} — {CATEGORIES[c]}",
            )
            max_results = st.slider("Max results", min_value=5, max_value=50, value=20)

    if search_clicked:
        st.session_state.auto_loaded = True
        st.session_state.hf_meta = {}
        if not query_input:
            st.warning("Enter a topic to search.")
        else:
            st.session_state.results_are_trending = False
            with st.spinner("Searching arXiv and Hugging Face Papers..."):
                results, hf_meta, issues = search_combined(query_input, max_results, selected_categories)
                st.session_state.results = results
                st.session_state.hf_meta = hf_meta
                for issue in issues:
                    st.warning(issue)
                if not results and not issues:
                    st.warning("No papers found — try different keywords.")
    elif not st.session_state.auto_loaded:
        st.session_state.auto_loaded = True
        with st.spinner("Loading this week's trending papers from Hugging Face..."):
            try:
                results, hf_meta = search_huggingface("", 10, [])
                st.session_state.results = results
                st.session_state.hf_meta = hf_meta
                st.session_state.results_are_trending = True
            except Exception as e:
                st.warning(f"Couldn't load trending papers automatically: {e}")

    if not st.session_state.results:
        st.info("Search a topic above to see papers here.")
        return

    st.subheader(
        "Trending this week" if st.session_state.results_are_trending else f"{len(st.session_state.results)} papers found"
    )
    ranked_by_upvotes = bool(st.session_state.hf_meta)
    for i, result in enumerate(st.session_state.results, start=1):
        rank = i if ranked_by_upvotes else None
        render_result_card(result, st.session_state.hf_meta.get(result.entry_id), rank)



def paper_detail_page():
    inject_search_engine_css()
    init_session_state()

    if st.button("← Back to search"):
        st.switch_page(SEARCH_PAGE)

    entry_id = st.session_state.selected_paper_id
    result = st.session_state.results_by_id.get(entry_id) if entry_id else None

    if not result:
        st.info("No paper selected yet — go back and open one from the search results.")
        return

    authors = ", ".join(a.name for a in result.authors)
    st.title(result.title)
    st.caption(f"{result.published.date()} · {authors}")

    hf_info = st.session_state.hf_meta.get(entry_id)
    if hf_info:
        st.markdown(f"🤗 **{hf_info['upvotes']} upvotes** on [Hugging Face Papers]({hf_info['hf_url']})")
    st.markdown(f"**Categories:** {', '.join(result.categories)}")
    st.markdown(result.summary.replace("\n", " "))
    st.markdown(f"[Abstract page]({result.entry_id}) · [PDF]({result.pdf_url})")

    st.divider()

    is_processed = entry_id in st.session_state.paper_cache
    if not is_processed:
        if st.button("Summarize this paper", type="primary"):
            with st.spinner("Downloading PDF and generating summary..."):
                try:
                    text = extract_paper_text(result)
                    summary = summarize_paper(text)
                    st.session_state.paper_cache[entry_id] = {"text": text, "summary": summary}
                    st.session_state.chat_histories.setdefault(entry_id, [])
                except Exception as e:
                    st.error(f"Failed to process paper: {e}")
                    return
            st.rerun()
        return

    cached = st.session_state.paper_cache[entry_id]
    st.subheader("Summary")
    st.markdown(cached["summary"])

    st.subheader("Discuss this paper")
    history = st.session_state.chat_histories.setdefault(entry_id, [])
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    question = st.chat_input("Ask a question about this paper...")
    if question:
        history.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            try:
                answer = chat_about_paper(cached["text"], history[:-1], question)
            except Exception as e:
                answer = f"Error: {e}"
        history.append({"role": "assistant", "content": answer})
        st.rerun()



st.set_page_config(page_title="AI Paper Tracker", layout="wide")
init_session_state()


SEARCH_PAGE = st.Page(search_page, title="Search", icon="🔍", default=True)
PAPER_PAGE = st.Page(paper_detail_page, title="Paper", icon="📄")

pg = st.navigation([SEARCH_PAGE, PAPER_PAGE], position="hidden")
pg.run()