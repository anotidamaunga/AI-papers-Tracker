import os
import tempfile

import arxiv
import pymupdf4llm
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

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

AI_LABS = [
    "OpenAI",
    "Google DeepMind",
    "Anthropic",
    "Meta AI",
    "Microsoft Research",
    "Google Research",
    "Mistral AI",
    "xAI",
    "Allen Institute for AI",
    "Nvidia",
]

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
    client = arxiv.Client(num_retries=5, delay_seconds=2)
    return list(client.results(search))


def extract_paper_text(result):
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = result.download_pdf(dirpath=tmp_dir)
        return pymupdf4llm.to_markdown(pdf_path)


def render_paper(result):
    entry_id = result.entry_id
    authors = ", ".join(a.name for a in result.authors)

    is_processed = entry_id in st.session_state.paper_cache

    with st.expander(f"{result.title}  —  {result.published.date()}", expanded=is_processed):
        st.markdown(f"**Authors:** {authors}")
        st.markdown(f"**Categories:** {', '.join(result.categories)}")
        st.markdown(result.summary.replace("\n", " "))
        st.markdown(f"[Abstract page]({result.entry_id}) · [PDF]({result.pdf_url})")

        if not is_processed:
            if st.button("Summarize this paper", key=f"summarize_{entry_id}"):
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

        question = st.chat_input("Ask a question about this paper...", key=f"chat_input_{entry_id}")
        if question:
            history.append({"role": "user", "content": question})
            with st.spinner("Thinking..."):
                try:
                    answer = chat_about_paper(cached["text"], history[:-1], question)
                except Exception as e:
                    answer = f"Error: {e}"
            history.append({"role": "assistant", "content": answer})
            st.rerun()


def main():
    st.set_page_config(page_title="AI Paper Tracker", layout="wide")
    st.title("AI Paper Tracker")
    st.caption("Search arXiv by topic or AI lab, get an LLM summary, then chat with the paper.")

    st.session_state.setdefault("paper_cache", {})
    st.session_state.setdefault("chat_histories", {})
    st.session_state.setdefault("results", [])

    if not DEEPSEEK_API_KEY:
        st.error("DEEPSEEK_API_KEY is not set. Add it to your .env file to enable summaries and chat.")

    with st.sidebar:
        st.header("Search")
        mode = st.radio("Search by", ["Topic", "AI Lab"])

        if mode == "Topic":
            query_input = st.text_input("Topic / keywords", placeholder="e.g. retrieval augmented generation")
            base_query = f"all:{query_input}" if query_input else ""
        else:
            lab_choice = st.selectbox("AI Lab", AI_LABS + ["Custom..."])
            lab_name = st.text_input("Custom lab name") if lab_choice == "Custom..." else lab_choice
            lab_topic = st.text_input("Also narrow by topic (optional)", placeholder="e.g. attention transformer")
            if lab_name and lab_topic:
                base_query = f'all:"{lab_name}" AND all:{lab_topic}'
            elif lab_name:
                base_query = f'all:"{lab_name}"'
            else:
                base_query = ""
            st.caption(
                "arXiv has no lab/affiliation field, so this matches the lab name in abstracts and comments — "
                "treat results as best-effort, and pair with a topic for much better precision."
            )

        selected_categories = st.multiselect(
            "Restrict to categories (optional)",
            options=list(CATEGORIES.keys()),
            format_func=lambda c: f"{c} — {CATEGORIES[c]}",
        )
        max_results = st.slider("Max results", min_value=5, max_value=50, value=10)
        sort_label = st.selectbox("Sort by", ["Relevance", "Submitted date"])
        sort_by = arxiv.SortCriterion.Relevance if sort_label == "Relevance" else arxiv.SortCriterion.SubmittedDate

        search_clicked = st.button("Search", type="primary")

    if search_clicked:
        if not base_query:
            st.warning("Enter a topic or choose/enter an AI lab first.")
        else:
            with st.spinner("Querying arXiv..."):
                try:
                    st.session_state.results = search_arxiv(base_query, max_results, sort_by, selected_categories)
                except Exception as e:
                    st.error(f"arXiv search failed: {e}")
                    st.session_state.results = []

    if not st.session_state.results:
        st.info("Run a search from the sidebar to see papers here.")
        return

    st.subheader(f"{len(st.session_state.results)} papers found")
    for result in st.session_state.results:
        render_paper(result)


if __name__ == "__main__":
    main()
