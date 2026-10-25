"""Constitution NLP Chatbot: Streamlit app.   Run:  streamlit run app.py"""
import json
from collections import Counter

import pandas as pd
import streamlit as st

from cbot.data import load_records, ROOT
from cbot.chatbot import ConstitutionBot
from cbot.preprocess import explain_pipeline, pos_tags, preprocess, HAS_NLTK
from cbot.retrieval import HAS_GENSIM, METHODS
from cbot.ner import extract_entities, spacy_entities
from cbot import advanced

st.set_page_config(page_title="Constitution NLP Chatbot", page_icon="📜", layout="wide")


@st.cache_resource(show_spinner="Training NLP models...")
def get_bot(use_w2v):
    return ConstitutionBot(load_records(), use_word2vec=use_w2v)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("⚙️ Settings")
method = st.sidebar.selectbox("Retrieval model", METHODS, index=METHODS.index("hybrid"),
                              format_func=lambda m: {"keyword": "Original keyword match", "bow": "Bag-of-Words", "tfidf": "TF-IDF",
                                                     "bm25": "BM25", "tfidf_expand": "TF-IDF + query expansion", "hybrid": "Hybrid (best)"}[m])
top_k = st.sidebar.slider("Results to show", 1, 5, 3)
use_w2v = st.sidebar.checkbox("Add Word2Vec (Gensim)", value=False, disabled=not HAS_GENSIM,
                              help="Needs `pip install gensim`" if not HAS_GENSIM else "Trains Word2Vec on the articles")
bot = get_bot(use_w2v)
st.sidebar.markdown("---")
st.sidebar.caption(f"📚 {len(bot.records)} articles (verified from the official PDF)")
st.sidebar.caption(f"NLTK: {'✅' if HAS_NLTK else '➖ built-in fallback'} · Gensim: {'✅' if HAS_GENSIM else '➖'}")

st.title("📜 Constitution of India: NLP Chatbot")
st.caption("Ask in English or Hinglish, e.g. “who appoints the governor”, “article 21A”, “shiksha ka adhikar”.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_log" not in st.session_state:
    st.session_state.query_log = []


def render_answer(ans):
    st.markdown(ans["message"])
    for rec, score in ans["results"]:
        with st.container(border=True):
            st.markdown(f"**📘 Article {rec['id']}: {rec['title']}**")
            st.caption(f"Part {rec['part']}: {rec['part_name']}  ·  Area: {rec['group']}")
            st.write(rec["description"])
            if rec["summary_text"]:
                st.info("Summary: " + rec["summary_text"])
            if score < 1.0:
                st.progress(min(max(score, 0.0), 1.0), text=f"Relevance {score:.2f}")
    if ans.get("area"):
        st.caption("🧭 Predicted area (Naive Bayes): " + " · ".join(f"{n} ({p:.0%})" for n, p in ans["area"]))
    if ans.get("entities"):
        st.caption("🏷️ Entities: " + ", ".join(f"{t} `{l}`" for t, l in ans["entities"]))


tab_chat, tab_art, tab_nlp, tab_ana, tab_eval, tab_adv, tab_about = st.tabs(
    ["💬 Chatbot", "🔢 Browse articles", "🧪 NLP Explorer", "📊 Analytics", "📈 Evaluation", "🌐 Translate & Summarise", "ℹ️ About"])

# ------------------------------------------------------------------ chat
with tab_chat:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.write(m["content"])
            else:
                render_answer(m["content"])
    prompt = st.chat_input("Ask about the Constitution…")
    if prompt:
        ans = bot.answer(prompt, k=top_k, method=method)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": ans})
        st.session_state.query_log.append({"query": prompt, "intent": ans["intent"],
                                           "top_article": ans["results"][0][0]["id"] if ans["results"] else None})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            render_answer(ans)
    if st.session_state.messages and st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------ browse
with tab_art:
    c1, c2 = st.columns([1, 2])
    with c1:
        num = st.text_input("Article number (e.g. 14, 21A, 370)")
        groups = sorted({r["group"] for r in bot.records})
        group = st.selectbox("…or browse an area", ["(choose)"] + groups)
    with c2:
        if num:
            rec = bot.get_article(num)
            if rec:
                render_answer({"message": f"Article {rec['id']}", "results": [(rec, 1.0)]})
            else:
                st.error("Article not found")
        elif group != "(choose)":
            for rec in [r for r in bot.records if r["group"] == group]:
                with st.expander(f"Article {rec['id']}: {rec['title']}"):
                    st.write(rec["description"])

# ------------------------------------------------------------------ NLP explorer
with tab_nlp:
    st.subheader("See every NLP step")
    text = st.text_input("Type a sentence", "The President may dissolve the Lok Sabha on the advice of the Prime Minister", key="nlp_text")
    if text:
        steps = explain_pipeline(text)
        st.table(pd.DataFrame({"Step": list(steps), "Output": [str(v) for v in steps.values()]}))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top TF-IDF terms**")
            terms = bot.retriever.top_terms(text)
            st.dataframe(pd.DataFrame(terms, columns=["term", "tf-idf"]), hide_index=True)
            st.markdown("**POS tags (NLTK)**")
            tags = pos_tags(text)
            st.write(tags if tags else "Install NLTK + `averaged_perceptron_tagger_eng` to enable POS tagging")
        with c2:
            st.markdown("**Named entities (rule-based)**")
            st.write(extract_entities(text) or "none")
            sp = spacy_entities(text)
            st.markdown("**Named entities (spaCy)**")
            st.write(sp if sp is not None else "spaCy / en_core_web_sm not installed")
            st.markdown("**Naive Bayes: which part of the Constitution?**")
            area = bot.classifier.predict(text, k=4)
            st.bar_chart(pd.DataFrame({"probability": [p for _, p in area]}, index=[n for n, _ in area]))
        st.markdown("**Next-word suggestions (bigram model with add-k smoothing)**")
        st.write(", ".join(w for w, _ in bot.lm.suggest(text)) or "(none)")

# ------------------------------------------------------------------ analytics
with tab_ana:
    st.subheader("Dataset")
    cnt = Counter(r["group"] for r in bot.records)
    st.bar_chart(pd.Series(cnt, name="articles"))
    words = Counter(w for r in bot.records for w in preprocess(r["title"] + " " + r["description"]))
    st.markdown("**Most frequent (stemmed) terms**")
    st.bar_chart(pd.Series(dict(words.most_common(20)), name="frequency"))
    st.subheader("This session")
    if st.session_state.query_log:
        log = pd.DataFrame(st.session_state.query_log)
        st.dataframe(log, hide_index=True)
        st.caption(f"{len(log)} queries · {(log['intent'] == 'low_confidence').sum()} low-confidence")
    else:
        st.write("Ask something in the chat tab first.")

# ------------------------------------------------------------------ evaluation
with tab_eval:
    res = ROOT / "results" / "evaluation.json"
    if not res.exists():
        st.warning("Run `python -m cbot.evaluate` once to generate results.")
    else:
        E = json.loads(res.read_text())
        names = {"keyword": "Original keyword match", "bow": "Bag-of-Words", "tfidf": "TF-IDF", "bm25": "BM25",
                 "tfidf_expand": "TF-IDF + expansion", "hybrid": "Hybrid"}
        for key, title in (("A_title_heldout", "A: title → article (held-out, ~460 queries)"), ("B_manual", "B: 45 hand-written / Hinglish / misspelt queries")):
            st.subheader(title)
            st.dataframe(pd.DataFrame(E[key]).T.rename(index=names).style.format("{:.3f}"))
        st.subheader("C: Area classifier (5-fold CV)")
        st.dataframe(pd.DataFrame(E["C_classifier"]).T.style.format("{:.3f}"))
        st.subheader("D: Bigram LM perplexity vs smoothing k")
        st.dataframe(pd.DataFrame({"k": list(E["D_perplexity"]), "perplexity": [str(v) for v in E["D_perplexity"].values()]}), hide_index=True)
        img = ROOT / "results" / "retrieval_comparison.png"
        if img.exists():
            st.image(str(img))

# ------------------------------------------------------------------ advanced
with tab_adv:
    st.subheader("Machine translation (Seq2Seq encoder-decoder) and BERT summarisation")
    st.caption("Needs `pip install transformers torch sentencepiece` and internet on first use (models are downloaded).")
    left, right = st.columns(2)
    with left:
        src = st.text_area("Text to translate", "Every citizen has the right to equality before the law.")
        direction = st.radio("Direction", ["en-hi", "hi-en"], format_func=lambda d: "English → Hindi" if d == "en-hi" else "Hindi → English", horizontal=True)
        if st.button("Translate"):
            try:
                with st.spinner("Translating…"):
                    st.success(advanced.translate(src, direction))
            except advanced.AdvancedUnavailable as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Translation failed: {e}")
    with right:
        area = st.selectbox("Summarise all articles of an area", sorted({r["group"] for r in bot.records}))
        n_sent = st.slider("Sentences", 2, 6, 3)
        if st.button("Summarise with BERT"):
            joined = " ".join(r["title"] + ". " + r["description"] for r in bot.records if r["group"] == area)
            try:
                with st.spinner("Embedding sentences with BERT…"):
                    st.success(advanced.bert_summarize(joined, n_sent))
            except advanced.AdvancedUnavailable as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Summarisation failed: {e}")

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown("""
### How it maps to the syllabus
| Unit | Topic | In this project |
|---|---|---|
| 1 | NLP phases, tokenisation, stemming, POS, NER | `preprocess.py`, `ner.py`, NLP Explorer tab |
| 2 | BoW, CountVectorizer, TF-IDF, Naive Bayes | `retrieval.py`, `classifier.py` |
| 3 | Word2Vec / Gensim, embeddings | optional Word2Vec + LSA in `retrieval.py` |
| 3 | N-grams, Markov models, smoothing | `lm.py` (bigram LM, add-k) |
| 4 | RNN / LSTM | see README "future work" |
| 5 | Machine translation, BERT | `advanced.py` (MarianMT, BERT extractive summary) |
""")
    audit = ROOT / "data" / "audit_report.csv"
    if audit.exists():
        df = pd.read_csv(audit)
        st.markdown("### Dataset audit (old JSON vs official PDF)")
        st.write(df["status"].value_counts().rename("articles").to_frame())
        st.dataframe(df[df["status"].isin(["WRONG", "MISSING"])].head(100), hide_index=True)
