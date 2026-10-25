# Constitution of India: NLP Chatbot (final-year version)

A retrieval chatbot over the Constitution of India that applies the techniques from the NLP theory and lab syllabus
(pre-processing, BoW/TF-IDF, Naive Bayes, n-gram language models, NER, Word2Vec, Seq2Seq translation, BERT).

## Run it
```bash
pip install -r requirements.txt          # core needs only streamlit + scikit-learn + pandas + matplotlib
streamlit run app.py                     # web app
python main.py                           # command-line version
python -m cbot.evaluate                  # regenerate the evaluation numbers/chart (results/)
python -m unittest discover -s tests     # 8 tests
```
Optional (nicer, not required): `python -m nltk.downloader stopwords wordnet averaged_perceptron_tagger_eng punkt`,
`python -m spacy download en_core_web_sm`. Without them, built-in fallbacks are used automatically.

## What changed from the original project
| Original | Now |
|---|---|
| Counts how many query words appear in the text | TF-IDF (word 1-2 grams) + character n-grams + LSA + IDF-weighted coverage, with BM25/BoW baselines |
| No pre-processing | regex clean, tokenise, stop-words (English + legal + Hinglish), stemming, lemmatisation |
| English only | Hinglish/Hindi query expansion (`adhikar`→right, `rashtrapati`→president, ...) and typo tolerance |
| Dataset of 425 hand-typed entries | **Verified dataset of 464 articles rebuilt from the official PDF** (see below) |
| Form + button | Chat interface, article browser, NLP explorer, analytics, evaluation, translation and BERT-summary tabs |
| No measurement | Four experiments with numbers for your report |

### Dataset audit (important)
`build_dataset.py` re-extracts every article number, title, Part and opening text from `data/Indian Constitution.pdf` and compares it with your
original JSON (`data/audit_report.csv`). Result for the 464 real articles: **198 matched, 75 reworded (kept), 125 had a wrong title/number,
66 were missing.** Example: the old file listed "Article 110: Calling Attention Motion" but Article 110 is *Definition of "Money Bills"*;
roughly Articles 98-260 in the old file were parliamentary-procedure topics under wrong numbers. Your Hinglish summaries are kept wherever the old title was correct.

## Syllabus mapping
| Syllabus | File |
|---|---|
| NLTK/spaCy/Gensim intro | `preprocess.py` (NLTK), `ner.py` (spaCy optional), `retrieval.py` (Gensim optional) |
| Stop-words, stemming, lemmatisation, regex | `preprocess.py` |
| NLP phases, POS tagging, NER | NLP Explorer tab, `ner.py` |
| BoW, CountVectorizer, TF-IDF | `retrieval.py` |
| Word2Vec / embeddings | `retrieval.py` (tick the box in the sidebar; needs gensim) and LSA |
| Naive Bayes | `classifier.py` |
| N-grams, Markov model, smoothing | `lm.py` |
| Machine translation (Seq2Seq encoder-decoder) | `advanced.py` -> `translate()` (MarianMT en<->hi) |
| Summarisation with pre-trained BERT | `advanced.py` -> `bert_summarize()` |
| RNN/LSTM text classification | not included (see future work) |

## Results (from `python -m cbot.evaluate`)
Ties are scored with the average rank, so a method cannot win by giving many articles the same score.

**A. Title -> article, held-out** (index has description + summary only; 464 queries)

| Method | Top-1 | Top-3 | MRR |
|---|---|---|---|
| Original keyword match | 0.356 | 0.582 | 0.507 |
| Bag-of-Words cosine | 0.506 | 0.688 | 0.616 |
| TF-IDF | 0.498 | 0.713 | 0.625 |
| BM25 | 0.534 | 0.728 | 0.650 |
| **Hybrid** | **0.595** | **0.769** | **0.695** |

**B. 45 hand-written English / Hinglish / misspelt queries** (`eval_queries.json`): keyword 0.333 top-1 / 0.600 top-3; **hybrid 0.778 / 0.911**.
Hybrid weights were tuned on experiment A only, so B is an unbiased check. Remaining misses are mostly near-duplicates (e.g. Art. 352 vs 353, both about emergency).

**C. Area classifier** (10 areas, 5-fold CV): Multinomial Naive Bayes 68.8% accuracy (macro-F1 0.684) vs majority baseline 22.6%; Logistic Regression 74.8%.
**D. Bigram LM perplexity** (held-out): unsmoothed = infinity; add-k with k=0.01 -> 104.8; k=1 -> 350.6, so a small k is best.

## Limitations (say these in your viva; it shows you understand the work)
* Area labels come from the Part of the Constitution, not human annotation. The model learns Part-style wording.
* Experiment B has only 45 queries; extend `eval_queries.json` for tighter estimates.
* Some "summary" lines are auto-generated (equal to the title) for the 191 articles that had no valid Hinglish summary. Add real summaries to improve the display.
* Translation/BERT tabs download pre-trained models on first use and were written against the documented APIs but not run in the development sandbox (no internet there).
* Descriptions are the opening text of each article as extracted from the PDF; a few contain extraction artefacts (e.g. missing spaces).

## Future work
LSTM/GRU classifier for query intent (Unit 4), fine-tuned BERT/sentence-transformer retriever, voice input, answer generation with citations.

## Deploy (Streamlit Community Cloud)
1. Push this folder to a public GitHub repo (app.py must be in the repo root).
2. Go to https://share.streamlit.io -> New app -> pick the repo, branch `main`, main file `app.py`.
3. Advanced settings -> Python 3.11 or 3.12 -> Deploy.
`requirements.txt` holds only the core packages; optional libraries are in `requirements-extras.txt`.
The Translate / BERT tab needs `requirements-extras.txt` and is not enabled on the free cloud build.
