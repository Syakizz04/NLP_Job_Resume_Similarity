# EZhire — Matching Résumés to Job Descriptions

EZhire reads a résumé and a job description (JD) and gives a **match score from 0 to
100**, plus a simple verdict: **Strong Match**, **Potential Fit**, or **Poor Match**.
If you give it several résumés for one job, it ranks them so a recruiter can see the
best candidates first.

It does this by combining two ideas:
1. **AI that understands meaning** — three language models that judge how similar a
   résumé and a JD are *in meaning*, not just in shared words.
2. **Synonym-aware keyword matching** — a classic technique (WordNet) that spots
   related words (*"manager" ≈ "supervisor"*) even when the wording differs.

Blending the two gives better, more reliable rankings than either one alone. A separate
exact-word tool (TF-IDF) powers the app's "matched / missing keywords" view but does not
feed the score. Everything runs inside Jupyter notebooks and ends in a simple web app
(built with Gradio).

---

## 1. The Idea Behind It

### What problem are we solving?
Given a résumé and a job description, we want a number that says *how good a fit this
person is*. We then use that number to **sort candidates** from best to worst.

We treat this as a **"how similar are these two texts?"** problem. We are not sorting
résumés into fixed boxes — we are predicting a score on a sliding scale. What matters
most is getting the **order** right (the strongest candidate should land at the top),
because a recruiter cares more about *who is better than whom* than about the exact
number itself.

### The data we learned from
- **Dataset:** a public collection of résumé + job-description pairs from Hugging
  Face (`0xnbk/resume-ats-score-v1-en`). Each pair comes with a real "ATS score"
  (the kind of score real hiring software gives) and a label of `No Fit`,
  `Potential Fit`, or `Good Fit`.
- We test the system on **1,275 résumé–JD pairs** it never saw during training
  (641 No Fit, 319 Potential Fit, 315 Good Fit).
- The original scores run roughly from 18 to 91. We **rescale them to a 0–1 range**
  so the model has a clean target to learn and to score against.

### The main choices we made (and why)
| Choice | Why we did it |
|--------|---------------|
| **Split long documents into smaller pieces ("chunks")** | Résumés and JDs are often too long for a language model to read in one go. Instead of cutting off the end (and losing skills or experience), we break the text into overlapping pieces so nothing is lost. |
| **Add a short label to each piece** | Each piece gets a tag saying whether it came from the résumé or the JD, plus the document's first sentence — so a single piece still makes sense on its own. |
| **Train the model to predict a similarity score** | Our target is a graded score (0–1), so we teach the model to output a matching score directly, rather than a yes/no decision. |
| **Train three different models, then keep the best** | Three popular language models (MPNet, RoBERTa, BERT) have different strengths. We train all three and let the test results pick the winner. |
| **Add synonym matching to the AI** | The AI understands meaning but may phrase things differently from the JD. A synonym helper (WordNet) catches related words. It makes different mistakes from the AI, so blending the two lifts the ranking. |
| **Compare candidates by rank before mixing** | Because we care about ordering, we convert every method's output into rankings before blending them. This stops one method from drowning out another just because its numbers happen to be bigger. |
| **Let a small "referee" model learn the best mix** | Instead of guessing how much to trust each method, we train a simple model to combine them, and test it fairly on data it didn't help tune. |

---

## 2. How It's Built

### The journey of the data, start to finish

```
                ┌─────────────────────────────────────────────────────────┐
                │   Dataset of résumé + job-description pairs (HuggingFace) │
                └─────────────────────────────────────────────────────────┘
                                          │
                       Split into résumé vs. JD, clean the text,
                          rescale the target score to 0–1
                                          │
        ┌─────────────────┬──────────────┴───────────────┐
        ▼                 ▼                               ▼
   Train Model A      Train Model B                  Train Model C
     (MPNet)           (RoBERTa)                       (BERT)
        └─────────────────┴───────────────┬──────────────┘
                                          ▼
                          Saved trained models
                                          │
                                          ▼
                    Comparison notebook: test all three,
              add keyword methods, and find the best combination
                                          │
                                          ▼
              Saved results files (scores, settings, charts data)
                                          │
                                          ▼
                       Web app (Gradio dashboard)
```

### What each notebook does
| Notebook | Job |
|----------|-----|
| `EZhire_Model_A.ipynb` | Trains language model **MPNet**. |
| `EZhire_Model_B.ipynb` | Trains language model **RoBERTa**. |
| `EZhire_Model_C.ipynb` | Trains language model **BERT**. |
| `EZhire_Metrics_Comparison.ipynb` | Tests all three, adds the keyword methods, picks the best combination, and saves everything. |
| `EZhire_Gradio_Setup.ipynb` | Loads the saved results and runs the web app. |

### Cleaning up the text
The notebooks share the same text-preparation steps:
- **Separating** the résumé from the JD (they're stored together in one field).
- **Tidying** spacing and line breaks.
- **Fixing stuck-together words.** PDFs often glue words together
  (`SummaryI`, `VistaWindows98`). We split these back into real words
  (`summary i`, `vista windows 98`) so keyword matching works — while leaving short
  abbreviations like `PMP` or `JDE` untouched.

### How the AI scores a pair
1. **Break into pieces.** Both the résumé and the JD are split into overlapping
   chunks small enough for the model to read, with a tag added to each.
2. **Turn text into numbers.** The model converts each piece into a list of numbers
   ("embedding") that captures its meaning.
3. **Compare every piece to every piece.** We measure how similar each résumé piece
   is to each JD piece.
4. **Combine into one score.** The final score mostly rewards the **strongest
   matches** between pieces, with a smaller bonus for matching *broadly* across the
   whole document. This becomes the 0–100 match score.

### How we trained each model
| Setting | Value | Plain meaning |
|---------|-------|---------------|
| Training rounds | 4 | How many times the model studied the data. |
| Batch size | 4 | How many examples it looks at before adjusting. |
| Learning rate | 0.00002 | How big each adjustment step is (small = careful). |
| Max text length | 384 words/pieces | How much text fits in one chunk. |
| Random seed | 42 | Fixes randomness so results can be reproduced. |

Each finished model is saved to `saved_models/` (and zipped into `training_zipped/`
for easy transfer).

### The keyword helpers
- **WordNet (synonym matching).** Uses a dictionary of word meanings to spot
  **synonyms** — so "manager" and "supervisor" can count as related even though
  they're different words. This catches matches the AI phrases differently, and it is
  **part of the final scorer**.
- **TF-IDF (exact-word matching).** Scores how many important words the résumé and JD
  share, giving rare/specialist words more weight, and survives things like `c#` and
  `.net`. We tested it inside the scorer too, but it slightly *hurt* the ranking when
  combined with the AI — so it is **not part of the score**. It is kept only to power
  the app's "matched / missing keywords" view for a single résumé.

On its own the synonym helper is weak, but it notices things the AI misses — which is
why blending it in nudges the ranking up.

### Picking the best combination
1. Convert every method's output into **rankings** so they're on equal footing.
2. Try blending the AI with the keyword helpers and see which mix ranks candidates
   best — measured by **NDCG** and tested carefully on held-back data (out-of-fold)
   so we don't fool ourselves.
3. The winning combination becomes the official scorer. Final verdicts are:
   **70 and up = Strong Match**, **45–69 = Potential Fit**, **below 45 = Poor Match**.

### What gets saved (`artifacts/` folder)
| File | What's inside |
|------|---------------|
| `ensemble_config.json` | Which model and combination won, and their scores. |
| `metrics_df.csv` | The full results table for every method. |
| `validation_scores.csv` | Scores and verdicts for all 1,275 test pairs. |
| `tfidf_union.joblib` | The saved keyword-matching tool (reused by the app). |

### The web app
The Gradio dashboard has three tabs:
- **Score Résumés** — paste a job description, upload résumé **PDFs (or a ZIP of
  PDFs)**, and get a ranked table and bar chart of candidates. Scoring is **locked to
  the winning ensemble** (no model picker). For a single résumé it also shows which
  keywords **match**, which are **missing**, and which appear **only in the résumé**.
- **Score Heatmap** — a colour grid and distribution chart comparing how the three AI
  models (BERT, RoBERTa, MPNet) and the **winning ensemble** scored across the test set.
- **Model Comparison** — tables and charts comparing the three models on the stable
  ranking metrics (NDCG, Precision@50, Spearman), with MAE shown separately.

---

## 3. The Results

All numbers below come from testing on the **1,275 pairs the system never trained on**,
compared against the real ATS scores. Of those pairs, **479 (37.6%)** are genuine good
fits — so a method that ranked at random would score about 0.376 on Precision.

**How to read the columns** (we judge on three metrics, all measured over the full
test set so they're statistically stable):
- **NDCG** — how good the *ranking* is, weighting the top of the list most. Higher is
  better. *This is our headline number — the metric we optimise the final combination for.*
- **Precision@50** — of the 50 pairs the method ranks highest, the share that are
  genuine good fits. Higher is better (random ≈ 0.38).
- **Spearman** — how well the predicted *ordering* agrees with the real scores
  (closer to 1.0 is better).

| Method | NDCG | P@50 | Spearman |
|--------|:----:|:----:|:--------:|
| **Final ensemble (AI + synonyms)** ✅ | 0.9754 | 0.90 | **0.7729** |
| RoBERTa | **0.9767** | 0.92 | 0.7674 |
| MPNet | 0.9747 | **0.94** | 0.7673 |
| BERT *(chosen AI arm)* | 0.9725 | **0.94** | 0.7716 |
| WordNet synonyms alone | 0.8853 | 0.60 | 0.1269 |
| TF-IDF keywords alone | 0.8796 | 0.58 | 0.1612 |

**Choosing the AI arm.** The three language models are a near dead-heat on the stable
metrics, so we pick the one with the best *combined* mean of NDCG, P@50 and Spearman:

| AI model | NDCG | P@50 | Spearman | Combined |
|----------|:----:|:----:|:--------:|:--------:|
| **BERT** ✅ | 0.9725 | 0.94 | 0.7716 | **0.895** |
| MPNet | 0.9747 | 0.94 | 0.7673 | 0.894 |
| RoBERTa | 0.9767 | 0.92 | 0.7674 | 0.888 |

BERT wins by a whisker (0.895 vs 0.894 vs 0.888) thanks to the best Precision@50 and
Spearman, so it becomes the AI part of the system.

**Why this combination wins (ranking quality of each blend, measured by NDCG):**
| Combination | NDCG |
|-------------|:----:|
| **AI + synonyms (BERT + WordNet)** ✅ *(chosen)* | **0.9754** |
| AI alone (BERT) | 0.9753 |
| AI + exact keywords | 0.9740 |
| AI + exact keywords + synonyms | 0.9735 |

### What we learned
- **All three AI models work well** (rank agreement ≈ 0.77 with real scores) and are
  **statistically tied** on the stable metrics, confirming the chunking approach is
  effective. **BERT** edges ahead on the combined ranking mean and is chosen as the AI
  arm; it also fits the raw scores most tightly of the three.
- **Synonym matching adds a small lift** over the AI alone (NDCG 0.9754 vs 0.9753).
  **Exact-keyword TF-IDF actually *hurt* the ranking** when added to the AI, so it is
  left out of the scorer (and kept only for the keyword-overlap display).
- **The final system blends the AI with the synonym helper (BERT + WordNet)** — the
  best-ranking option we tried, though the margin over the AI alone is tiny.
- **Rankings are strong overall** (NDCG ≈ 0.975), which is exactly what you want for
  shortlisting candidates.

---

## 4. How to Run It

Run the notebooks in this order (a GPU makes training much faster):

1. **Train the models** — run `EZhire_Model_A`, `EZhire_Model_B`, and
   `EZhire_Model_C`. Each saves a trained model.
2. **Compare and combine** — run `EZhire_Metrics_Comparison` to test the models, add
   the keyword methods, pick the best combination, and save the results.
3. **Launch the app** — run `EZhire_Gradio_Setup` to open the dashboard.

> Already have trained models? Skip step 1: unzip the files in `training_zipped/`
> into `saved_models/` and start at step 2.

### What you need installed
```
sentence-transformers, transformers, datasets   (the AI models)
scikit-learn, nltk, wordninja                    (keyword methods + text cleanup)
PyMuPDF                                           (reading PDFs)
gradio, plotly                                    (the web app + charts)
torch, numpy, pandas, scipy, joblib              (general tools)
```

---

## 5. What's in This Folder

```
EZhire_final_pipeline/
├── EZhire_Model_A.ipynb              # Train MPNet
├── EZhire_Model_B.ipynb              # Train RoBERTa
├── EZhire_Model_C.ipynb              # Train BERT
├── EZhire_Metrics_Comparison.ipynb   # Test models + build the final combination
├── EZhire_Gradio_Setup.ipynb         # Run the web app
├── saved_models/                     # The trained AI models
└── artifacts/
    ├── ensemble_config.json          # Which model + combination won
    ├── metrics_df.csv                # Full results table
    ├── validation_scores.csv         # Scores + verdicts for all test pairs
    └── tfidf_union.joblib            # Saved keyword-matching tool
```
