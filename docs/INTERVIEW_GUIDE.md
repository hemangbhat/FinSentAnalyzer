# FinSight: technical interview guide

How to present this project in a technical interview: the story, a demo script,
the limitations to raise yourself, likely questions, and the numbers to know.

## Readiness checklist

Done:

- [x] ML pipeline, API, 10-page dashboard, Docker, and CI green on `main`
- [x] 119 automated tests; ruff and mypy in CI
- [x] Every headline number traces to a file in `results/`, with bootstrap
      confidence intervals (`scripts/bootstrap_ci.py`)
- [x] Limitations documented in the README, `docs/MODEL_CARD.md` and `docs/TRACKER.md`

Do before sending the repo to anyone:

- [ ] Deploy (README → Deployment → Option 1) and put the URL on README line 14
- [ ] Record a 15-second GIF of Home → Single Analysis as `docs/demo.gif`
- [ ] Open the live app an hour before the interview; free apps sleep and take
      about a minute to wake
- [ ] Rehearse the 30-second pitch and the demo script below out loud

---

The one idea to land: **you didn't stop at a good test score.** You stress-tested
the model on real data, found that it failed, worked out why, and fixed it. Most
candidates show a 92% accuracy number. You can show 0.90 → 0.46 → 0.84 and
explain every arrow. That reads as engineering judgement, not tutorial-following.

---

## 1. The 30-second pitch

> "FinSight classifies financial text as positive, neutral or negative. I started
> with classical baselines, TF-IDF plus an SVM, which scored 0.90 macro-F1 on
> Financial PhraseBank. Then I tested it on 2,400 real financial news headlines
> it had never seen, and it dropped to 0.46, barely better than always predicting
> 'neutral'. The model had learned annual-report phrasing, not sentiment. I
> fine-tuned FinBERT on in-domain news on a GPU and got back to 0.84. It's served
> through a FastAPI service with validation, auth and rate limiting, has a
> Streamlit dashboard for explainability and error analysis, and it's covered by
> 119 tests in CI."

Practise it out loud until it takes under 35 seconds.

---

## 2. The 2-minute walkthrough (use the Home page as your slide)

1. **Problem.** Financial sentiment feeds trading signals, risk monitoring and
   news triage. Errors are asymmetric: calling bad news neutral is expensive.
2. **Data.** Financial PhraseBank, the 2,264 sentences where all annotators
   agreed. Stratified 80/10/10 split. Heavily imbalanced: about 61% neutral, so
   always predicting neutral already scores about 62% accuracy. **That's why I
   report macro-F1, not accuracy.**
3. **Baselines.** 7 classical models on TF-IDF (unigrams and bigrams, 10k
   features). 5-fold stratified CV, because a 226-row test set swings by ±3–4%.
   SVM and Gradient Boosting tie under CV (0.838 vs 0.835). I picked SVM because
   it's faster and more stable.
4. **Stress test.** Real Twitter financial-news headlines (MIT licence,
   n=2,388). Macro-F1 falls 0.90 → 0.46.
5. **Diagnosis.** SHAP shows several top SVM features are stopwords (`is`, `to`,
   `the`, `from`). The model keys on the *style* of Finnish annual reports. Short
   tweet-style headlines have none of those cues.
6. **Fix.** FinBERT (110M parameters, BERT pre-trained on financial text)
   fine-tuned on the news training split for 4 epochs on a T4, with class
   weights for imbalance, mixed precision, and best-checkpoint restore. Result:
   0.87 accuracy, 0.84 macro-F1.
7. **Production.** FastAPI with Pydantic validation (5,000-char limit, batches
   of 256), optional API key, Redis-backed rate limiting, nginx load balancing
   across replicas, model registry with SHA-256 hashes, monthly retraining
   workflow, Docker.

---

## 3. Live demo script (about 3 minutes, rehearse it)

| Step | Where | What to say |
|---|---|---|
| 1 | **Home** | "Every number here is read from the results files, so the UI can't drift from the evaluation." Point at Exhibit 1: each score is a dot with its 95% bootstrap interval, and the 0.90 → 0.46 → 0.84 drop and recovery is the story. |
| 2 | **Single Analysis** → click *Plant closure, 300 jobs cut* | SVM says **neutral (81%)**. "That's wrong, and it's a nice example of the TF-IDF model's limit." |
| 3 | Switch the sidebar model to **FinBERT (fine-tuned)** and click the same example | Compare the verdicts and the probability bands. "The model choice persists across pages because the sidebar is rendered once by an `st.navigation` router." If fine-tuned FinBERT isn't in the menu on the deployed app, its weights aren't on the host yet (README → Deployment); use **FinBERT (zero-shot)** instead. |
| 4 | **Explainability**, then **Word Insights** | Show the sentence tinted by contribution, then SHAP: `is`, `to`, `the` are among the strongest features. "This is how I diagnosed the generalization failure." Word Insights flags the same stopwords. |
| 5 | **Error Analysis** | The error flow, then the confidence chart. "Mean confidence is lower on errors, so a threshold could route uncertain sentences to a human. Mean confidence alone doesn't prove calibration; a reliability diagram would." |
| 5b | **Model Registry** | Exhibit 1: "On a 226-sentence test set the top baselines' intervals overlap. They're statistically tied, which is why I pick models on cross-validation." |
| 6 | **Generalization** | The real-news exhibit with intervals, then the FinBERT confusion matrix. The main remaining confusion is neutral vs negative (92 neutral headlines read as negative). Scroll to the synthetic pipeline demo and say plainly that it isn't evidence about markets. |
| 7 | Terminal: `curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"text":"Profit fell 20%"}'` | "Same models behind an API. The UI can run in split mode against it." |

**Before the interview:** deploy it and record the GIF (see the checklist at the
top). A live URL in your résumé is worth more than anything below.

---

## 4. Say the weaknesses before they find them

Interviewers trust candidates who volunteer limitations. Raise these yourself:

1. **The 0.84 isn't out-of-distribution for FinBERT.** It was trained on the
   news *train* split and scored on the *validation* split, so that's in-domain
   held-out. The honest claim is "domain-matched training data closes the gap,"
   not "FinBERT generalizes."
2. **There's a small selection bias in the 0.84.** Early stopping picked the
   best epoch using the first 500 rows of that same validation split, then
   reported on all 2,388. The fix is a separate dev split carved from `train`.
   It's in the tracker as a next step.
3. **The 226-row PhraseBank test set is small.** That's why the CV mean ± std is
   the number to trust.
4. **The TF-IDF pipeline doesn't remove stopwords**, and SHAP shows the model
   relying on them.
5. **Stock prediction is chance-level** and the bundled headlines are synthetic.
   It's labelled "Experimental" in the UI. Call it an engineering demo of the
   yfinance → FinBERT → LSTM plumbing. Never call it a forecasting result.
6. **The "Deep Analysis" reasoning is rule-based templates, not an LLM.** Don't
   let anyone think otherwise.

A good line to use: *"The most valuable result in this project is a negative
one: the 0.46."*

---

## 5. Likely questions, with strong answers

**Why macro-F1 instead of accuracy?**
About 61% of the data is neutral, so accuracy rewards ignoring the minority
classes. Macro-F1 weights negative, neutral and positive equally, and the
minority classes are the ones that matter for trading.

**Why not just use FinBERT from the start?**
Baselines set the bar and are explainable. They also run in microseconds on a
CPU. I needed to know what a transformer actually buys. On PhraseBank the SVM is
competitive. The transformer only pays off under domain shift.

**How did you handle class imbalance?**
`class_weight='balanced'` for the classical models, stratified splits and folds,
and inverse-frequency class weights in the cross-entropy loss for FinBERT.

**How do you know you're not overfitting?**
5-fold stratified CV with mean ± std. A held-out test set touched once. The real
proof is the OOD test, which is what exposed the problem.

**Walk me through fine-tuning.**
Tokenize, AdamW at lr 2e-5, batch size 32, 4 epochs, a 3-class head, weighted
cross-entropy, fp16 AMP on CUDA, track validation macro-F1 each epoch, and
restore the best checkpoint. DeBERTa crashed under fp16 because its logits came
back in half precision against an fp32 loss. I cast logits to float32 and
disabled the GradScaler for that model family. That's a real debugging story;
tell it.

**How is it deployed? How would it scale?**
Streamlit UI plus a FastAPI inference service. Models load once and are cached.
Requests are validated with Pydantic. Rate limits live in Redis so they hold
across replicas, and nginx load-balances across `--scale api=N`. The UI deploys
to Streamlit Community Cloud with CPU-only PyTorch. The 438 MB fine-tuned
FinBERT weights live in Git LFS. If the host only gets the LFS pointer, the app
detects it and, when `FINSIGHT_FINBERT_REPO` is set, loads the weights from the
Hugging Face Hub instead; otherwise it hides that model and keeps working. Small result
files are kept in plain git so they never depend on LFS. To scale further: batch
inference, ONNX or quantized FinBERT for CPU, and an async queue for batch jobs.

**How would you monitor it in production?**
Log the prediction, confidence and latency (`src/monitoring.py` already does
this). Watch the predicted-class distribution and mean confidence for drift.
Sample low-confidence predictions for human labelling, and feed those into the
monthly retraining workflow, which is gated on tests and registry review.

**What's the hardest bug you hit?**
Pick one you can tell end to end. Good candidates:
- The DeBERTa fp16/fp32 dtype mismatch.
- The deployed app failing to load FinBERT because the host checked out a Git
  LFS pointer instead of the weights, fixed by detecting the pointer and falling
  back to the HF Hub.
- The scheduled retraining that failed silently for three months on a missing
  dependency, fixed by pinning CI requirements.

**How confident are you in those scores?**
I bootstrap them (`scripts/bootstrap_ci.py`, 2,000 resamples). On the 226-row PhraseBank test set, SVM's 0.902 has a 95% interval of [0.850, 0.945], which overlaps Gradient Boosting, Logistic Regression and the ensemble. They are statistically tied, which is why I choose models on 5-fold CV and not one test split. On the 2,388 real headlines the intervals are tight (SVM 0.44–0.49, FinBERT 0.82–0.86), so the gap between them is real. The app draws every score with its interval.

**Found any bugs in your own code?**
Yes. The entity extractor compiled every regex with IGNORECASE, so the ticker rule `[A-Z]{2,5}` matched any 2–5 letters and chopped "The company" into "The co" / "mpany". I wrote failing tests, made that one pattern case-sensitive with word boundaries, and filtered currency codes (EUR, USD) that look like tickers. Good story: a one-flag bug that only showed up when the UI displayed the output.

**What would you do next?**
1. A clean re-run with a separate dev split.
2. Stopword and n-gram tuning on the baseline.
3. Probability calibration with temperature scaling.
4. Entity-level sentiment ("Apple up, Samsung down" in one headline).
5. Distil FinBERT for cheaper CPU serving.

**Why SHAP? What are its limits?**
SHAP gives additive attributions grounded in game theory, both local and global.
Limits: it's expensive for non-tree models (linear models use coefficients
here), correlated features split credit between them, and it explains the model,
not the world.

---

## 6. Numbers cheat sheet (memorise these)

| Item | Value |
|---|---|
| PhraseBank (AllAgree) | 2,264 sentences · 80/10/10 · ~61% neutral |
| SVM, test (n=226) | 92.5% acc · **0.90** macro-F1 |
| SVM, 5-fold CV | **0.838 ± 0.016** macro-F1 |
| Gradient Boosting, CV | 0.835 ± 0.008 (tied with SVM) |
| Voting ensemble, test | 92.0% acc · 0.891 macro-F1 |
| SVM on real news (n=2,388) | 0.67 acc (majority 0.66) · **0.46** macro-F1 |
| DistilBERT, 1 epoch, CPU | 0.81 acc · 0.73 macro-F1 |
| FinBERT fine-tuned, 4 epochs, T4 | **0.87** acc · **0.84** macro-F1 (9,543 train / 2,388 val) |
| SVM test macro-F1, 95% bootstrap CI | 0.902 [0.850, 0.945] (n=226) |
| SVM on real news, 95% CI | 0.460 [0.436, 0.485] |
| FinBERT on held-out news, 95% CI | 0.841 [0.823, 0.858] |
| Tests | 119 · ruff + mypy in CI |

---

## 7. Delivery tips

- **Lead with the story, not the stack.** Name tools only when they explain a
  decision.
- **Own every claim.** If a line is in the README, be ready to whiteboard it,
  for example how TF-IDF weights a term or what macro-F1 averages. If you can't
  explain something yet, learn it before the interview.
- **Use the word "because."** "I chose SVM *because* CV showed it tied with
  Gradient Boosting at a fraction of the cost."
- **Have the repo open** at `src/predict.py` (the unified inference interface)
  and `api/main.py`. Code walkthroughs are a common follow-up.
- **End with what you learned:** "A high test score on one dataset told me
  almost nothing about real-world performance. Out-of-distribution evaluation
  is now the first thing I do."
