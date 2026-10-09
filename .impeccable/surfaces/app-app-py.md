---
version: 1
slug: "app-app-py"
primary_target: "app/app.py"
related_targets: ["app/home.py","app/ui.py"]
---

# Surface: FinSight dashboard (all pages)

Scope: every page of the Streamlit app (`app/app.py`, `app/home.py`, `app/views/*`, `app/ui.py`, `.streamlit/config.toml`). Visitor mode: Operate.
Audience: finance analysts scoring text, and technical evaluators judging the engineering (equal weight). First-minute takeaway: engineering depth.
Constraints: name FinSight fixed; Streamlit widgets only (theme + injected CSS); avoid template look, flashiness, low readability, default-Streamlit plainness.

## Direction contract

THESIS: FinSight reads as a central bank's quarterly report on its own models. Every result is a numbered exhibit with its source and its uncertainty band. It refuses the dark, glowing KPI-card dashboard the category ships by default.

OWN-WORLD: White report ground with a cool off-white panel tone, a deep-teal shell (sidebar rail), near-black teal-tinted ink, hairline rules instead of card shadows. Sentiment pairs colour with a glyph and a word: negative oxblood ▼, neutral slate ▶, positive green ▲. Charts use a teal ramp; uncertainty is drawn as shaded bands. One workhorse sans (Public Sans) with tabular figures everywhere numbers appear; Material Symbols icons only.

STORY: The visitor sees that the system measures itself honestly (in-domain, out-of-domain, fine-tuned), with the spread shown, then runs a prediction and drills into the machinery: explainability, errors, registry, API.

FIRST VIEWPORT: Teal rail on the left (logo, grouped navigation, model selector, engine status). On the right, a report masthead (title, one-line summary, the metadata line "Models 9 · Tests 116 · Data: PhraseBank + Twitter Financial News") and, directly below, Exhibit 1: macro-F1 by evaluation set drawn as bars with CV/uncertainty bands, the source line under it. The primary action is "Analyze a sentence" at the right of the masthead.

FORM: Central-bank inflation-report publication (fan-chart uncertainty as signature move). Position 1 on my ordered list (my pick, chosen by the user over the assigned index 6). Seed key 6f52133d (degraded roll, no challengers).

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
