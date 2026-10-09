---
name: FinSight
description: A central bank's quarterly report on its own sentiment models.
colors:
  report-white: "#FFFFFF"
  panel-mist: "#F2F5F4"
  panel-soft: "#F7F9F8"
  hairline: "#D5DEDC"
  hairline-strong: "#B9C6C3"
  chart-grid: "#E6ECEA"
  ink: "#12211F"
  ink-body: "#2E3F3C"
  ink-muted: "#4B5D5A"
  ink-faint: "#5E6E6A"
  report-teal: "#0E6E6A"
  teal-tint: "#E3F0EE"
  teal-mist: "#8FB8B2"
  shell-teal: "#0B3A3C"
  shell-raised: "#124749"
  shell-rule: "#24595A"
  shell-text: "#E3EEEC"
  shell-label: "#A9C4C0"
  shell-accent: "#8FD6C9"
  slate-accent: "#45627A"
  ochre: "#9A5B16"
  ochre-tint: "#FBF3E8"
  ochre-rule: "#E8CFA8"
  positive-green: "#1D7451"
  positive-tint: "#E2F0E9"
  positive-rule: "#BFDCCD"
  negative-oxblood: "#A3322B"
  negative-tint: "#F6E4E2"
  negative-rule: "#E6C3BF"
  neutral-slate: "#4F6470"
  neutral-tint: "#E6EBEE"
  neutral-rule: "#CBD5DB"
typography:
  display:
    fontFamily: "Public Sans, system-ui, sans-serif"
    fontSize: "2.6rem"
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: "-0.02em"
  headline:
    fontFamily: "Public Sans, system-ui, sans-serif"
    fontSize: "2.05rem"
    fontWeight: 700
    lineHeight: 1.15
    letterSpacing: "-0.015em"
  title:
    fontFamily: "Public Sans, system-ui, sans-serif"
    fontSize: "1.14rem"
    fontWeight: 650
    lineHeight: 1.3
    letterSpacing: "-0.015em"
  figure:
    fontFamily: "Public Sans, system-ui, sans-serif"
    fontSize: "1.65rem"
    fontWeight: 700
    lineHeight: 1.25
    fontFeature: "tnum, lnum"
  standfirst:
    fontFamily: "Public Sans, system-ui, sans-serif"
    fontSize: "1.04rem"
    fontWeight: 400
    lineHeight: 1.55
  body:
    fontFamily: "Public Sans, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.6
  label:
    fontFamily: "Public Sans, system-ui, sans-serif"
    fontSize: "0.8rem"
    fontWeight: 600
    lineHeight: 1.4
  source:
    fontFamily: "Public Sans, system-ui, sans-serif"
    fontSize: "0.78rem"
    fontWeight: 400
    lineHeight: 1.4
rounded:
  hairline: "2px"
  band: "3px"
  tag: "4px"
  nav: "5px"
  base: "6px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "18px"
  section: "34px"
  container: "1180px"
components:
  button-primary:
    backgroundColor: "{colors.report-teal}"
    textColor: "{colors.report-white}"
    rounded: "{rounded.base}"
  page-link:
    textColor: "{colors.report-teal}"
    rounded: "{rounded.nav}"
    padding: "0.3rem 0.4rem"
  page-link-hover:
    backgroundColor: "{colors.teal-tint}"
    textColor: "{colors.report-teal}"
  sentiment-tag-positive:
    backgroundColor: "{colors.positive-tint}"
    textColor: "{colors.positive-green}"
    rounded: "{rounded.tag}"
    padding: "2px 9px"
  sentiment-tag-neutral:
    backgroundColor: "{colors.neutral-tint}"
    textColor: "{colors.neutral-slate}"
    rounded: "{rounded.tag}"
    padding: "2px 9px"
  sentiment-tag-negative:
    backgroundColor: "{colors.negative-tint}"
    textColor: "{colors.negative-oxblood}"
    rounded: "{rounded.tag}"
    padding: "2px 9px"
  chip-muted:
    backgroundColor: "{colors.panel-mist}"
    textColor: "{colors.ink-body}"
    rounded: "{rounded.tag}"
    padding: "2px 9px"
  key-figure:
    textColor: "{colors.ink}"
    typography: "{typography.figure}"
    padding: "12px 16px"
  banner:
    backgroundColor: "{colors.panel-soft}"
    textColor: "{colors.ink-body}"
    rounded: "{rounded.base}"
    padding: "12px 16px"
  banner-warning:
    backgroundColor: "{colors.ochre-tint}"
    textColor: "{colors.ink-body}"
    rounded: "{rounded.base}"
    padding: "12px 16px"
  empty-state:
    backgroundColor: "{colors.panel-soft}"
    textColor: "{colors.ink-muted}"
    rounded: "{rounded.base}"
    padding: "34px 28px"
  probability-band:
    rounded: "{rounded.band}"
    height: "26px"
  sidebar-shell:
    backgroundColor: "{colors.shell-teal}"
    textColor: "{colors.shell-text}"
---

# Design System: FinSight

## Overview

**Creative North Star: "The Inflation Report"**

FinSight reads as a central bank's quarterly report on its own models. The page is a white report ground framed by a deep-teal shell; results are set as numbered exhibits, each followed by a source line, and every estimate that has a spread is drawn with its uncertainty band. Structure comes from hairline rules, not from boxes or shadows: a thick teal rule opens a page, an ink rule opens a table of figures, a pale rule opens each section.

Density is that of a publication, not a dashboard: one workhorse sans with tabular figures, a reading measure held under 75ch for standfirsts and captions, and colour reserved for meaning (teal for the house, the three sentiment hues for verdicts, ochre for the one thing that needs attention). The world refuses the dark, glowing KPI-card dashboard: no gradient fills, no card shadows, no glow.

**Key Characteristics:**
- White report ground, cool mist panels, deep-teal sidebar shell.
- Hairline rules carry all structure; shadows are absent at rest.
- Numbered exhibits with a "Source:" line beneath.
- Uncertainty drawn as bands: dot-and-interval charts, segmented probability bands.
- Sentiment always triple-coded: colour, glyph (▲ ▶ ▼) and word.
- One sans (Public Sans, self-hosted 400/500/600/700), tabular lining figures wherever numbers appear.
- Sentence case for every heading, label and action in the report body.

## Colors

A restrained report palette: teal-tinted inks on white, one house teal, three sentiment hues and an ochre caution, every text colour at WCAG AA (4.5:1 or better) on both the white ground and the mist panel.

### Primary
- **Report Teal** (`report-teal`): the house colour. Primary buttons, links and page links, the 3px masthead rule, the "Exhibit n" numeral, focus rings, the first chart series, and the predicted-good row in interval charts. 6.07:1 on white.
- **Teal Tint** (`teal-tint`): hover wash behind page links; the only fill teal is allowed to become.
- **Teal Mist** (`teal-mist`): the last, lightest categorical chart series.

### Secondary
- **Shell Teal** (`shell-teal`): the sidebar rail, the one dark surface in the system. With it: **Shell Raised** (inputs on the rail), **Shell Rule** (hairlines on the rail), **Shell Text** (body copy, 10.5:1), **Shell Label** (group labels and notes, 6.7:1), **Shell Accent** (active-page tick, status dot, links on the rail).
- **Slate Accent** (`slate-accent`): the neutral comparison series in charts (for example the in-domain baseline row) and the third split segment.

### Tertiary
- **Ochre** (`ochre`): warnings, the "experimental" marker, and the one chart series that needs attention. Never decorative. With **Ochre Tint** and **Ochre Rule** for warning banners and the experimental tag.

### Sentiment
- **Positive Green** (`positive-green`), **Neutral Slate** (`neutral-slate`), **Negative Oxblood** (`negative-oxblood`): verdict words, figure values and bars that encode a class. Each has a tint (tag fill, unpredicted band segment) and a rule (tag border, banner border).

### Neutral
- **Report White** (`report-white`): the page ground and chart hover labels.
- **Panel Mist** (`panel-mist`) and **Panel Soft** (`panel-soft`): secondary backgrounds, dataframe headers, banners, empty states, legacy panels.
- **Hairline** (`hairline`): every section rule, table row rule, figure divider and widget border. **Hairline Strong** (`hairline-strong`): axis lines, dashed empty-state border, scrollbars. **Chart Grid** (`chart-grid`): gridlines inside the Plotly template.
- **Ink** (`ink`): headings, figure values, table heads' underline rule. **Ink Body** (`ink-body`): running text and chart text. **Ink Muted** (`ink-muted`): standfirsts, captions, labels. **Ink Faint** (`ink-faint`): source lines, metadata, reference lines (5.36:1, the floor).

### Named Rules
**The Meaning-Only Colour Rule.** Outside the shell and the house teal, colour encodes a class or a state. A key figure takes a sentiment colour only when its value is that sentiment; otherwise it is ink.

**The Triple-Code Rule.** Sentiment is never colour alone: positive is green ▲, neutral is slate ▶, negative is oxblood ▼, and the word is always written beside the glyph.

## Typography

**Display Font:** Public Sans (with system-ui, sans-serif)
**Body Font:** Public Sans (with system-ui, sans-serif)
**Label/Mono Font:** Public Sans for labels; system monospace for code only.

**Character:** One civic, workmanlike sans set like a government publication: firm 650-700 headings with slight negative tracking, plain 400 body, and tabular lining numerals so columns of figures align.

### Hierarchy
- **Display** (700, 2.6rem, 1.1; 2.1rem under 760px): the verdict word only ("▲ Positive").
- **Headline** (700, 2.05rem, 1.15; 1.6rem under 760px): the page masthead title.
- **Title** (650, 1.14rem, 1.3): section and exhibit headings.
- **Figure** (700, 1.65rem, 1.25, tabular): key-figure values.
- **Standfirst** (400, 1.04rem, 1.55, max 68ch): the one-line summary under a masthead; captions run 0.9rem, max 75ch.
- **Body** (400, 1rem, 1.6): running text and list items.
- **Label** (600, 0.8rem): key-figure labels, verdict labels, table heads (0.78rem), sentence case.
- **Source** (400, 0.78rem, ink faint): "Source: ..." lines and metadata (0.82rem).

### Named Rules
**The Tabular Figures Rule.** Every number on screen (figures, metrics, dataframes, tables, band legends) uses `tabular-nums lining-nums`.

**The Sentence Case Rule.** Headings, labels, buttons and table heads in the report body are sentence case. The only uppercase, tracked text in the system is the sidebar's navigation group labels (0.7-0.72rem, 700, 0.12em), which belong to the shell, not the report.

## Layout

A single report column, max 1180px wide, top padding 4.4rem (3.6rem and 16px side padding under 760px). Pages open with a masthead (title, standfirst, metadata line) and an optional primary action in a narrow right column. Sections are separated by a hairline and 34px of space; exhibit and section headings sit 14px below their rule. Key figures run as one auto-fit grid row (minimum 170px per cell). Multi-column content (contents lists, "what happened" steps) is laid out row by row so it reads in order when stacked on a phone. Interval charts place row labels above each band rather than in a left gutter, so charts keep full width at phone size. Breakpoint: 760px.

## Elevation & Depth

The system is flat. Depth is expressed by rules and tone, never by cast shadows: a 3px teal rule over the masthead, a 1px ink rule over key figures, verdicts and table heads, 1px hairlines between sections and rows, and mist panels for secondary surfaces. The header bar is translucent white (92%) over a hairline. The one shadow in the system is a state response: a soft teal lift under the primary button on hover.

### Shadow Vocabulary
- **Primary hover lift** (`box-shadow: 0 2px 8px rgba(14,110,106,0.28)`): primary button hover only.
- **Active nav tick** (`box-shadow: inset 2px 0 0 #8FD6C9`): an inset rule marking the current page on the shell, not elevation.

### Named Rules
**The Hairline Rule.** Structure is drawn with 1px rules (3px for the masthead). If a container needs separating, give it a rule or a mist fill, never a shadow.

## Shapes

Small, quiet corners: 6px for widgets, buttons, banners, empty states and expanders; 5px for navigation and page links; 4px for tags; 3px for probability bands and the experimental marker; 2px for term-table weight bars. Ruled structures (key figures, tables, sections, verdicts) have no corners at all because they are rules, not boxes. Empty states and file drop zones use a dashed border.

## Components

### Buttons
Native Streamlit buttons themed through config.
- **Shape:** gently rounded (6px).
- **Primary:** report teal fill, white 600 label, sentence case ("Analyze a sentence"), Material Symbols icon optional.
- **Hover / Focus:** 0.15s ease on background, border and shadow; primary lifts with the teal hover shadow. Focus is a 2px teal outline at 2px offset everywhere.
- **Page links:** quiet teal 600 text links with a teal-tint wash on hover; used as secondary actions and for the contents list.

### Chips (sentiment tags and pills)
- **Style:** inline, 2px 9px, 4px radius, 600 weight at 0.84em, 1px border. Sentiment kinds use their tint fill, hue text and rule border, and always carry the glyph plus the word.
- **Muted:** mist fill, body ink, hairline border, for non-sentiment keywords.

### Cards / Containers
- **Corner Style:** 6px.
- **Background:** panel soft.
- **Shadow Strategy:** none (see Elevation & Depth).
- **Border:** 1px hairline.
- **Internal Padding:** 18px 20px. Used sparingly; results prefer ruled exhibits over boxes.

### Inputs / Fields
Native widgets with visible 1px hairline borders, 6px radius, teal caret and teal focus. File drop zones are dashed.

### Navigation
The deep-teal shell holds the logo, grouped page navigation (Material Symbols icons per page, uppercase tracked group labels in shell label), the model-engine selector and a ruled engine-status block (7px status dots, shell text). The active page gets a 14% shell-accent wash, an inset 2px accent tick and white 600 text.

### Masthead
A 3px teal top rule, 18px gap, the headline title, a muted standfirst (max 68ch) and a faint metadata line with bold figures. Calling it resets exhibit numbering.

### Exhibits and source lines (signature)
Every chart or result table is an exhibit: a hairline-ruled title-weight heading whose prefix "Exhibit n" is set in teal 700, numbered from 1 on each page, followed by an optional muted caption and, after the exhibit, a faint "Source: ..." line naming the artifact file.

### Key-figure row (signature)
Figures are one ruled table row, not cards: ink rule above, hairline below, hairline dividers between cells, a muted label, a tabular figure value and a faint sub-line (often the confidence interval). Under 760px cells stack, separated by hairlines.

### Uncertainty bands (signature)
- **Interval chart:** one row per estimate; a translucent band (25% opacity, 9px) from low to high, open end caps, a white-ringed dot at the estimate, the value in bold to the right, the row label above the band, and an optional dotted faint reference line (for example "always 'neutral'").
- **Probability band on verdicts:** the three class probabilities as one 26px segmented bar (2px gaps, 3px radius) in negative-neutral-positive order. The predicted class is solid in its hue; the others are tints. A three-column legend gives glyph, word and tabular percentage.

### Verdict
Ink rule above, a muted "Predicted sentiment" label, the display-size verdict in its sentiment hue with glyph and word, a "Model confidence" line, then the probability band.

### Term tables
Vocabulary as a ruled table: muted sentence-case heads over an ink rule, row hairlines, the term in ink 600, a 6px weight bar in the class hue (2px radius, scaled to the largest weight), a right-aligned tabular value, and optionally a sentiment tag for the class it pushes toward.

### Banners and empty states
Banners are panel-soft, hairline-bordered rows with an authored 1.75-stroke SVG icon (teal info, green success, ochre warning, oxblood error, each with its tint and rule). Empty states use a dashed strong hairline, a 26px teal SVG icon, an ink h4 and a muted line of guidance.

### Charts
A registered Plotly template makes every chart a report figure: transparent ground, Public Sans 12px in body ink, left-aligned 14px ink titles, horizontal legends top-left, chart-grid gridlines, strong-hairline axes, white hover labels, and the categorical sequence teal, ochre, slate, oxblood, green, teal mist. Sequential fills use the teal ramp from panel mist to near-black teal.

## Do's and Don'ts

### Do:
- **Do** open every page with the masthead and number its results as exhibits, each followed by a "Source:" line naming the results file.
- **Do** draw any estimate with a spread as a band (interval chart or probability band) rather than a bare number.
- **Do** pair every sentiment colour with its glyph (▲ positive, ▶ neutral, ▼ negative) and its word.
- **Do** set key figures as one ruled row with an ink top rule and hairline dividers.
- **Do** use tabular lining figures for every number and keep every text colour at 4.5:1 or better.
- **Do** use Material Symbols for native widgets and the authored 1.75-stroke SVG set inside custom HTML.
- **Do** take chart colours from the template sequence and sentiment hues from the sentiment tokens, never ad hoc hexes.

### Don't:
- **Don't** put shadows, gradients or glows on containers; separate with hairlines or a mist fill.
- **Don't** return to the dark, glowing KPI-card dashboard or rounded metric cards.
- **Don't** encode sentiment by colour alone, or colour a figure that is not a sentiment value.
- **Don't** add eyebrows or kickers above headings, gradient text, or emoji icons.
- **Don't** use title case or uppercase in the report body; uppercase tracking belongs to the shell's nav group labels only.
- **Don't** load a second typeface; Public Sans carries every role.
