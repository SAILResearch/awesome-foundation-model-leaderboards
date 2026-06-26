# Contribution Guide for Automated Agents

This file tells the GitHub Copilot coding agent (and any other automated
contributor) how to add entries to `README.md`. Follow it exactly.

## Goal

Add newly discovered AI **leaderboards and benchmarks** to `README.md`, sourced
from HuggingFace Papers (https://huggingface.co/papers). Each run is triggered by
a weekly issue listing the past 7 daily pages to scan.

## Inclusion rules

Add an entry **only if** all of the following hold:

- It introduces or hosts a leaderboard, benchmark, ranking, or evaluation
  platform — not merely a model or dataset release.
- It is **AI-related**.
- It is **actively maintained** (recent activity; not archived/abandoned).
- It has a **working, live leaderboard/benchmark website** that displays the
  ranking or evaluation results. An arXiv page or paper PDF alone does **not**
  qualify — the paper must point to a real, browsable leaderboard site or
  interactive HuggingFace Space hosting the leaderboard.

## Deduplication (mandatory)

Before adding anything, search `README.md` for both the candidate's **URL** and
its **name**. If either already appears anywhere in the file, **skip it**. Do not
add near-duplicates of an existing entry.

## Row format

Each entry is a single Markdown table row:

```
| [Name](url) | One concise sentence describing what it evaluates or ranks. |
```

- The name is the link text. The URL **must** be the live leaderboard/benchmark
  website (or the interactive HuggingFace Space hosting it) — **never** an arXiv
  link, paper PDF, or HuggingFace Papers URL.
- The description is one sentence, present tense, ending with a period.
- Match the tone and length of surrounding rows.
- Where a table is already sorted alphabetically by name, insert the new row in
  alphabetical position. Otherwise append to the end of the table.

## Where to place an entry

Pick the single best-fitting table. Top-level sections:

- **Tools** — backends, submission/eval tooling, competition-creation utilities.
- **Challenges** — competition-hosting platforms (Kaggle, EvalAI, Tianchi, ...).
- **Rankings** — the main body, organized as:

  - **Model Ranking** with these categories (use the closest match):
    Comprehensive, Text, Code, Image, Video, Math, Agent, Research, Business,
    Safety, Medical, Audio, Embodied, 3D, Game, Multimodal, Time Series.
  - **Database Ranking**
  - **Dataset Ranking**
  - **Metric Ranking**
  - **Infrastructure Ranking**
  - **Paper Ranking**
  - **Usage Ranking**
  - **Company Ranking**

If a candidate spans several model domains, use **Comprehensive**. If it does not
fit any existing category, prefer the closest one rather than inventing a new
section.

## Output

- Make all edits in `README.md` only.
- Open a pull request; **do not merge** — a human reviews it.
- If no qualifying new entries are found this week, comment on the triggering
  issue stating that, and close it without opening a PR.
