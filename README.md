# Re-Framing the Narrative: Passive to Active Voice in Media
### NLP-Driven Detection & Correction of Agent-Drop in Reports of Violence Against Women

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![NLP](https://img.shields.io/badge/NLP-spaCy-red.svg)](https://spacy.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

In mainstream media, reports of violence against women and femicide frequently employ the **passive voice** or **agent-drop headlines** (e.g., *"49 women murdered in April"*, *"She was killed at the train station"*). This linguistic choice structurally sanitizes the crime, detaches accountability from the perpetrator, and frames systematic violence as if it were a natural, agentless occurrence.

**This project** is an NLP-driven linguistic intervention tool designed to dismantle this passive media bias. By analyzing the syntactic dependency trees of news headlines, the system deconstructs passive structures, exposes hidden or implied subjects, and reconstructs the sentences into the **active voice**, placing the perpetrator into the grammatically accountable position they belong.

---

## Core Philosophy & Smart Syntactic Logic

Rather than relying on basic keyword matching, the tool utilizes advanced morphological and syntactic parsing to accurately reconstruct context:

* **Exposing the Hidden Subject:** Structural erasures like `She was killed` are intercepted, pulling the victim out from behind ambiguous pronouns and explicitly translating the target into `the woman`.
* **Statistical vs. Situational Plurality:** Large-scale statistical data or macro-headlines (e.g., *"49 women murdered"*) are automatically assigned the plural perpetrator **`Men`** instead of an inaccurate singular *"A man"*, framing it accurately as a systemic societal statistic.
* **Context-Aware Dependency Resolution:** In headlines like *"Two children's mother was stabbed"* the system is not fooled by the numeric descriptor `Two` or the plural noun `children`. It dynamically traces the tree to verify that the true syntactic victim is the singular `mother`, correctly preserving the perpetrator as **`A man`**.

---

## 🚀 The Paradigm Shift (Sample Transformations)

| Original Media Framing (Passive / Agent-Drop) | Re-Framed Linguistic Output (Active Voice) | Syntactic Resolution Logic |
| :--- | :--- | :--- |
| ❌ *She was killed at the train station.* | **A man** killed **the woman** at the train station. | Removed pronoun erasure; exposed both the victim and the perpetrator. |
| ❌ *49 women murdered in April.* | **Men** murdered 49 women in April. | Resolved implicit short-passive; assigned macro-plurality (**Men**). |
| ❌ *Two children's mother was stabbed to death.* | **A man** stabbed two children's mother to death. | Ignored local numeric modifiers; locked onto the singular head noun (`mother`). |

---

## 🛠️ Architecture & Installation

The tool is built on top of the industry-standard **spaCy NLP pipeline** framework and utilizes **WordCloud / Matplotlib** for downstream analytical visualization.

### 1. Prerequisites
Install the required packages and download the English core NLP model via your terminal:

```bash
pip install spacy matplotlib wordcloud
python -m spacy download en_core_web_sm