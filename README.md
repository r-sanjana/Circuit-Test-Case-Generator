# 🔌 Circuit Test Case Generator

A Streamlit application that takes an electronic/hardware circuit — as a schematic image, a netlist, or a plain-text description — and uses **Google Gemini** (via **LangChain**) and a **LangGraph map-reduce pipeline** to generate a structured, circuit-specific set of test cases. Results can be reviewed in the app and downloaded as a formatted **Word (.docx)** document.

---

## ✨ Features

- **Multiple input methods** — upload a schematic image (PNG/JPG), paste a netlist or plain-English description, or upload a text-based circuit file.
- **Circuit analysis step** — identifies the circuit name, inputs/outputs, a plain-English description, and (for digital circuits) a full truth table, before generating any test cases.
- **Genuine LangGraph map-reduce generation** — test cases are generated using real LangGraph primitives (`StateGraph`, `Send()`, an `operator.add` reducer, and a reduce node) rather than a manual loop dressed up to look like map-reduce.
- **Category-aware coverage** — the circuit analysis step decides which test-case categories genuinely apply (functional, boundary, fault, power, protection) instead of forcing irrelevant categories onto simple circuits.
- **Structured, validated output** — every test case and analysis result is enforced via Pydantic schemas, so output is always consistent and complete, never free-form text.
- **Pipeline visualization** — an inline diagram (built with plain SVG, no external JS libraries) shows the actual LangGraph map-reduce structure for the current run, plus an optional panel showing LangGraph's own raw graph export as evidence it's a genuine `StateGraph`.
- **Word document export** — generates a professional `.docx` report with a circuit summary, an at-a-glance summary table (color-coded by priority), and detailed per-test-case sections — including blank "Actual Result / Pass-Fail / Tested By" fields so the document doubles as a real test execution log.
- **Session history** — every generation is kept in the sidebar for the current session, so earlier results aren't lost when you analyze a new circuit.
- **Model switching** — pick between multiple Gemini models from the sidebar; useful for working around free-tier rate limits, since each model has its own separate quota.

---

## 🏗️ Architecture

```
                         START
                           │
                 (circuit analysis decides
                  applicable categories)
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
           MAP-1         MAP-2        MAP-3
        (functional)   (boundary)    (fault)
     one focused Gemini call per category,
           run concurrently via Send()
              │            │            │
              └────────────┼────────────┘
                           │
                           ▼
                        REDUCE
              (merge, trim to max count,
                  renumber TC-001...)
                           │
                           ▼
                          END
```

- **MAP step** — for every applicable category, LangGraph's `Send()` API fans out one concurrent, narrowly-focused LLM call (e.g. *"generate only functional test cases for this circuit"*), instead of one call trying to cover every category at once.
- **REDUCE step** — LangGraph's own scheduler guarantees this only runs after every parallel branch has returned. It merges all branches' results (via an `operator.add` state reducer), trims to the requested count, and renumbers everything sequentially (`TC-001`, `TC-002`, ...).

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| LLM | [Google Gemini](https://ai.google.dev/) |
| LLM integration | [LangChain](https://www.langchain.com/) (`langchain-google-genai`) |
| Orchestration | [LangGraph](https://www.langchain.com/langgraph) (`StateGraph`, `Send()`, reducers) |
| Structured output | [Pydantic](https://docs.pydantic.dev/) |
| Word export | [python-docx](https://python-docx.readthedocs.io/) |
| Diagrams | Plain SVG (UI) + [Pillow](https://pillow.readthedocs.io/) (raster export for Word docs) |
| Config | [python-dotenv](https://pypi.org/project/python-dotenv/) |

---

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

> If `GOOGLE_API_KEY` isn't set in `.env`, the app will show a password field in the sidebar so you can paste a key at runtime instead.

---

The app opens automatically in your browser (typically `http://localhost:8501`).

**Basic workflow:**

1. Give the app a circuit — upload a schematic image, paste a netlist/description, or upload a text file.
2. Click **🔍 Analyze Circuit** — see the identified circuit name, inputs/outputs, truth table (if applicable), and a recommended test-case count.
3. Choose how many test cases to generate (auto-capped to a sensible range based on the analysis).
4. Click **🧪 Generate Test Cases** — the LangGraph map-reduce pipeline runs, and results appear in the app.
5. Click **⬇️ Download Test Cases as Word Document** to export.

---

## 📁 Project Structure

```
circuit-test-case-generator/
├── app.py              # Main application (UI, LangGraph pipeline, Word export)
├── requirements.txt     # Python dependencies
├── .env                 # Your Gemini API key (not committed — see .gitignore)
├── .env.example          # Template for .env
└── README.md
```

---

## 🧠 How the LangGraph Pipeline Works

1. **Circuit analysis** (a single Gemini call) identifies the circuit and decides which test-case categories genuinely apply — a simple 2-input logic gate might only need `functional` and `boundary`, while a power supply circuit might need all five categories.
2. **Map step** — `_route_categories()` returns one `Send()` object per applicable category. LangGraph schedules these as concurrent calls to `_generate_category_node()`, each producing test cases for only its assigned category.
3. **Merge** — each branch's returned `test_cases` list is automatically concatenated onto a shared state field, thanks to the field being annotated with LangGraph's `operator.add` reducer.
4. **Reduce step** — `_reduce_node()` runs once every branch has completed (guaranteed by LangGraph's scheduler), trims the merged list to the requested count, and renumbers every test case sequentially.

---

## ⚠️ Known Limitations

- **Session history is not persisted** — results are kept only in Streamlit's session state; refreshing or closing the tab clears history. Download anything you want to keep.
- **Free-tier rate limits** — running categories in parallel means multiple requests hit the Gemini API at roughly the same time, which is more likely to trip a free-tier rate limit than sequential calls. Switching models in the sidebar (each has its own quota) is the current workaround.
- **No duplicate detection across categories yet** — the reduce step trims and renumbers but doesn't currently check for near-duplicate test cases generated independently by different category branches.
- **English-language circuits only** — schematic labels and descriptions are assumed to be in English.

---

## 🗺️ Roadmap / Possible Enhancements

- [ ] Duplicate/near-duplicate test case detection in the reduce step
- [ ] Excel/CSV export alongside the Word document
- [ ] Per-test-case regeneration instead of only full-batch regeneration
- [ ] Automatic retry with backoff on Gemini rate-limit errors
- [ ] LangSmith tracing integration for visual proof of parallel execution
- [ ] Persist session history to disk (JSON or SQLite)
- [ ] PDF export option
- [ ] Multi-circuit batch analysis

---
