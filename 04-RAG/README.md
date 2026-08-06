# 04 — Retrieval-Augmented Generation (RAG)

Build a question-answering system that answers from **your own documents** instead of from whatever the
language model happened to memorise during training.

The corpus in [`documents/`](./documents) is a small set of automotive knowledge-base articles
(EV charging, drivetrain, driver assistance, servicing, safety, warranty, model line-up, manufacturing).

> ⚠️ The documents are **simplified teaching material written for this exercise**. Figures, model names
> and warranty terms are illustrative and are not real product specifications. That is deliberate — it
> means the LLM cannot possibly know the answers from pre-training, so any correct answer must have come
> from retrieval. That is exactly what we want to demonstrate.

---

## 1. Setup

### Step 1 — Python environment

From a terminal, in this folder:

```powershell
cd 04-RAG
uv venv .venv
uv sync
.\.venv\Scripts\Activate.ps1
```

Then in VS Code, open `simple_RAG.ipynb` and select the `.venv` kernel (top right → Select Kernel →
Python Environments → `.venv`).

The first run downloads the embedding model (~420 MB) from Hugging Face. This happens **once**; later
runs load it from the local cache.

### Step 2 — Choose a language model

The notebook needs an LLM to generate answers. Pick **one** of the options below.
The notebook has a single `get_llm()` helper, so switching provider changes exactly one setting.

| Option | Provider | Cost | Best for |
|---|---|---|---|
| **A (default)** | **Ollama**, local | Free, no key, no limits | Machines with ≥8 GB free RAM |
| **B** | **Groq** API | Free tier, **no credit card** | Weaker laptops, fastest responses |
| **C** | **Google Gemini** API | Free tier, no credit card | Alternative to Groq, works well in Colab |

#### Option A — Ollama (recommended default)

1. Download the installer from <https://ollama.com/download> and run it. Ollama installs as a background
   service and starts automatically with Windows.
2. Pull a model:
   ```powershell
   ollama pull llama3.1
   ```
3. Verify it works:
   ```powershell
   ollama run llama3.1
   ```
   Type a prompt, check you get a reply, then `/bye` to exit.

Ollama listens on `http://localhost:11434`. If your machine struggles with `llama3.1` (8B), use the
smaller `llama3.2:3b` instead — pull it and set `OLLAMA_MODEL=llama3.2:3b`.

**Why local is the default:** no keys, no rate limits, no internet dependency, and your documents never
leave your machine. That last point is the actual reason companies build RAG systems in the first place.

#### Option B — Groq (free API fallback)

1. Sign in at <https://console.groq.com> with a Google account. No credit card required.
2. Create an API key.
3. Copy `.env.example` to `.env` and set:
   ```
   LLM_PROVIDER=groq
   GROQ_API_KEY=gsk_your_key_here
   ```

Groq serves `llama-3.3-70b-versatile` — the same model family as the Ollama default, so notebook
behaviour stays comparable. It is also extremely fast.

#### Option C — Google Gemini (free API fallback)

1. Get a key at <https://aistudio.google.com/apikey>. No credit card required.
2. In `.env`:
   ```
   LLM_PROVIDER=gemini
   GOOGLE_API_KEY=your_key_here
   ```

> 🔐 **Never commit API keys.** `.env` is git-ignored. Never paste a key directly into a notebook cell —
> notebook outputs get committed, and a key pasted in code is a key you have published.

### Running in Google Colab

Colab works for this module. Free CPU is plenty for embeddings and FAISS. Use Groq or Gemini for
generation — running Ollama inside Colab is possible but awkward and not recommended.

---

## 2. How to work through the notebook

Open [`simple_RAG.ipynb`](./simple_RAG.ipynb) and work top to bottom.

| Part | Content | Jira |
|---|---|---|
| **Part 0** | Concepts and setup. Fully worked and runnable — **read and run this first.** | — |
| **Part 1** | Baseline RAG pipeline | **M4.1** |
| **Part 2** | Retrieval quality experiments | **M4.2** |
| **Part 3** | Conversational RAG | **M4.3** |
| **Part 4** | Evaluation and failure analysis | **M4.4** |
| **Part 5** | Personal extension | **M4.5** |

Part 0 is worked for you. Parts 1–5 are **skeletons with TODOs** — that is where you write code.

Each ticket part follows the same shape:

> 🎯 **Goal** → 📚 **What you need to know** → 🔧 **YOUR TURN** → ✅ **Definition of Done** → 📝 **Conclusion**

Fill in the conclusion cell in your own words. *"This did not work and here is what I think went wrong"*
is a perfectly good conclusion — negative results are results, and they are usually where the learning is.

---

## 3. The one rule that breaks everything if you get it wrong

**Index and query must use the same embedding model.**

The vector store is built by embedding your documents. When you ask a question, the question is embedded
too, and the two are compared. If those embeddings came from different models, the comparison is
meaningless — different models put "meaning" in different places in vector space.

Two ways this goes wrong:

| Situation | What happens |
|---|---|
| Models have different dimensions (768 vs 384) | FAISS raises a dimension error — **loud, you notice immediately** |
| Models have the same dimensions but are different | Everything runs, results are quietly nonsense — **silent, and much worse** |

So: **if you change the embedding model, you must re-embed the whole corpus and rebuild the vector
store.** It is not a drop-in swap. The notebook stores each index in a per-model folder
(`faiss_index__<model-name>/`) specifically to make this mistake hard to make.

Part 0.5 demonstrates the silent failure on purpose so you can recognise it.

---

## 4. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ConnectionError` / `Connection refused` on port 11434 | Ollama isn't running | Run `ollama serve`, or switch to Groq/Gemini in `.env` |
| `model "llama3.1" not found` | Model not downloaded | `ollama pull llama3.1` |
| First cell takes minutes with a progress bar | Downloading the embedding model (~420 MB) | Normal, once only. Let it finish |
| Ollama answers extremely slowly / machine freezes | Not enough RAM for an 8B model | Use `llama3.2:3b`, or switch to Groq |
| `AssertionError` about dimensions when searching | Index built with a different embedding model | Delete the `faiss_index__*` folder and rebuild |
| Answers suddenly became nonsense after you changed a setting | Stale index from a previous embedding model | Delete `faiss_index__*` and rebuild |
| `ModuleNotFoundError` | Wrong kernel selected | Select the `.venv` kernel; re-run `uv sync` |
| Kernel dies while embedding | Out of memory | Reduce corpus size, or use `all-MiniLM-L6-v2` |
| Answer is "I don't know" for something you know is in the docs | Retrieval problem, not the LLM | Print the retrieved chunks (Part 0.7) — raise `k`, or reduce chunk size |
| `RateLimitError` from Groq/Gemini | Free tier limit reached | Wait, or switch back to Ollama |

**Golden debugging rule:** when an answer is wrong, always look at the retrieved chunks *before*
blaming the model. Most RAG failures are retrieval failures, and you cannot see them from the answer alone.

---

## 5. What you should understand by the end

- Why RAG exists, and what problem it solves that a bare LLM cannot
- What an embedding is, and why similar meanings end up close together
- Why chunk size and overlap change your answers
- How to inspect what was retrieved, and why that is the primary debugging tool
- The difference between a **retrieval** failure and a **generation** failure
- Why conversational follow-ups need the question rewritten before retrieval
- How to evaluate a RAG system honestly, including where it fails
