# RegisterVaani Lite

**A Voice-to-Form AI Assistant for Community Field Workers**

RegisterVaani Lite turns synthetic field notes into editable visit-register suggestions with local reference retrieval, evidence, deterministic checks and explicit human approval. It is a data-entry demonstration—not a medical diagnosis or clinical decision-support system. Use only synthetic records and reference material.

## Start locally

Requirements: Python 3.10–3.12, Node.js 18+, npm, and internet access on first run so the Sentence Transformers model can be downloaded. FAISS, embeddings, SQLite, and the web app run on your computer. LLM access is optional.

```bash
# From the project root
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

# Terminal 1 — API
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 6007

# Terminal 2 — frontend
npm install
npm run dev
```

Open the Vite URL, normally <http://127.0.0.1:5173>. API health: <http://127.0.0.1:6007/api/health>; interactive API docs: <http://127.0.0.1:6007/docs>. SQLite is created automatically at `backend/data/registervaani.sqlite3`; the local FAISS index and its chunk metadata are saved beside it. Do not commit local data or secrets. The `.gitignore` excludes runtime artifacts and `.env`.

## Optional LLM provider

The full reference retrieval and English local demo extraction work without an API key. The local fallback deliberately extracts conservatively from explicit English phrases and is clearly labelled as demo mode. Regional-language translation into standard English register fields requires a configured OpenAI-compatible LLM; it is not available in the offline fallback. To use an OpenAI-compatible LLM for structured field extraction and translation:

1. Copy `.env.example` to `.env` in the repository root.
2. Set `OPENAI_API_KEY`, `OPENAI_BASE_URL` (default `https://api.openai.com/v1`), and `OPENAI_MODEL` to values from a provider you choose.
3. Restart the API process.

Keys are read only by the backend and never sent to the browser. The transcript and retrieved references are sent to that external provider when LLM mode is enabled. **Do not send real personal, citizen or health information.** If the model call fails, the app clearly marks its conservative local fallback rather than claiming LLM extraction succeeded. No paid service is mandatory.

The language selector supports browser dictation locales for English (India), Hindi, Kannada, Tamil, Telugu, Malayalam, Marathi, Bengali, Gujarati, Punjabi and Urdu. Typed notes can also be entered in other languages, but select the closest supported language so the model receives that hint. In LLM mode, text fields are requested in standard English, names and places are transliterated to Latin script, and evidence remains quoted from the original note. Review every translation before approval. Browser speech recognition support and locale availability vary by browser/device; Chrome or Edge on localhost is the recommended demo setup.

## Use the application

1. On **New visit**, type a synthetic field note or choose a language and use optional browser dictation; check and correct the transcript. Dictation locales depend on browser and device support.
2. Select **Extract register fields**. The service searches local Markdown references via Sentence Transformers embeddings and FAISS, then extracts into the fixed Pydantic schema and applies deterministic checks.
3. On **Review**, inspect the original note, retrieved context, each editable field, evidence and flags. Answer one clarification at a time (up to two rounds); use “unknown” if a requested answer is not available. Then save the draft or explicitly approve it.
4. Browse or filter saved records and export a CSV on **Visit records**. The dashboard reports actual SQLite metrics.
5. Add a local Markdown guide or run an actual semantic query in **Knowledge base**. Rebuild the persisted index after adding a document.
6. Run the fixed synthetic checks under **Evaluation**; results are stored locally and are not accuracy claims about real field work.

Approval requires person name and village plus no illustrative age/weight/date validation errors. These bounds are demo data-quality checks, not medical standards. Approved entries cannot be edited in this MVP. A draft can be cancelled from its review page. Cancel removes the draft row; it does not remove the supplied Markdown references.

## Architecture & data flow

```text
┌──────────────────────────────────────────────┐
│ React + TypeScript + Vite                     │
│ Dashboard · New Visit · Review · Records      │
│ Knowledge Base · Evaluation · Settings        │
└──────────────────────┬───────────────────────┘
                       │ HTTP / JSON (CORS: local Vite only)
┌──────────────────────▼───────────────────────┐
│ FastAPI + Pydantic                             │
│ extract · clarify · validate · approve · CSV  │
└──────────┬────────────────────┬───────────────┘
           │                    │
  Markdown reference docs       ├───────────────┐
           │                    │               │
  section-aware chunks          │ optional      │
           ▼                    │ OpenAI-       ▼
  Sentence Transformers         │ compatible  SQLite
  local embeddings              │ JSON LLM    visits + evaluation runs
           ▼                    │
  FAISS vector search ──────────┴─ retrieved passages
                                      │
                                      ▼
                         Pydantic checks → editable review
                                      │ explicit human approval
                                      ▼
                           approved record → records / CSV
```

Reference text is context, never executable instructions. The LLM prompt requires JSON in the fixed schema, quotes from the original note, and null for unsupported values; quoted evidence is checked against the submitted transcript. The non-LLM demo extractor is a small transparent heuristic, **not an LLM**. Every suggestion requires human review. Transcription (where available) is a separate optional browser capability.

### API routes

- `GET /api/health`, `GET /api/dashboard`
- `POST /api/visits/extract`, `POST /api/visits/clarify`
- `POST /api/visits`, `GET /api/visits`, `GET /api/visits/{id}`, `PATCH /api/visits/{id}`
- `DELETE /api/visits/{id}` (draft/unresolved only), `POST /api/visits/{id}/approve`
- `GET /api/visits/export/csv`
- `GET /api/knowledge-base/documents`, `POST /api/knowledge-base/documents` (Markdown multipart), `POST /api/knowledge-base/reindex`, `POST /api/knowledge-base/search`
- `POST /api/evaluation/run`, `GET /api/evaluation/results`

## Evaluation definitions

The included fixed corpus contains **40 synthetic cases**: 29 complete notes (including explicit-date and alternate-phrasing notes) plus 11 edge cases (unknown measurements, an invalid age and weight, an ambiguous weekday, a relative follow-up week, informal wording, mixed punctuation, a negative follow-up, contradictory ages, empty input, and household/worker-only details). The evaluator reports exact field-value matches over only explicitly expected fields; numeric values are represented consistently as numbers. Run it from the Evaluation page to see the current measured values on this machine. Required-field completeness counts populated, correct person-name and village values over the two required fields for every case. Invalid detection is the fraction of explicitly invalid synthetic age/weight cases flagged by the same validation rules. Missing-value hallucination measures non-null output where the expected value is null. Retrieval Recall@3 is measured on three fixed questions and their expected source document names. Manual correction rate remains **not measured** because the dataset contains no human correction annotations. The extractor smoke checks currently exercise each deterministic test case and compare its expected fields; the API's evaluation endpoint additionally measures the retrieval cases against a built index. Metrics evaluate only the current deterministic fallback and synthetic corpus, not the configured external LLM, speech recognition, real-world accuracy, or clinical performance.

Because the rule-based demo extractor and this fixed corpus were written together, a high score here validates that the pipeline, schema, validation and scoring work as specified — it is **not** an estimate of extraction accuracy on new, unseen field notes. Report it that way in a viva.

Retrieval Recall@3 is only reported once a real FAISS index exists. With the model installed and the index rebuilt over the three bundled references (11 chunks), it measured **1.0** on the three fixed questions, i.e. the expected source document appeared in the top 3 each time. Treat that as a smoke test of the retrieval wiring, not a retrieval-quality claim: three queries over three documents is far too small a sample to generalise from. Recall stays blank until you rebuild an index yourself.

## Troubleshooting

- **Frontend says API offline / CORS:** start the backend on `127.0.0.1:6007`; start Vite on its normal `127.0.0.1:5173` origin. Set `CORS_ORIGINS` if using another local dev port. Set `VITE_API_URL` only if the API base changes (the default is `http://127.0.0.1:6007/api`).
- **FAISS / embeddings unavailable:** ensure backend requirements installed in the active virtual environment; use a supported Python version and check disk/network access for the first local model download. Visit Knowledge Base → **Rebuild FAISS index**. A failed build is surfaced as an API error; extraction explains when retrieval context was unavailable.
- **Offline embedding download:** download/cache `sentence-transformers/all-MiniLM-L6-v2` from a trusted source before running, or set `EMBEDDING_MODEL` to a compatible local Sentence Transformers model directory. A relative path is resolved against the project root, so `EMBEDDING_MODEL=.cache/models/all-MiniLM-L6-v2` works from any shell. If the Hugging Face download stalls, fetch the model files (`modules.json`, `config.json`, `config_sentence_transformers.json`, `sentence_bert_config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`, `1_Pooling/config.json`, `model.safetensors`) with `curl` into that folder instead.
- **LLM unavailable / malformed JSON:** check the optional environment values and provider's JSON response support, then restart the backend. Extraction reports its explicitly labelled local fallback; no secret needs to be entered in the browser.
- **Microphone unavailable:** speech recognition depends on browser support, permission, and secure-context policies. On localhost, try a supported Chromium browser and grant microphone permission; otherwise type or correct the transcript.
- **SQLite changes not visible:** API and frontend must use the same project directory; persistent file path is `DATABASE_PATH` or `backend/data/registervaani.sqlite3`. Restarting the API does not remove records.

## Project structure

```text
backend/main.py                 FastAPI routes, SQLite, validation, FAISS, evaluation
backend/knowledge_base/*.md     Synthetic local references
backend/data/                   Runtime SQLite and FAISS artifacts (ignored)
src/App.tsx                     React routes, screens and API workflow
src/api.ts                      Typed fetch helpers
src/style.css                   Responsive design system
.env.example                    Optional backend configuration template
```
