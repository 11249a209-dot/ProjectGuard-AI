# ProjectGuard AI

ProjectGuard AI is a local, faculty-led academic project similarity review prototype. It accepts PDF and DOCX uploads, extracts document text, compares it with an institution-scoped SQLite repository, and presents transparent textual, semantic, and section-level signals. Similarity is evidence for review, not a plagiarism finding.

## Requirements

- Python 3.11 or newer (the core dependencies should be installed in a virtual environment)
- Node.js 20.19+ and npm
- Internet access for the first dependency install; Sentence-BERT model download is optional

## Run locally

Open two terminals from this project directory.

Backend:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

Before starting the API, edit `backend/.env` locally and set your faculty name, email, password (at least 8 characters), and department. That account is created or its password refreshed at startup. Keep `.env` private; it is ignored by Git. The API starts at `http://127.0.0.1:5000` and creates `backend/database/projectguard.sqlite3`, upload storage, and five clearly labelled sample repository records.

Frontend:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Open the Vite URL printed in the terminal (normally `http://localhost:5173`). To point the frontend at another API origin, set `VITE_API_URL` to the origin without `/api`, for example `http://127.0.0.1:5000`.

Run the backend tests from the activated backend environment:

```powershell
cd backend
python -m unittest discover -s tests -v
```

After installing `requirements-semantic.txt`, the real model-load integration test can also be run with:

```powershell
$env:PROJECTGUARD_RUN_REAL_SBERT = "1"
python -m unittest discover -s tests -v
```

## Demo accounts

- Student: `student@projectguard.demo` / `Student123!`
- Faculty: the email/password configured in `backend/.env` (defaults to `faculty@projectguard.demo` / `Faculty123!` when no configuration is provided)

Registration creates student accounts. Faculty accounts are provisioned from the local configuration; the simple login screen accepts either role’s email and password.

## Analysis method

1. PyMuPDF reads PDF text and python-docx reads DOCX paragraphs and tables. Image-only/scanned PDFs need OCR and are rejected when too little text can be extracted.
2. NLTK tokenizes and normalizes text, removes English stop words, and lemmatizes when the WordNet corpus is available. If that corpus is unavailable, the analyzer uses NLTK's Porter stemmer; scikit-learn's built-in English stop-word list avoids a required corpus download.
3. Scikit-learn TF-IDF with unigram/bigram features and cosine similarity calculates textual overlap against every eligible repository project. Section scores and matching sentence pairs are derived from the actual extracted documents, and ranked scores are retained for the report.
4. Install `backend/requirements-semantic.txt` to enable Sentence-BERT (`all-MiniLM-L6-v2`). Its weights are downloaded on first use and the model instance is cached for reuse. Long documents are split into sentence chunks and represented by a normalized mean embedding to limit encoder truncation. If package/model loading or inference fails, semantic analysis is marked unavailable; the system does not present TF-IDF as a semantic score.
5. With Sentence-BERT available, overall score is 60% TF-IDF and 40% Sentence-BERT cosine similarity. With the fallback, overall score equals TF-IDF. The report names both the scoring and sentence-matching methods. Semantic overlap alone does not establish copied content.
6. The student Project Idea Checker submits a proposed title and abstract for the same repository comparison without creating a project record. Its revision prompts are evidence-based suggestions, not guarantees of originality.

The displayed review bands are 0–15% low, 16–30% moderate, 31–50% high, and over 50% very high. These are configurable review indicators, not universal academic standards. Final academic decisions require faculty review.

## Storage and API

SQLite, uploads, and generated reports live under `backend/` by default. Set `PROJECTGUARD_DB`, `PROJECTGUARD_UPLOADS`, and `PROJECTGUARD_REPORTS` to change those locations. The API includes health, registration/login/logout, project upload/list/detail, `POST /api/ideas/check`, repository search, ranked analysis/results, faculty dashboard/comparison/review, and PDF report endpoints. Bearer session tokens are stored in SQLite; this is a local prototype, not a production identity provider.

For production deployment, add institution-managed authentication, CSRF protections for browser sessions, request rate limiting, encrypted document storage, retention policies, audit logging, and operational controls around student records.
