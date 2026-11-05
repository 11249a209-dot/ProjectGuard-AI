import json
import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import nltk
import numpy as np
import pymupdf as fitz
from docx import Document
from flask import Flask, jsonify, make_response, request, send_file
from flask_cors import CORS
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent


def load_env_file(path):
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        key, value = key.strip(), value.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


load_env_file(BASE_DIR / ".env")
DB_PATH = Path(os.environ.get("PROJECTGUARD_DB", BASE_DIR / "database" / "projectguard.sqlite3"))
UPLOAD_DIR = Path(os.environ.get("PROJECTGUARD_UPLOADS", BASE_DIR / "uploads"))
REPORT_DIR = Path(os.environ.get("PROJECTGUARD_REPORTS", BASE_DIR / "reports"))
ALLOWED_EXTENSIONS = {"pdf", "docx"}
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
SECTION_NAMES = ["Abstract", "Introduction", "Objectives", "Methodology", "Proposed System", "Conclusion"]
SBERT_MODEL = None
SBERT_ATTEMPTED = False
SBERT_LOCK = threading.Lock()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
CORS(app, supports_credentials=False)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, factory=ClosingConnection)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, exception_type, exception, traceback):
        try:
            return super().__exit__(exception_type, exception, traceback)
        finally:
            self.close()


def extract_text(path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with fitz.open(path) as document:
            return "\n".join(page.get_text("text") for page in document).strip()
    if suffix == ".docx":
        document = Document(path)
        blocks = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
        for table in document.tables:
            blocks.extend(" ".join(cell.text for cell in row.cells) for row in table.rows)
        return "\n".join(blocks).strip()
    raise ValueError("Only PDF and DOCX documents are supported.")


def seed_database(connection):
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('student', 'faculty')),
            department TEXT NOT NULL DEFAULT 'Computer Science'
        );
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            title TEXT NOT NULL,
            abstract TEXT NOT NULL DEFAULT '',
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL DEFAULT '',
            department TEXT NOT NULL,
            academic_year TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'Research',
            team_members TEXT NOT NULL DEFAULT '',
            body_text TEXT NOT NULL DEFAULT '',
            is_sample INTEGER NOT NULL DEFAULT 0,
            upload_date TEXT NOT NULL,
            analysis_status TEXT NOT NULL DEFAULT 'Pending',
            review_status TEXT NOT NULL DEFAULT 'Awaiting review'
        );
        CREATE TABLE IF NOT EXISTS similarity_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            reference_project_id INTEGER REFERENCES projects(id),
            textual_similarity REAL NOT NULL,
            semantic_similarity REAL NOT NULL,
            semantic_available INTEGER NOT NULL DEFAULT 0,
            overall_similarity REAL NOT NULL,
            semantic_method TEXT NOT NULL,
            section_results TEXT NOT NULL,
            matching_content TEXT NOT NULL,
            similar_projects TEXT NOT NULL DEFAULT '[]',
            suggestion_results TEXT NOT NULL DEFAULT '[]',
            analysis_date TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS faculty_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            faculty_id INTEGER NOT NULL REFERENCES users(id),
            review_status TEXT NOT NULL,
            comments TEXT NOT NULL DEFAULT '',
            review_date TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL
        );
    """)
    result_columns = {row["name"] for row in connection.execute("PRAGMA table_info(similarity_results)")}
    migrations = {
        "semantic_available": "INTEGER NOT NULL DEFAULT 0",
        "similar_projects": "TEXT NOT NULL DEFAULT '[]'",
        "suggestion_results": "TEXT NOT NULL DEFAULT '[]'",
    }
    for column, definition in migrations.items():
        if column not in result_columns:
            connection.execute(f"ALTER TABLE similarity_results ADD COLUMN {column} {definition}")
    student = connection.execute("SELECT id FROM users WHERE email = ?", ("student@projectguard.demo",)).fetchone()
    if not student:
        cursor = connection.execute(
            "INSERT INTO users(name,email,password_hash,role,department) VALUES(?,?,?,?,?)",
            ("Jordan Lee", "student@projectguard.demo", generate_password_hash("Student123!"), "student", "Computer Science"),
        )
        student_id = cursor.lastrowid
    else:
        student_id = student["id"]
    faculty_name = os.environ.get("PROJECTGUARD_FACULTY_NAME", "Dr. Avery Chen").strip()
    faculty_email = os.environ.get("PROJECTGUARD_FACULTY_EMAIL", "faculty@projectguard.demo").strip().lower()
    faculty_password = os.environ.get("PROJECTGUARD_FACULTY_PASSWORD", "Faculty123!")
    faculty_department = os.environ.get("PROJECTGUARD_FACULTY_DEPARTMENT", "Computer Science").strip()
    if len(faculty_name) < 2 or "@" not in faculty_email or len(faculty_password) < 8:
        raise RuntimeError("Configured faculty credentials require a name, valid email, and password with at least 8 characters.")
    faculty = connection.execute("SELECT id FROM users WHERE email = ?", (faculty_email,)).fetchone()
    if not faculty:
        connection.execute(
            "INSERT INTO users(name,email,password_hash,role,department) VALUES(?,?,?,?,?)",
            (faculty_name, faculty_email, generate_password_hash(faculty_password), "faculty", faculty_department or "Computer Science"),
        )
    elif os.environ.get("PROJECTGUARD_FACULTY_EMAIL") or os.environ.get("PROJECTGUARD_FACULTY_PASSWORD"):
        connection.execute(
            "UPDATE users SET name=?,password_hash=?,role='faculty',department=? WHERE email=?",
            (faculty_name, generate_password_hash(faculty_password), faculty_department or "Computer Science", faculty_email),
        )
    samples = [
        ("Sample · Campus Energy Monitor", "A low-cost sensor network that forecasts classroom energy demand and recommends efficient lighting and HVAC schedules.", "The system collects occupancy, temperature, and power readings from campus rooms. A forecasting model identifies peak demand and recommends schedules that reduce energy use without affecting student comfort.", "Computer Science", "2025–26", "Sustainability"),
        ("Sample · Student Support Assistant", "A retrieval-based assistant that answers student questions using verified university policies and service information.", "The proposed assistant indexes approved university documents, retrieves relevant passages, and presents concise answers with source links. Evaluation focuses on answer relevance, privacy, and escalation to staff when confidence is low.", "Information Technology", "2025–26", "Applied AI"),
        ("Sample · Crop Health Vision", "A mobile image classification tool to help small farms identify common crop leaf diseases and receive practical guidance.", "The project trains a convolutional image classifier on labeled leaf photographs. The mobile interface reports likely disease categories and confidence, while an offline mode supports field use with limited connectivity.", "Agricultural Engineering", "2024–25", "Applied AI"),
        ("Sample · Campus Energy Forecasting", "Predictive campus building management using smart meters, room occupancy, and machine learning for energy optimization.", "Smart meters and occupancy sensors stream building data into a predictive model. The dashboard forecasts energy consumption and suggests optimized HVAC and lighting controls for university buildings while preserving indoor comfort.", "Computer Science", "2024–25", "Sustainability"),
        ("Sample · Library Seat Finder", "A privacy-aware occupancy map that helps students find available study spaces across the university library.", "The application combines anonymous desk sensors with floor maps to estimate seat availability. Students can filter by quiet zones, equipment, and accessibility, while the system avoids storing personally identifying movement data.", "Information Technology", "2025–26", "Campus Services"),
    ]
    count = connection.execute("SELECT COUNT(*) AS count FROM projects WHERE is_sample = 1").fetchone()["count"]
    if count == 0:
        for title, abstract, body, department, year, category in samples:
            connection.execute(
                """INSERT INTO projects(user_id,title,abstract,filename,department,academic_year,category,body_text,is_sample,upload_date,analysis_status,review_status)
                   VALUES(?,?,?,?,?,?,?,?,1,?,'Analyzed','Approved')""",
                (student_id, title, abstract, "Sample corpus", department, year, category, f"{title}\nAbstract\n{abstract}\nIntroduction\n{body}\nObjectives\nImprove campus services through a reliable and accessible digital system.\nMethodology\nCollect data, preprocess records, evaluate a baseline model, and review results with users.\nProposed System\nA web dashboard presents recommendations and supports transparent human review.\nConclusion\nThe prototype demonstrates a practical, privacy-conscious workflow and identifies future evaluation needs.", utc_now()),
            )
    connection.commit()


def init_db():
    with connect_db() as connection:
        seed_database(connection)


init_db()


def current_user():
    header = request.headers.get("Authorization", "")
    token = header.removeprefix("Bearer ").strip()
    if not token:
        return None
    with connect_db() as connection:
        return connection.execute(
            "SELECT users.* FROM sessions JOIN users ON users.id=sessions.user_id WHERE sessions.token=?", (token,)
        ).fetchone()


def auth_required(role=None):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            user = current_user()
            if not user:
                return jsonify({"error": "Please sign in to continue."}), 401
            if role and user["role"] != role:
                return jsonify({"error": "This action is not available for your account role."}), 403
            request.user = user
            return function(*args, **kwargs)
        return wrapped
    return decorate


def project_dict(row, include_text=False):
    value = dict(row)
    value["is_sample"] = bool(value.get("is_sample", 0))
    if include_text:
        return value
    value.pop("body_text", None)
    value.pop("file_path", None)
    value.pop("user_id", None)
    return value


def get_project(connection, project_id):
    return connection.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()


def preprocess(text):
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    tokens = nltk.tokenize.regexp_tokenize(cleaned, r"[a-z0-9]+")
    try:
        stop_words = set(nltk.corpus.stopwords.words("english"))
    except LookupError:
        stop_words = set(TfidfVectorizer(stop_words="english").get_stop_words() or ())
    try:
        lemmatizer = nltk.stem.WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    except LookupError:
        tokens = [nltk.stem.PorterStemmer().stem(token) for token in tokens if token not in stop_words]
    return " ".join(tokens)


def get_sbert():
    global SBERT_MODEL, SBERT_ATTEMPTED
    with SBERT_LOCK:
        if SBERT_ATTEMPTED:
            return SBERT_MODEL
        SBERT_ATTEMPTED = True
        if os.environ.get("PROJECTGUARD_DISABLE_SBERT", "").lower() in {"1", "true", "yes"}:
            return None
        try:
            from sentence_transformers import SentenceTransformer
            SBERT_MODEL = SentenceTransformer(os.environ.get("PROJECTGUARD_SBERT_MODEL", "all-MiniLM-L6-v2"))
        except Exception as error:
            app.logger.info("Sentence-BERT unavailable; using TF-IDF fallback: %s", error)
            SBERT_MODEL = None
        return SBERT_MODEL


def section_texts(text):
    headings = {name.lower(): name for name in SECTION_NAMES}
    pattern = re.compile(r"(?im)^\s*(abstract|introduction|objectives|methodology|proposed system|conclusion)\s*[:\-]?\s*$")
    matches = list(pattern.finditer(text))
    sections = {name: "" for name in SECTION_NAMES}
    for index, match in enumerate(matches):
        name = headings[match.group(1).lower()]
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[name] = text[match.end():end].strip()
    if not sections["Abstract"]:
        sections["Abstract"] = text[: min(len(text), 1400)]
    return sections


def compare_sentences(source, reference, model=None):
    split = lambda text: [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if len(part.split()) >= 5]
    source_sentences = split(source)
    reference_sentences = split(reference)
    if not source_sentences or not reference_sentences:
        return []
    method = "TF-IDF lexical overlap"
    threshold = 0.24
    try:
        if model is not None:
            embeddings = model.encode(source_sentences + reference_sentences, normalize_embeddings=True)
            source_vectors = np.asarray(embeddings[:len(source_sentences)])
            reference_vectors = np.asarray(embeddings[len(source_sentences):])
            scores = np.clip(source_vectors @ reference_vectors.T, 0, 1)
            method = "Sentence-BERT sentence cosine"
            threshold = 0.58
        else:
            matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000).fit_transform(source_sentences + reference_sentences)
            scores = cosine_similarity(matrix[:len(source_sentences)], matrix[len(source_sentences):])
    except Exception as error:
        app.logger.info("Sentence-pair model unavailable; using lexical matches: %s", error)
        try:
            matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000).fit_transform(source_sentences + reference_sentences)
            scores = cosine_similarity(matrix[:len(source_sentences)], matrix[len(source_sentences):])
        except ValueError:
            return []
    pairs = []
    for row_index, sentence_scores in enumerate(scores):
        col_index = int(np.argmax(sentence_scores))
        score = float(sentence_scores[col_index])
        if score >= threshold:
            pairs.append({"submitted": source_sentences[row_index], "reference": reference_sentences[col_index], "similarity": round(score * 100, 1), "method": method})
    return sorted(pairs, key=lambda item: item["similarity"], reverse=True)[:6]


def project_text(project):
    return f"{project.get('title', '')}\nAbstract\n{project.get('abstract', '')}\n{project.get('body_text', '')}".strip()


def document_embedding(model, text):
    chunks = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if len(part.split()) >= 5]
    if not chunks:
        chunks = [text[:1500]]
    vectors = np.asarray(model.encode(chunks, normalize_embeddings=True))
    centroid = vectors.mean(axis=0)
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm else centroid


def section_similarity(source_text, reference_text):
    source_sections = section_texts(source_text)
    reference_sections = section_texts(reference_text)
    sections = {}
    for name in SECTION_NAMES:
        pair = [source_sections[name], reference_sections[name]]
        if not pair[0].strip() or not pair[1].strip():
            sections[name] = 0.0
            continue
        try:
            vectors = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(pair)
            sections[name] = round(float(cosine_similarity(vectors[0:1], vectors[1:])[0, 0] * 100), 1)
        except ValueError:
            sections[name] = 0.0
    return sections


def originality_suggestions(sections, matches, semantic_available, textual, semantic):
    suggestions = []
    strongest = max(sections.items(), key=lambda item: item[1]) if sections else None
    if strongest and strongest[1] >= 15:
        suggestions.append(f"The {strongest[0].lower()} has the strongest section overlap ({strongest[1]:.1f}%). Clarify the project-specific research question, context, data, or design decisions in that section.")
    if matches:
        suggestions.append(f"Review the {len(matches)} matched passage pair(s): cite shared background where appropriate and make the project's own claims and contribution explicit.")
    if semantic_available and semantic >= textual + 10:
        suggestions.append("Meaning similarity is higher than wording similarity. Explain how your planned users, constraints, or evaluation differ from related work.")
    if not suggestions:
        suggestions.append("State a specific research question, target context, and measurable evaluation plan so faculty can distinguish the project’s contribution.")
    return suggestions


def eligible_candidates(connection, exclude_id=None, role="student", user_id=None):
    conditions = ["body_text != ''"]
    params = []
    if exclude_id is not None:
        conditions.append("id != ?")
        params.append(exclude_id)
    if role != "faculty":
        conditions.append("(is_sample=1 OR review_status IN ('Approved','Reviewed') OR user_id=?)")
        params.append(user_id)
    return [dict(row) for row in connection.execute(f"SELECT * FROM projects WHERE {' AND '.join(conditions)}", params).fetchall()]


def analysis_payload(result, project, user):
    reference = result.get("reference")
    can_view_reference_text = bool(
        reference and (user["role"] == "faculty" or reference["is_sample"] or reference["user_id"] == user["id"])
    )
    matches = result.get("matches", [])
    if matches and not can_view_reference_text:
        matches = [{**match, "reference": None, "reference_restricted": True} for match in matches]
    similar_projects = []
    for item in result.get("similar_projects", []):
        candidate = item["reference"]
        similar_projects.append({
            "reference": project_dict(candidate),
            "textual_similarity": item["textual_similarity"],
            "semantic_similarity": item["semantic_similarity"],
            "semantic_available": item["semantic_available"],
            "overall_similarity": item["overall_similarity"],
            "sections": item["sections"],
            "matching_count": item["matching_count"],
        })
    return {
        "id": result.get("id"),
        "project": project_dict(project) if project else None,
        "reference": project_dict(reference) if reference else None,
        "textual_similarity": result["textual_similarity"],
        "semantic_similarity": result["semantic_similarity"],
        "semantic_available": result["semantic_available"],
        "overall_similarity": result["overall_similarity"],
        "semantic_method": result["semantic_method"],
        "match_method": result.get("match_method", "TF-IDF lexical overlap"),
        "sections": result.get("sections", {}),
        "matches": matches,
        "similar_projects": similar_projects,
        "suggestions": result.get("suggestions", []),
        "analysis_date": result.get("analysis_date"),
    }


def analyze_text(text, candidates):
    candidates = [candidate for candidate in candidates if project_text(candidate)]
    if not candidates:
        return {"textual_similarity": 0.0, "semantic_similarity": None, "semantic_available": False, "overall_similarity": 0.0, "semantic_method": "TF-IDF fallback (Sentence-BERT unavailable)", "reference": None, "sections": {}, "matches": [], "similar_projects": [], "suggestions": ["No eligible repository projects were available to compare. Results do not establish originality."]}
    documents = [text] + [project_text(candidate) for candidate in candidates]
    cleaned = [preprocess(document) for document in documents]
    matrix = TfidfVectorizer(ngram_range=(1, 2), max_features=12000, min_df=1).fit_transform(cleaned)
    text_scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    model = get_sbert()
    semantic_available = False
    if model is not None:
        try:
            embeddings = np.vstack([document_embedding(model, document) for document in documents])
            semantic_scores = embeddings[1:] @ embeddings[0]
            semantic_scores = np.clip(semantic_scores, 0, 1)
            semantic_available = True
            method = "Sentence-BERT (all-MiniLM-L6-v2)"
        except Exception as error:
            app.logger.info("Sentence-BERT inference failed; using TF-IDF fallback: %s", error)
            semantic_scores = None
            method = "TF-IDF fallback (Sentence-BERT unavailable)"
    else:
        semantic_scores = None
        method = "TF-IDF fallback (Sentence-BERT unavailable)"
    overall_scores = text_scores * 0.6 + semantic_scores * 0.4 if semantic_available else text_scores
    ranked_indices = np.argsort(overall_scores)[::-1]
    ranked = []
    for index in ranked_indices:
        candidate = candidates[int(index)]
        candidate_text = project_text(candidate)
        sections = section_similarity(text, candidate_text)
        ranked.append({
            "reference": candidate,
            "textual_similarity": round(float(text_scores[index]) * 100, 1),
            "semantic_similarity": round(float(semantic_scores[index]) * 100, 1) if semantic_available else None,
            "semantic_available": semantic_available,
            "overall_similarity": round(float(overall_scores[index]) * 100, 1),
            "sections": sections,
        })
    best = ranked[0]
    matches = compare_sentences(text, project_text(best["reference"]), model if semantic_available else None)
    best.update({"matches": matches, "match_method": matches[0]["method"] if matches else ("Sentence-BERT sentence cosine" if semantic_available else "TF-IDF lexical overlap")})
    suggestions = originality_suggestions(best["sections"], matches, semantic_available, best["textual_similarity"], best["semantic_similarity"] or 0)
    similar_projects = [{
        "reference": item["reference"],
        "reference_project_id": item["reference"]["id"],
        "textual_similarity": item["textual_similarity"],
        "semantic_similarity": item["semantic_similarity"],
        "semantic_available": semantic_available,
        "overall_similarity": item["overall_similarity"],
        "sections": item["sections"],
        "matching_count": len(matches) if item is best else 0,
    } for item in ranked]
    return {
        "textual_similarity": best["textual_similarity"],
        "semantic_similarity": best["semantic_similarity"],
        "semantic_available": semantic_available,
        "overall_similarity": best["overall_similarity"],
        "semantic_method": method,
        "reference": best["reference"],
        "sections": best["sections"],
        "matches": matches,
        "match_method": best["match_method"],
        "similar_projects": similar_projects,
        "suggestions": suggestions,
    }


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "ProjectGuard AI API"})


@app.post("/api/register")
def register():
    data = request.get_json(silent=True) or {}
    name, email = str(data.get("name", "")).strip(), str(data.get("email", "")).strip().lower()
    password, department = str(data.get("password", "")), str(data.get("department", "Computer Science")).strip()
    if len(name) < 2 or "@" not in email or len(password) < 8:
        return jsonify({"error": "Enter a name, valid email, and password with at least 8 characters."}), 400
    try:
        with connect_db() as connection:
            cursor = connection.execute(
                "INSERT INTO users(name,email,password_hash,role,department) VALUES(?,?,?,?,?)",
                (name, email, generate_password_hash(password), "student", department or "Computer Science"),
            )
            token = uuid.uuid4().hex
            connection.execute("INSERT INTO sessions(token,user_id,created_at) VALUES(?,?,?)", (token, cursor.lastrowid, utc_now()))
            user = connection.execute("SELECT id,name,email,role,department FROM users WHERE id=?", (cursor.lastrowid,)).fetchone()
            connection.commit()
        return jsonify({"token": token, "user": dict(user)}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "An account with that email already exists."}), 409


@app.post("/api/login")
def login():
    data = request.get_json(silent=True) or {}
    email, password = str(data.get("email", "")).strip().lower(), str(data.get("password", ""))
    with connect_db() as connection:
        user = connection.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if not user or not check_password_hash(user["password_hash"], password):
            return jsonify({"error": "Email or password is incorrect."}), 401
        token = uuid.uuid4().hex
        connection.execute("INSERT INTO sessions(token,user_id,created_at) VALUES(?,?,?)", (token, user["id"], utc_now()))
        connection.commit()
        return jsonify({"token": token, "user": {key: user[key] for key in ("id", "name", "email", "role", "department")}})


@app.post("/api/logout")
@auth_required()
def logout():
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    with connect_db() as connection:
        connection.execute("DELETE FROM sessions WHERE token=?", (token,))
        connection.commit()
    return jsonify({"ok": True})


@app.get("/api/me")
@auth_required()
def me():
    user = request.user
    return jsonify({key: user[key] for key in ("id", "name", "email", "role", "department")})


@app.post("/api/projects/upload")
@auth_required("student")
def upload_project():
    file = request.files.get("file")
    title = request.form.get("title", "").strip()
    abstract = request.form.get("abstract", "").strip()
    if not file or not file.filename or not title or not abstract:
        return jsonify({"error": "A document, project title, and abstract are required."}), 400
    filename = secure_filename(file.filename)
    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Upload a PDF or DOCX document."}), 415
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}_{filename}"
    target = UPLOAD_DIR / stored_name
    file.save(target)
    try:
        text = extract_text(target)
        if len(text.split()) < 20:
            raise ValueError("The document contains too little extractable text. Scanned PDFs need OCR before analysis.")
    except Exception as error:
        target.unlink(missing_ok=True)
        return jsonify({"error": str(error) or "Could not read the uploaded document."}), 400
    with connect_db() as connection:
        cursor = connection.execute(
            """INSERT INTO projects(user_id,title,abstract,filename,file_path,department,academic_year,category,team_members,body_text,upload_date)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (request.user["id"], title, abstract, filename, str(target), request.form.get("department", request.user["department"]),
             request.form.get("academic_year", "2025–26"), request.form.get("category", "Research"), request.form.get("team_members", ""), text, utc_now()),
        )
        project = get_project(connection, cursor.lastrowid)
        connection.commit()
    return jsonify({"project": project_dict(project), "extracted_words": len(text.split())}), 201


@app.get("/api/projects")
@auth_required()
def projects():
    with connect_db() as connection:
        if request.user["role"] == "faculty":
            rows = connection.execute("SELECT p.*,u.name AS student_name FROM projects p JOIN users u ON u.id=p.user_id ORDER BY p.upload_date DESC").fetchall()
        else:
            rows = connection.execute("SELECT * FROM projects WHERE user_id=? AND is_sample=0 ORDER BY upload_date DESC", (request.user["id"],)).fetchall()
        result = []
        for row in rows:
            item = project_dict(row)
            latest = connection.execute("SELECT overall_similarity FROM similarity_results WHERE project_id=? ORDER BY analysis_date DESC LIMIT 1", (row["id"],)).fetchone()
            item["overall_similarity"] = latest["overall_similarity"] if latest else None
            result.append(item)
    return jsonify({"projects": result})


@app.get("/api/projects/<int:project_id>")
@auth_required()
def project_detail(project_id):
    with connect_db() as connection:
        project = get_project(connection, project_id)
        if not project or (request.user["role"] != "faculty" and project["user_id"] != request.user["id"] and not project["is_sample"]):
            return jsonify({"error": "Project not found."}), 404
        return jsonify({"project": project_dict(project)})


@app.post("/api/ideas/check")
@auth_required("student")
def idea_check():
    data = request.get_json(silent=True) or {}
    title = str(data.get("title", "")).strip()
    abstract = str(data.get("abstract", "")).strip()
    if len(title) < 3 or len(title) > 140 or len(abstract) < 30 or len(abstract) > 1200:
        return jsonify({"error": "Enter a project title and an abstract between 30 and 1,200 characters."}), 400
    with connect_db() as connection:
        candidates = eligible_candidates(connection, role="student", user_id=request.user["id"])
        result = analyze_text(f"{title}\nAbstract\n{abstract}", candidates)
    return jsonify({"result": analysis_payload(result, None, request.user)})


@app.get("/api/repository")
@auth_required()
def repository():
    department = request.args.get("department", "").strip()
    year = request.args.get("academic_year", "").strip()
    query = request.args.get("q", "").strip()
    sql = "SELECT p.*,u.name AS student_name FROM projects p JOIN users u ON u.id=p.user_id WHERE (p.is_sample=1 OR p.review_status IN ('Approved','Reviewed'))"
    params = []
    if department:
        sql += " AND p.department=?"; params.append(department)
    if year:
        sql += " AND p.academic_year=?"; params.append(year)
    if query:
        sql += " AND (p.title LIKE ? OR p.abstract LIKE ?)"; params.extend([f"%{query}%", f"%{query}%"])
    sql += " ORDER BY p.upload_date DESC"
    with connect_db() as connection:
        rows = connection.execute(sql, params).fetchall()
    return jsonify({"projects": [project_dict(row) for row in rows]})


@app.post("/api/analyze/<int:project_id>")
@auth_required()
def analyze(project_id):
    with connect_db() as connection:
        project = get_project(connection, project_id)
        if not project or (request.user["role"] != "faculty" and project["user_id"] != request.user["id"]):
            return jsonify({"error": "Project not found or not available for analysis."}), 404
        candidates = eligible_candidates(connection, project_id, request.user["role"], request.user["id"])
        result = analyze_text(project_text(dict(project)), candidates)
        ref = result["reference"]
        stored_similar_projects = [{
            "reference_project_id": item["reference"]["id"],
            "textual_similarity": item["textual_similarity"],
            "semantic_similarity": item["semantic_similarity"],
            "semantic_available": item["semantic_available"],
            "overall_similarity": item["overall_similarity"],
            "sections": item["sections"],
            "matching_count": item["matching_count"],
        } for item in result["similar_projects"][:10]]
        semantic_storage = result["semantic_similarity"] if result["semantic_available"] else result["textual_similarity"]
        analysis_date = utc_now()
        cursor = connection.execute(
            """INSERT INTO similarity_results(project_id,reference_project_id,textual_similarity,semantic_similarity,semantic_available,overall_similarity,semantic_method,section_results,matching_content,similar_projects,suggestion_results,analysis_date)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (project_id, ref["id"] if ref else None, result["textual_similarity"], semantic_storage, int(result["semantic_available"]), result["overall_similarity"], result["semantic_method"], json.dumps(result["sections"]), json.dumps(result["matches"]), json.dumps(stored_similar_projects), json.dumps(result["suggestions"]), analysis_date),
        )
        connection.execute("UPDATE projects SET analysis_status='Analyzed' WHERE id=?", (project_id,))
        connection.commit()
        result["id"] = cursor.lastrowid
        result["analysis_date"] = analysis_date
    return jsonify({"result": analysis_payload(result, project, request.user)})


@app.get("/api/results/<int:project_id>")
@auth_required()
def results(project_id):
    with connect_db() as connection:
        project = get_project(connection, project_id)
        if not project or (request.user["role"] != "faculty" and project["user_id"] != request.user["id"] and not project["is_sample"]):
            return jsonify({"error": "Report not found."}), 404
        result = connection.execute("SELECT * FROM similarity_results WHERE project_id=? ORDER BY analysis_date DESC LIMIT 1", (project_id,)).fetchone()
        if not result:
            return jsonify({"error": "This project has not been analyzed yet."}), 404
        reference = get_project(connection, result["reference_project_id"]) if result["reference_project_id"] else None
        stored_similar = json.loads(result["similar_projects"] or "[]")
        similar_projects = []
        for item in stored_similar:
            candidate = get_project(connection, item["reference_project_id"])
            if candidate:
                similar_projects.append({**item, "reference": dict(candidate)})
        data = {
            "id": result["id"],
            "reference": dict(reference) if reference else None,
            "textual_similarity": result["textual_similarity"],
            "semantic_similarity": result["semantic_similarity"] if result["semantic_available"] else None,
            "semantic_available": bool(result["semantic_available"]),
            "overall_similarity": result["overall_similarity"],
            "semantic_method": result["semantic_method"],
            "match_method": "Sentence-BERT sentence cosine" if result["semantic_available"] else "TF-IDF lexical overlap",
            "sections": json.loads(result["section_results"]),
            "matches": json.loads(result["matching_content"]),
            "similar_projects": similar_projects,
            "suggestions": json.loads(result["suggestion_results"] or "[]"),
            "analysis_date": result["analysis_date"],
        }
        return jsonify({"result": analysis_payload(data, project, request.user)})


@app.get("/api/faculty/dashboard")
@auth_required("faculty")
def faculty_dashboard():
    with connect_db() as connection:
        summary = connection.execute("""SELECT COUNT(*) AS submissions,
            SUM(CASE WHEN analysis_status='Analyzed' THEN 1 ELSE 0 END) AS analyzed,
            SUM(CASE WHEN analysis_status='Analyzed' AND review_status='Awaiting review' THEN 1 ELSE 0 END) AS needs_review
            FROM projects WHERE is_sample=0""").fetchone()
        average = connection.execute("SELECT AVG(overall_similarity) AS score FROM similarity_results r JOIN projects p ON p.id=r.project_id WHERE p.is_sample=0").fetchone()["score"]
        departments = connection.execute("SELECT department,COUNT(*) AS submissions FROM projects WHERE is_sample=0 GROUP BY department ORDER BY submissions DESC").fetchall()
        rows = connection.execute("SELECT p.*,u.name AS student_name FROM projects p JOIN users u ON u.id=p.user_id WHERE p.is_sample=0 ORDER BY p.upload_date DESC").fetchall()
        all_projects = []
        for row in rows:
            item = project_dict(row)
            latest = connection.execute("SELECT overall_similarity FROM similarity_results WHERE project_id=? ORDER BY analysis_date DESC LIMIT 1", (row["id"],)).fetchone()
            item["overall_similarity"] = latest["overall_similarity"] if latest else None
            all_projects.append(item)
    return jsonify({"summary": {**dict(summary), "average_similarity": round(average or 0, 1)}, "departments": [dict(row) for row in departments], "projects": all_projects})


@app.post("/api/faculty/compare")
@auth_required("faculty")
def faculty_compare():
    data = request.get_json(silent=True) or {}
    try:
        project_id = int(data.get("project_id"))
        reference_project_id = int(data.get("reference_project_id"))
    except (TypeError, ValueError):
        return jsonify({"error": "Choose two valid projects to compare."}), 400
    if project_id == reference_project_id:
        return jsonify({"error": "Choose two different projects."}), 400
    with connect_db() as connection:
        project = get_project(connection, project_id)
        reference = get_project(connection, reference_project_id)
        if not project or not reference:
            return jsonify({"error": "One or both projects could not be found."}), 404
        result = analyze_text(project_text(dict(project)), [dict(reference)])
        comparison = {
            "project": project_dict(project),
            "reference": project_dict(reference),
            "textual_similarity": result["textual_similarity"],
            "semantic_similarity": result["semantic_similarity"],
            "semantic_available": result["semantic_available"],
            "overall_similarity": result["overall_similarity"],
            "semantic_method": result["semantic_method"],
            "sections": result["sections"],
            "matches": result["matches"],
            "suggestions": result["suggestions"],
            "analysis_date": utc_now(),
        }
    return jsonify({"comparison": comparison})


@app.post("/api/faculty/review/<int:project_id>")
@auth_required("faculty")
def faculty_review(project_id):
    data = request.get_json(silent=True) or {}
    status = str(data.get("review_status", "")).strip()
    comments = str(data.get("comments", "")).strip()
    if status not in {"Reviewed", "Needs Revision", "Approved"}:
        return jsonify({"error": "Choose Reviewed, Needs Revision, or Approved."}), 400
    with connect_db() as connection:
        project = get_project(connection, project_id)
        if not project or project["is_sample"]:
            return jsonify({"error": "Submission not found."}), 404
        connection.execute("INSERT INTO faculty_reviews(project_id,faculty_id,review_status,comments,review_date) VALUES(?,?,?,?,?)", (project_id, request.user["id"], status, comments, utc_now()))
        connection.execute("UPDATE projects SET review_status=? WHERE id=?", (status, project_id))
        connection.commit()
    return jsonify({"ok": True, "review_status": status})


@app.get("/api/reports/<int:project_id>/pdf")
@auth_required()
def report_pdf(project_id):
    with connect_db() as connection:
        project = get_project(connection, project_id)
        result = connection.execute("SELECT * FROM similarity_results WHERE project_id=? ORDER BY analysis_date DESC LIMIT 1", (project_id,)).fetchone()
        if not project or not result or (request.user["role"] != "faculty" and project["user_id"] != request.user["id"] and not project["is_sample"]):
            return jsonify({"error": "A report is not available for this project."}), 404
        reference = get_project(connection, result["reference_project_id"]) if result["reference_project_id"] else None
        sections = json.loads(result["section_results"])
        matches = json.loads(result["matching_content"])
        reference_allowed = bool(reference and (request.user["role"] == "faculty" or reference["is_sample"] or reference["user_id"] == request.user["id"]))
        if matches and not reference_allowed:
            matches = [{**match, "reference": None} for match in matches]
        similar_projects = json.loads(result["similar_projects"] or "[]")
        ranked_lines = []
        for item in similar_projects[:10]:
            candidate = get_project(connection, item["reference_project_id"])
            if candidate:
                ranked_lines.append(f"{candidate['title'][:72]} — {item['overall_similarity']:.1f}% overall, {item['textual_similarity']:.1f}% textual")
        suggestions = json.loads(result["suggestion_results"] or "[]")
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / f"projectguard-report-{project_id}.pdf"
        pdf = canvas.Canvas(str(path), pagesize=letter)
        width, height = letter
        y = height - 52
        pdf.setTitle(f"ProjectGuard similarity report: {project['title']}")
        pdf.setFillColorRGB(0.08, 0.16, 0.30); pdf.setFont("Helvetica-Bold", 18); pdf.drawString(48, y, "ProjectGuard AI | Similarity report")
        y -= 32; pdf.setFillColorRGB(0.12, 0.18, 0.27); pdf.setFont("Helvetica-Bold", 12); pdf.drawString(48, y, project["title"][:85])
        y -= 24; pdf.setFont("Helvetica", 10)
        semantic_line = f"Semantic similarity: {result['semantic_similarity']:.1f}%" if result["semantic_available"] else "Semantic analysis unavailable; TF-IDF fallback used"
        for line in [f"Overall similarity: {result['overall_similarity']:.1f}%", f"Textual similarity: {result['textual_similarity']:.1f}%", semantic_line, f"Scoring method: {result['semantic_method']}", f"Closest repository match: {reference['title'] if reference else 'None found'}"]:
            pdf.drawString(48, y, line); y -= 17
        if ranked_lines:
            y -= 4; pdf.setFont("Helvetica-Bold", 11); pdf.drawString(48, y, "Other ranked repository matches"); y -= 18
            pdf.setFont("Helvetica", 9)
            for line in ranked_lines:
                if y < 72:
                    pdf.showPage(); y = height - 52; pdf.setFont("Helvetica", 9)
                pdf.drawString(55, y, line); y -= 14
        y -= 6; pdf.setFont("Helvetica-Bold", 11); pdf.drawString(48, y, "Section comparison"); y -= 18
        pdf.setFont("Helvetica", 10)
        for name, score in sections.items():
            pdf.drawString(55, y, f"{name}: {score:.1f}%"); y -= 15
        y -= 6; pdf.setFont("Helvetica-Bold", 11); pdf.drawString(48, y, "Matching passages"); y -= 18
        pdf.setFont("Helvetica", 9)
        for match in matches[:4]:
            passage_pairs = [("Submitted", match["submitted"])]
            if match.get("reference"):
                passage_pairs.append(("Reference", match["reference"]))
            else:
                passage_pairs.append(("Reference", "Passage withheld: source document access is restricted."))
            for label, text in passage_pairs:
                for line in [text[index:index + 105] for index in range(0, min(len(text), 210), 105)]:
                    if y < 72:
                        pdf.showPage(); y = height - 52; pdf.setFont("Helvetica", 9)
                    pdf.drawString(55, y, f"{label}: {line}"); y -= 13
            y -= 4
        if suggestions:
            if y < 115:
                pdf.showPage(); y = height - 52
            y -= 6; pdf.setFont("Helvetica-Bold", 11); pdf.drawString(48, y, "Originality suggestions (not guarantees)"); y -= 18
            pdf.setFont("Helvetica", 9)
            for suggestion in suggestions:
                line = suggestion[:110]
                if y < 72:
                    pdf.showPage(); y = height - 52; pdf.setFont("Helvetica", 9)
                pdf.drawString(55, y, f"• {line}"); y -= 14
        if y < 85:
            pdf.showPage(); y = height - 52
        pdf.setFillColorRGB(0.3, 0.34, 0.4); pdf.setFont("Helvetica-Oblique", 8)
        disclaimer = "Similarity scores indicate textual or semantic overlap and do not independently establish plagiarism. Final academic decisions require faculty review. Review thresholds are configurable indicators, not universal standards."
        pdf.drawString(48, 42, disclaimer[:130]); pdf.drawString(48, 30, disclaimer[130:])
        pdf.save()
    return send_file(path, as_attachment=True, download_name=path.name, mimetype="application/pdf")


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "The file is larger than the 16 MB upload limit."}), 413


@app.errorhandler(500)
def internal_error(_error):
    return jsonify({"error": "The server could not complete the request."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=True)
