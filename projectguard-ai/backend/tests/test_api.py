import io
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

TEST_DIR = tempfile.TemporaryDirectory(prefix="projectguard-tests-")
os.environ["PROJECTGUARD_DB"] = str(Path(TEST_DIR.name) / "test.sqlite3")
os.environ["PROJECTGUARD_UPLOADS"] = str(Path(TEST_DIR.name) / "uploads")
os.environ["PROJECTGUARD_REPORTS"] = str(Path(TEST_DIR.name) / "reports")
os.environ["PROJECTGUARD_DISABLE_SBERT"] = "1"

import pymupdf as fitz
from docx import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as app_module
from app import analyze_text, app, analysis_payload, load_env_file, seed_database


class ProjectGuardApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()
        student = cls.client.post("/api/login", json={"email": "student@projectguard.demo", "password": "Student123!"})
        faculty = cls.client.post("/api/login", json={"email": "faculty@projectguard.demo", "password": "Faculty123!"})
        cls.student_headers = {"Authorization": f"Bearer {student.get_json()['token']}"}
        cls.faculty_headers = {"Authorization": f"Bearer {faculty.get_json()['token']}"}

    def test_seeded_repository_and_role_access(self):
        repository = self.client.get("/api/repository", headers=self.student_headers)
        self.assertEqual(repository.status_code, 200)
        self.assertEqual(len(repository.get_json()["projects"]), 5)
        self.assertTrue(all(item["is_sample"] for item in repository.get_json()["projects"]))
        self.assertEqual(self.client.get("/api/faculty/dashboard", headers=self.student_headers).status_code, 403)

    def test_dotenv_file_loads_quoted_values_without_overriding_environment(self):
        key = "PROJECTGUARD_TEST_ENV_VALUE"
        previous = os.environ.get(key)
        os.environ[key] = "environment-value"
        env_file = Path(TEST_DIR.name) / "test.env"
        env_file.write_text(f'{key}="file-value"\nPROJECTGUARD_UNUSED=value\n', encoding="utf-8")
        try:
            load_env_file(env_file)
            self.assertEqual(os.environ[key], "environment-value")
            os.environ.pop(key)
            load_env_file(env_file)
            self.assertEqual(os.environ[key], "file-value")
        finally:
            os.environ.pop(key, None)
            if previous is not None:
                os.environ[key] = previous

    def test_configured_faculty_credentials_can_login(self):
        credentials = {
            "PROJECTGUARD_FACULTY_NAME": "Morgan Faculty",
            "PROJECTGUARD_FACULTY_EMAIL": "morgan.faculty@example.edu",
            "PROJECTGUARD_FACULTY_PASSWORD": "PrivateFacultyPass123!",
            "PROJECTGUARD_FACULTY_DEPARTMENT": "Data Science",
        }
        with patch.dict(os.environ, credentials):
            with app_module.connect_db() as connection:
                seed_database(connection)
        response = self.client.post(
            "/api/login",
            json={"email": credentials["PROJECTGUARD_FACULTY_EMAIL"], "password": credentials["PROJECTGUARD_FACULTY_PASSWORD"]},
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json()["user"]["role"], "faculty")
        self.assertEqual(response.get_json()["user"]["department"], "Data Science")
        with app_module.connect_db() as connection:
            connection.execute("DELETE FROM sessions WHERE user_id=(SELECT id FROM users WHERE email=?)", (credentials["PROJECTGUARD_FACULTY_EMAIL"],))
            connection.execute("DELETE FROM users WHERE email=?", (credentials["PROJECTGUARD_FACULTY_EMAIL"],))
            connection.commit()

    def test_docx_upload_analysis_review_and_pdf_report(self):
        doc = Document()
        doc.add_paragraph("Abstract")
        doc.add_paragraph("A campus energy monitoring system uses occupancy sensors and smart meters to forecast classroom demand. The platform recommends efficient lighting and HVAC schedules while preserving student comfort.")
        doc.add_paragraph("Introduction")
        doc.add_paragraph("The system collects occupancy, temperature, and power readings from campus rooms. A forecasting model identifies peak demand and recommends schedules that reduce energy use without affecting student comfort.")
        doc.add_paragraph("Objectives")
        doc.add_paragraph("Improve campus energy use through reliable and accessible data-driven recommendations for university buildings and learning spaces.")
        doc.add_paragraph("Methodology")
        doc.add_paragraph("Collect sensor readings, preprocess the records, train an energy forecasting model, and review recommendations with campus users.")
        doc.add_paragraph("Proposed System")
        doc.add_paragraph("A web dashboard presents energy recommendations and supports transparent human review by facility staff.")
        doc.add_paragraph("Conclusion")
        doc.add_paragraph("The prototype demonstrates a practical workflow for reducing campus energy demand and identifies future evaluation needs.")
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        response = self.client.post(
            "/api/projects/upload",
            headers=self.student_headers,
            data={
                "title": "Campus energy demand study",
                "abstract": "A sensor-based energy forecasting project for campus classrooms.",
                "department": "Computer Science",
                "academic_year": "2025–26",
                "category": "Sustainability",
                "team_members": "Jordan Lee",
                "file": (buffer, "energy-study.docx"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 201, response.get_json())
        project_id = response.get_json()["project"]["id"]
        analysis = self.client.post(f"/api/analyze/{project_id}", headers=self.student_headers)
        self.assertEqual(analysis.status_code, 200, analysis.get_json())
        result = analysis.get_json()["result"]
        self.assertEqual(result["semantic_method"], "TF-IDF fallback (Sentence-BERT unavailable)")
        self.assertFalse(result["semantic_available"])
        self.assertIsNone(result["semantic_similarity"])
        self.assertGreater(result["textual_similarity"], 0)
        self.assertGreaterEqual(result["overall_similarity"], 0)
        self.assertIn("Methodology", result["sections"])
        self.assertTrue(result["reference"]["is_sample"])
        self.assertEqual(len(result["similar_projects"]), 5)
        self.assertEqual(
            [item["overall_similarity"] for item in result["similar_projects"]],
            sorted((item["overall_similarity"] for item in result["similar_projects"]), reverse=True),
        )
        self.assertTrue(result["suggestions"])

        reference_id = result["reference"]["id"]
        comparison = self.client.post(
            "/api/faculty/compare",
            headers=self.faculty_headers,
            json={"project_id": project_id, "reference_project_id": reference_id},
        )
        self.assertEqual(comparison.status_code, 200, comparison.get_json())
        self.assertEqual(comparison.get_json()["comparison"]["reference"]["id"], reference_id)
        self.assertGreater(comparison.get_json()["comparison"]["textual_similarity"], 0)

        report = self.client.get(f"/api/results/{project_id}", headers=self.student_headers)
        self.assertEqual(report.status_code, 200)
        self.assertEqual(report.get_json()["result"]["project"]["id"], project_id)
        review = self.client.post(
            f"/api/faculty/review/{project_id}",
            headers=self.faculty_headers,
            json={"review_status": "Needs Revision", "comments": "Please cite the related campus energy work."},
        )
        self.assertEqual(review.status_code, 200)
        pdf = self.client.get(f"/api/reports/{project_id}/pdf", headers=self.faculty_headers)
        self.assertEqual(pdf.status_code, 200)
        self.assertEqual(pdf.mimetype, "application/pdf")
        self.assertTrue(pdf.data.startswith(b"%PDF"))
        pdf.close()

        other_account = self.client.post(
            "/api/register",
            json={"name": "Taylor Morgan", "email": "taylor@example.edu", "password": "SecurePass123!"},
        ).get_json()
        other_headers = {"Authorization": f"Bearer {other_account['token']}"}
        self.assertEqual(self.client.get(f"/api/projects/{project_id}", headers=other_headers).status_code, 404)
        self.assertEqual(self.client.get(f"/api/results/{project_id}", headers=other_headers).status_code, 404)
        self.assertEqual(self.client.get(f"/api/reports/{project_id}/pdf", headers=other_headers).status_code, 404)

    def test_project_idea_check_compares_without_persisting(self):
        before = self.client.get("/api/projects", headers=self.student_headers).get_json()["projects"]
        response = self.client.post(
            "/api/ideas/check",
            headers=self.student_headers,
            json={
                "title": "Campus classroom energy forecasting",
                "abstract": "A sensor network forecasts energy use in university classrooms and proposes schedules for heating, ventilation, and lighting based on occupancy and temperature data.",
            },
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        result = response.get_json()["result"]
        self.assertEqual(len(result["similar_projects"]), len(before) + 5)
        self.assertEqual(result["semantic_method"], "TF-IDF fallback (Sentence-BERT unavailable)")
        self.assertTrue(result["suggestions"])
        after = self.client.get("/api/projects", headers=self.student_headers).get_json()["projects"]
        self.assertEqual(len(after), len(before))

    def test_sentence_bert_scoring_path_reuses_injected_model(self):
        class FakeSentenceTransformer:
            calls = 0

            def encode(self, texts, normalize_embeddings=True):
                self.calls += 1
                vectors = []
                for text in texts:
                    lower = text.lower()
                    if "energy" in lower:
                        vectors.append([1.0, 0.0])
                    elif "library" in lower:
                        vectors.append([0.0, 1.0])
                    else:
                        vectors.append([0.8, 0.6])
                return np.asarray(vectors)

        previous_model = app_module.SBERT_MODEL
        previous_attempted = app_module.SBERT_ATTEMPTED
        model = FakeSentenceTransformer()
        app_module.SBERT_MODEL = model
        app_module.SBERT_ATTEMPTED = True
        try:
            result = analyze_text(
                "Campus energy forecasting uses smart meters and occupancy sensors. The energy model recommends efficient classroom schedules.",
                [
                    {"id": 1, "title": "Energy planning", "abstract": "Campus energy management", "body_text": "Energy forecasting uses smart meters and occupancy sensors."},
                    {"id": 2, "title": "Library seat finder", "abstract": "Find an open library place", "body_text": "The library service tracks available study seats."},
                ],
            )
        finally:
            app_module.SBERT_MODEL = previous_model
            app_module.SBERT_ATTEMPTED = previous_attempted
        self.assertTrue(result["semantic_available"])
        self.assertEqual(result["semantic_method"], "Sentence-BERT (all-MiniLM-L6-v2)")
        self.assertEqual(result["semantic_similarity"], 100.0)
        self.assertEqual(result["matches"][0]["method"], "Sentence-BERT sentence cosine")
        self.assertGreaterEqual(model.calls, 2)

    def test_sentence_bert_loader_caches_model_instance(self):
        loaded_names = []
        fake_model = object()

        def load_model(model_name):
            loaded_names.append(model_name)
            return fake_model

        previous_model = app_module.SBERT_MODEL
        previous_attempted = app_module.SBERT_ATTEMPTED
        app_module.SBERT_MODEL = None
        app_module.SBERT_ATTEMPTED = False
        fake_module = SimpleNamespace(SentenceTransformer=load_model)
        try:
            with patch.dict(sys.modules, {"sentence_transformers": fake_module}), patch.dict(
                os.environ,
                {"PROJECTGUARD_DISABLE_SBERT": "0", "PROJECTGUARD_SBERT_MODEL": "all-MiniLM-L6-v2"},
            ):
                first = app_module.get_sbert()
                second = app_module.get_sbert()
        finally:
            app_module.SBERT_MODEL = previous_model
            app_module.SBERT_ATTEMPTED = previous_attempted
        self.assertIs(first, fake_model)
        self.assertIs(second, fake_model)
        self.assertEqual(loaded_names, ["all-MiniLM-L6-v2"])

    @unittest.skipUnless(os.environ.get("PROJECTGUARD_RUN_REAL_SBERT") == "1", "set PROJECTGUARD_RUN_REAL_SBERT=1 to download and test model weights")
    def test_real_sentence_transformer_model_loads_and_encodes(self):
        previous_model = app_module.SBERT_MODEL
        previous_attempted = app_module.SBERT_ATTEMPTED
        previous_disable = os.environ.get("PROJECTGUARD_DISABLE_SBERT")
        app_module.SBERT_MODEL = None
        app_module.SBERT_ATTEMPTED = False
        os.environ["PROJECTGUARD_DISABLE_SBERT"] = "0"
        try:
            model = app_module.get_sbert()
            self.assertIsNotNone(model, "Sentence-BERT failed to load; check the logged fallback reason.")
            embeddings = app_module.document_embedding(
                model,
                "A campus energy forecasting system uses occupancy sensors. It recommends efficient lighting schedules for classrooms.",
            )
            self.assertEqual(embeddings.ndim, 1)
            self.assertAlmostEqual(float(np.linalg.norm(embeddings)), 1.0, places=4)
        finally:
            app_module.SBERT_MODEL = previous_model
            app_module.SBERT_ATTEMPTED = previous_attempted
            if previous_disable is None:
                os.environ.pop("PROJECTGUARD_DISABLE_SBERT", None)
            else:
                os.environ["PROJECTGUARD_DISABLE_SBERT"] = previous_disable

    def test_student_report_redacts_other_students_reference_passages(self):
        private_reference = {
            "id": 91,
            "user_id": 202,
            "is_sample": 0,
            "role": "student",
            "title": "Private submitted project",
        }
        result = {
            "reference": private_reference,
            "textual_similarity": 40.0,
            "semantic_similarity": None,
            "semantic_available": False,
            "overall_similarity": 40.0,
            "semantic_method": "TF-IDF fallback (Sentence-BERT unavailable)",
            "sections": {},
            "matches": [{"submitted": "My sentence", "reference": "Private source sentence", "similarity": 55.0}],
            "similar_projects": [],
            "suggestions": [],
        }
        payload = analysis_payload(result, None, {"id": 101, "role": "student"})
        self.assertIsNone(payload["matches"][0]["reference"])
        self.assertTrue(payload["matches"][0]["reference_restricted"])

    def test_pdf_upload_and_invalid_extension(self):
        pdf_doc = fitz.open()
        page = pdf_doc.new_page()
        page.insert_textbox(
            fitz.Rect(72, 72, 540, 700),
            "A detailed academic project document describing a complete software platform for library seat availability, occupancy sensors, accessible study spaces, and anonymous privacy-aware data management. The system includes several measurable objectives and a complete methodology for testing the service with students and faculty members.",
            fontsize=11,
        )
        payload = pdf_doc.tobytes()
        pdf_doc.close()
        response = self.client.post(
            "/api/projects/upload",
            headers=self.student_headers,
            data={"title": "Library occupancy service", "abstract": "An accessible library seat finder.", "file": (io.BytesIO(payload), "library.pdf")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 201, response.get_json())
        invalid = self.client.post(
            "/api/projects/upload",
            headers=self.student_headers,
            data={"title": "Invalid file", "abstract": "Not a supported document.", "file": (io.BytesIO(b"plain text"), "file.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(invalid.status_code, 415)


if __name__ == "__main__":
    unittest.main()
