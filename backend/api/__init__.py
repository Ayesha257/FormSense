"""HTTP API for the FormSense web app.

A thin layer over the existing backend: every endpoint calls functions that already
exist (FormSense.run, rules.audit, explain.explain_report, explain.pdf_bytes,
summary.checklist, rag.GuidelineRetriever). No checking logic lives here.

Run from the backend folder:

    python -m api                 # API + web app on http://localhost:8000
    python -m api --port 8080 --no-browser
"""

__version__ = "1.0.0"
