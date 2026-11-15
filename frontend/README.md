# FormSense web app

React + TypeScript (Vite) frontend for the FormSense API in `backend/api`.

## Run it (no Node needed)

The built app in `frontend/dist` is committed. The API serves it:

```bash
cd backend
python -m pip install -r requirements.txt   # adds fastapi, uvicorn, python-multipart
python -m api                               # opens http://localhost:8000
```

Options: `python -m api --port 8001 --no-browser`. The OCR models load when the server starts
(the sidebar shows "Reader loading" until they are ready). API docs: http://localhost:8000/docs

## Develop the frontend

Needs Node 18+.

```bash
cd backend && python -m api --no-browser     # terminal 1: API on :8000
cd frontend && npm install && npm run dev    # terminal 2: http://localhost:5173 (proxies /api to :8000)
```

After changing anything in `src/`, rebuild and commit `dist/` so teammates without Node get the update:

```bash
npm run typecheck
npm run build
```

## Pages

| Route | What it does |
| --- | --- |
| `/` | Overview: upload, stats, recent checks, common problems, samples |
| `/check` | Upload a PDF or photo (drag and drop), choose options, or check a sample form |
| `/forms/:id` | Result: problems circled on the straightened form, answers to correct and re-check, checklist, explanation, PDF/JSON report |
| `/history` | Every form checked, with search and filters |
| `/guidelines` | Ask a question; FormSense finds the guideline passage (same retrieval as the issue citations) |

## Code map

```
src/
  lib/api.ts              typed client for every /api route (upload uses XHR for progress)
  lib/hooks.tsx           health polling, async loader, media query
  lib/format.ts           dates, sizes, verdict and status wording
  components/             Dropzone, DocumentViewer (red-pen marks), ReviewPanels, SampleGrid, Toast, ui
  pages/                  Dashboard, Check, Result, History, Guidelines, NotFound
  styles/global.css       design tokens and all styles
```

## Where data goes

Each check is saved under `backend/data/history/<id>/` (original upload, straightened page, readings,
audit, corrections). That folder is gitignored because uploads can contain real people's details.
Set `FORMSENSE_HISTORY_DIR` to store it elsewhere.
