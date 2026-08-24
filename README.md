# Browser Automation Agent

An authenticated browser agent that logs into Gmail with [Camoufox](https://github.com/daijro/camoufox),
extracts recent messages, and turns them into a prioritised digest using a local
LLM via [Ollama](https://ollama.com). Execution is orchestrated with
[LangGraph](https://langchain-ai.github.io/langgraph/) and exposed over a FastAPI service.

Gmail is authenticated once, interactively. The browser profile is persisted per
user, so every later run reuses the session without repeating login or 2FA.

## How it works

```
POST /api/automations/email
        |
        v
  job row (SQLite)  ->  background worker
                            |
                            |  1. launch Camoufox with the user's saved profile
                            |  2. confirm the Gmail session is still valid
                            |  3. extract the 10 most recent messages
                            |  4. LangGraph: summarize -> validate -> retry (max 3)
                            |  5. write outputs/<user_id>/<job_id>.md
                            v
                     job marked completed
```

Job state lives in SQLite, so progress is pollable while the run is in flight and
survives a server restart.

### LangGraph flow

`summarize_emails` -> `validate_digest` -> conditional edge:

- valid digest -> `END`
- invalid, under 3 attempts -> `prepare_retry` -> back to `summarize_emails`
- invalid, out of attempts -> `handle_error` -> `END`

Retries exist for a flaky local model. Deterministic failures (an empty inbox)
are rejected before the graph runs, so they don't burn the retry budget.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Ollama running locally with a model pulled

## Setup

```bash
uv sync

# fetch the Camoufox browser binary (~120 MB)
uv run camoufox fetch

# the summarisation model
ollama pull qwen3:8b
```

Create `backend/.env` with a signing secret of at least 32 characters:

```bash
JWT_SECRET=<paste a long random string>
```

Generate one with:

```bash
uv run python -c "import secrets; print(secrets.token_urlsafe(48))"
```

The app refuses to start without it rather than falling back to a weak default.
A real `JWT_SECRET` environment variable takes precedence over the file.

## Running

```bash
uv run python -m backend.server
```

Serves on `http://127.0.0.1:8000`. Interactive API docs at `/docs`.

There is no auto-reload, so restart after code changes.

## Usage

**1. Create an account and get a token**

```bash
curl -X POST localhost:8000/api/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"yourpassword"}'

curl -X POST localhost:8000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"yourpassword"}'
```

Use the returned `access_token` as `Authorization: Bearer <token>` below.

**2. Connect Gmail (once per user)**

```bash
curl -X POST localhost:8000/api/browser-auth/gmail/connect -H "Authorization: Bearer $TOKEN"
```

A visible browser window opens. Sign in normally, including any 2FA or security
check. The request returns as soon as the session is confirmed and closes the
browser itself — no fixed wait. Set a generous client timeout, since it stays
open until you finish logging in.

Check the stored session at any time:

```bash
curl localhost:8000/api/browser-auth/gmail/status -H "Authorization: Bearer $TOKEN"
```

**3. Run the digest**

```bash
curl -X POST localhost:8000/api/automations/email -H "Authorization: Bearer $TOKEN"
# {"job_id":"job_59dea2599953","status":"queued"}

curl localhost:8000/api/automations/email/job_59dea2599953 -H "Authorization: Bearer $TOKEN"
```

The job progresses through `authenticating` -> `extracting_emails` ->
`summarizing` -> `validating` -> `completed`. Poll the second endpoint for
`status`, `step`, `progress`, and the finished `result`.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/auth/register` | Create an account |
| `POST` | `/api/auth/login` | Exchange credentials for a JWT |
| `GET` | `/api/auth/me` | Current user |
| `GET` | `/api/browser-auth/gmail/status` | Is the stored Gmail session valid? |
| `POST` | `/api/browser-auth/gmail/connect` | Interactive Gmail login |
| `POST` | `/api/automations/email` | Trigger a digest run |
| `GET` | `/api/automations/email/{job_id}` | Job status and result |
| `GET` | `/health` | Liveness |

## Output

Each completed run writes `outputs/<user_id>/<job_id>.md`:

```markdown
# Email Digest

- **Job:** `job_59dea2599953`
- **Generated:** 2026-08-24 18:57 UTC

## Summary

Security alerts from Google, a pending Airtel bill, and several job alerts.

## Needs Attention

### Check the security alert for the new sign-in on Mac OS

`high` | `required`

A new sign-in to your Google Account. If this wasn't you, secure the account.

## Action Items

- [ ] Check the security alert for the new sign-in on Mac OS
```

The same digest is stored as JSON in the job row, so the API returns it without
reading from disk.

## Layout

```
backend/
  agent/       LangGraph state, nodes, graph wiring
  api/routes/  FastAPI routers
  auth/        password hashing, JWT, request dependencies
  browser/     Camoufox lifecycle, Gmail login + extraction
  db/          SQLAlchemy models and session
  jobs/        job manager and the email worker
  llm/         Ollama client and the summarisation prompt
  services/    shared Pydantic models
```

## Where state lives

Two independent stores:

- `browser_agent.db` — accounts and job history. Created in the working
  directory you launch from, so start the server from the repo root.
- `browser_profiles/<user_id>/google/` — the persisted Firefox profile holding
  the Gmail session. Roughly 120 MB per connected account.

Deleting the database loses accounts but leaves Gmail connected. Deleting a
profile does the reverse. Both are gitignored.

## Notes and limitations

- **Local inference speed dominates.** Model calls have no timeout on purpose —
  an 8B model at partial GPU offload runs at roughly 6 tokens/sec, and a cap
  just kills long-but-healthy calls. Thinking mode is disabled (`think=False`)
  because the reasoning preamble cost far more than the digest itself. If the
  digests read as shallow, a smaller model that fits entirely in VRAM with
  thinking enabled is the better trade.
- **One browser per profile.** Concurrent runs for the same user are serialised
  with a lock, since Firefox will not open a profile twice. Different users run
  in parallel.
- **Extraction depends on Gmail's DOM.** Rows that don't parse are skipped
  rather than failing the run, but a Gmail redesign will need new selectors.
- **`create_all` only creates missing tables.** Changing a column means dropping
  `browser_agent.db`; there are no migrations.
- Single-node only — background jobs are asyncio tasks in the API process, so
  running multiple workers would need a real queue.
