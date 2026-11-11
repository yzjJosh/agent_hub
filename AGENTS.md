# Agent Hub - MCP Server for AI Browser Automation & Reasoning

## Project Context

A Python-based MCP (Model Context Protocol) server built on FastMCP. It provides browser automation, vision reasoning (images, PDFs, videos), TTS, and document classification capabilities over SSE transport.

**Tech stack:** Python 3.12+, FastMCP, Playwright/Patchright, Ultralytics (YOLO), Transformers, ONNX Runtime

**Key modules:**
- `browser/` - Chrome browser management, captcha solving, page snapshots
- `reason/` - LLM-powered image/PDF/video description, document classification, rotation detection
- `recognizer/` - Detection components and Playwright-based agent challengers
- `resolver/` - Reverse geocoding resolution
- `resources/` - WebDAV serving and MCP resource exposure of the resources folder
- `tts/` - Local text-to-speech via sherpa/voxcpm

The container runs Xvfb + Fluxbox (virtual display), a managed Chrome browser (cloakbrowser), noVNC for remote desktop, and the main server on container port 80 (host port 8315). The main server serves the MCP server at `/mcp`, the WebDAV endpoint at `/dav`, and an OpenAI-compatible TTS endpoint at `/v1/audio/speech` (`app/tts_server.py`, same engine as the `tts` tool). Model inference runs on the CPU.

**Environment variables** (in `.env`):
- `REASON_BACKEND_URL` - LLM backend for reasoning tasks
- `REASON_THINKING_BACKEND_URL` - LLM backend with thinking capability
- `COMFYUI_BACKEND_URL` - Required synchronous ComfyUI API wrapper URL for the `generate_video`, `generate_image`, and `edit_image` tools (e.g. `http://localhost:3000`)
- `LAYA_SERVER_URL` - Laya page classifier that checks whether an HTML page fetched with curl is a block page, so the resolver falls back to the browser (e.g. `http://localhost:8888`)
- `MODELS_DIR` - Path to ONNX/PT model files (default `/app/models`)
- `RESOURCES_DIR` - Resources folder served by WebDAV and MCP resources (default `/app/data/resources`)
- `DATA_DIR` - Root for run persistence: `records.db` and `runs/` (default `/app/data`)
- `LOCAL_TTS_*` - TTS server URL, voice, and model settings
- `DAV_BASE_URL` - Base URL of this server exposed to MCP clients for file transfer (default `http://localhost`); used by the `send_or_receive_file` tool

## Core File Paths

Where things live at the repo root:

- **`app/`** — production source code: MCP server (`app/server.py`, `app/main.py`), console, tools, and shared libs
- **`scripts/`** — operational/dev scripts (`run_tests.sh`, `agent_cli`, `console_preview`, `console_seed.py`)
- **`models/`** — ONNX/PT model files and downloaded HF models; git-ignored runtime artifacts, bind-mounted into the container at `/app/models`
- **`tests/app/`** — unit + live tests for `app/`, mirroring its structure (`tests/app/lib` ↔ `app/lib`, etc.)
- **`tests/scripts/`** — tests for `scripts/`, mirroring it (`scripts/X` ↔ `tests/scripts/test_X.py`); must run on the host, not in the container (see **Running Tests**)
- **`docs/`** — planning and design docs: plans under `docs/superpowers/plans/`, plus `docs/tech-design/` and `docs/ux-dashboard/`

## File Transmission Pattern

Files move between MCP clients and this server through the WebDAV endpoint at `/dav`:

- **Upload** — client `PUT`s to `/dav/upload/<filename>`; only `upload/` is read-write, everything else is read-only
- **Download** — client `GET`s from `/dav/<path>`, saving to `/tmp/` rather than the current folder
- **Read/list in-session** — the MCP resource list carries one entry per file; `resources:///<path>` shows a file's content; reading a folder (e.g. `resources:///upload/`) lists that folder
- **Retention** — files not used or written for 30 days are auto-deleted by the garbage collector (see `docs/resources/retention.md` for what counts as a use)
- **`tts/`** — output of the `tts` tool: synthesized audio files are saved here
- **`downloads/`** — output of the `web_fetch` tool: binary files that cannot be resolved to text (not image, audio, video, PDF, or text) are copied here and the tool returns a WebDAV download command for them
- **`media_gen/`** — output of the ComfyUI media tools (`generate_video`, `generate_image`, `edit_image`): generated media files are saved here
- **`runs/`** — output of every tracked task run: `runs/<task_id>/log` (captured stdout/stderr) and `runs/<task_id>/result` (full raw result); records in `$DATA_DIR/records.db` are GC'ed after 30 days (`agent_cli tools gc`)
- **Documentation entry point** — the `send_or_receive_file` tool (no arguments) returns these instructions to LLM clients; new tools that accept or produce files should reference this pattern in their docs

## Running Tests

Tests use pytest with asyncio auto mode. Test files mirror the `app/lib/` structure under `tests/app/lib/`.

The suite mixes two kinds of tests:

- **Unit tests** — the large majority (~91% of the ~1,800 test functions); no external dependencies, finish in a few minutes.
- **Live tests** — marked `@pytest.mark.live` (at function, class, or module level); require a reachable LLM backend, real Chrome, or the host Agent Hub server. Extremely slow.

**⚠️ `bash scripts/run_tests.sh` with no filters runs the ENTIRE suite — unit AND live — and takes ~20 minutes. It is NOT a unit-test run.** Use a marker filter to select what you mean:

Runs are **parallel by default** — `scripts/run_tests.sh` injects `-n 4 --dist worksteal`, cutting the unit suite to ~42s (88s serial). Go serial with `-n 0` (per-test debugging), pick another size with `-n 8`, or pass any of `-n`, `--numprocesses`, `--dist` yourself to control parallelism entirely, because your flags then replace the injected default.

**From `/var/ai/agent_hub/` (or a worktree root):**
```bash
# Unit tests only, parallel — the default for day-to-day iteration
bash scripts/run_tests.sh -m "not live"

# Unit tests only, serial — for stepping through a single test
bash scripts/run_tests.sh -m "not live" -n 0

# Unit tests only, with your own worker count and distribution mode
bash scripts/run_tests.sh -m "not live" -n 8 --dist loadgroup

# Live tests only, usually skip not fail when the backend is down
bash scripts/run_tests.sh -m live

# Only the live tests affected by a change — the usual post-change check
bash scripts/run_tests.sh -m live -k test_name

# Specific test file
bash scripts/run_tests.sh tests/app/lib/reason/test_reason.py

# Specific test by name (any kind in the selected scope)
bash scripts/run_tests.sh -k test_name

# Full suite, unit + live (~20 min) — avoid unless explicitly needed
bash scripts/run_tests.sh
```

Notes:

- Quote the marker expression: `-m "not live"`.
- Do not filter on `timeout` — `@pytest.mark.timeout(seconds)` is orthogonal to `live` and is also applied to unit tests; `-m "not live and not timeout"` would wrongly exclude them.
- `scripts/run_tests.sh` starts the `agent_hub` container if it is stopped, detects worktrees automatically, passes arguments through to pytest, mapping repo-root paths into the container, and injects the `-n 4 --dist worksteal` default unless you pass an xdist selector (it says so on stderr and runs serially when the image has no `pytest-xdist`).

Always use `scripts/run_tests.sh` — never invoke `docker exec ... pytest` directly.

**Recommended workflow:** after a change, run the affected unit tests first (a file path, or `-k test_name`, with `-m "not live"` to be safe), then run only the live tests affected by your change (e.g. `bash scripts/run_tests.sh -m live -k test_name`). Avoid running the full suite or the entire `-m live` set unless necessary.

### scripts/agent_cli

`scripts/agent_cli` is a standalone command line client for the live Agent Hub MCP server (subcommands: `tools list|show|call|gc`, `resources list|read|upload|delete|gc`). It defaults to `http://127.0.0.1:8315/mcp`, overridable via the `MCP_URL` environment variable; `resources gc` uses `RESOURCES_DIR` (default `data/resources/`), and `tools gc` sweeps stale tool run records and run dirs inside the Agent Hub container (default 30 days, `--days N` overrides).

**⚠️ Tests in `tests/scripts/` must run in the host environment, not in the docker container** — the live tests execute the script as a subprocess against the host port (8315), which is not reachable from inside the container. Run them directly on the host instead of via `scripts/run_tests.sh`.

## Build and Deploy

**All build and deploy commands MUST run from `/var/ai`, not `/var/ai/agent_hub`.** The Agent Hub service is included in the root `docker-compose.yml` via the `include` directive.

```bash
cd /var/ai

# Build the Agent Hub image
docker compose build agent_hub

# Start the Agent Hub service
docker compose up -d agent_hub

# Deploy code changes (source, tests, .env) — bind-mounted, restart only, no rebuild
docker compose restart agent_hub

# Rebuild and restart — only when the Dockerfile or pyproject.toml dependencies change
docker compose up -d --build agent_hub

# Stop the Agent Hub service
docker compose stop agent_hub

# View logs
docker compose logs -f agent_hub

# Check health
docker compose ps agent_hub
```

The service exposes (host ports; MCP and WebDAV on container port 80):
- Port 8315 - MCP server at `/mcp` and the WebDAV endpoint at `/dav`
- Port 5900 - VNC
- Port 6080 - noVNC web interface
- Port 9222 - Chrome DevTools Protocol

Source code, tests, `.env`, `models/`, `data/`, `.worktrees/` (at `/worktrees` in the container), and `log/` are bind-mounted for live development, so code changes deploy via `docker compose restart agent_hub` — no rebuild; a rebuild is required only when the Dockerfile or the package dependencies (`pyproject.toml`) change. The `.env` file must be present before starting.

## Development

Rules for agent-driven development:

1. **Create a git worktree for every change — no exceptions, and no size-based or docs-only exemption.** The rule covers code, tests, scripts, config, and tracked documentation (AGENTS.md, README) alike. This repo is developed in parallel by multiple agents: if everyone assumes their edit is "small enough" to do in the main checkout, concurrent working trees collide and corrupt each other. Skipping the worktree for a "trivial", "one-line", or "docs-only" edit is a violation — a one-line fix or a docs-only change gets a worktree exactly like a large feature. **Preflight check:** before editing ANY tracked file in the repo — including AGENTS.md and README — confirm the working directory is a worktree under `.worktrees/<branch>`; if you are in the main checkout, stop and create a worktree first. Never develop directly on `master` / the main checkout. **Design docs are the one exemption.** Design and planning docs (technical design, UX design, plans, context, research notes, mockups and their companion assets) are ephemeral and never committed — they live in the main checkout's `docs/` folder (`/var/ai/agent_hub/docs/<area>/<topic>/`) on `master`, not inside a worktree's `docs/`. When invoking a skill that produces design documents (**technical-design**, **ux-design**, **brainstorming**, **writing-plans**), you MUST instruct the skill to write those docs to the main checkout's `docs/`, and never `git add` or commit them. If you believe a worktree is unnecessary, ask the human partner first — never decide unilaterally. Use the **using-git-worktrees** skill. Worktrees live under `.worktrees/<branch>` and are bind-mounted into the container at `/worktrees/<branch>`; the main checkout is at `/app`. `scripts/run_tests.sh` detects the worktree automatically and maps test paths to the right container folder — run it from the worktree root. **Cleanup after merge:** once the branch is merged, the worktree MUST be removed (`git worktree remove .worktrees/<branch>`) and the merged branch deleted (`git branch -d <branch>`); leaving stale worktrees or merged branches behind is a violation.
2. **All design and planning documents live under `docs/` — never at the repo root.** Every design and planning artifact — UX design docs, technical design docs, implementation plans, research notes, and anything else in that category — must be created under the main checkout's repo-root `docs/` folder (typically `docs/<area>/<topic>/`), together with its companion code and assets (mockup generator scripts, rendered mockups, QA screenshots, fonts/images), and is never committed. Placing a design doc or any companion file at the repo root is a violation, same severity as skipping the worktree rule.
3. **The main agent is a CTO — it does no real work, and subagents are required, not optional.** The main agent's role is limited to high-level architectural decisions, breaking tasks down into small ones, and delegating. Every unit of real work (editing code, running tests, exploring the codebase, verification) must be delegated to a subagent — never done directly by the main agent, no matter how small the task seems.
4. **No parallel subagents — one at a time, sequentially.** Spawn a single subagent, wait for it to finish, review its result, then spawn the next. Do not dispatch concurrent subagents. Slow but high quality: trading speed for correctness is the intended behavior.
5. **Use the `code-review` skill, not `requesting-code-review`** — these are two different skills; always use `code-review` after code changes.
6. **Self-added or self-expanded lint-skips are strictly forbidden.** Never add a `# lint-skip:` comment (or pass `--exclude`) to silence the project linter (the code-standard skill's `lint.py`) on your own judgment — every violation must be fixed. Expanding the scope of an existing skip (widening the code it covers, adding rules to it, or broadening an `--exclude` pattern) is equally forbidden without explicit user approval. Skipping is a last resort and requires the user to explicitly authorize it (see **Skipping Rules** in the code-standard skill's LINT.md).
7. **Self-merging to master is strictly forbidden.** Never merge a branch into `master` on your own judgment — every single merge requires the user's explicit approval, each time, with no standing blanket permission. Present the reviewed, verified branch and wait for the user to instruct the merge.

## Code Standards

Refer to the **code-standard** skill for the full set of coding conventions and rules. The skill must be loaded before writing any new code.

### Async Requirement

The server is asyncio-based; never block the event loop.

- All blocking operations (network, file, subprocess) must use native asyncio: `httpx` for HTTP, `aiofiles` for file I/O, `run_cli` (`app/lib/cli`) for CLI commands.
- Any operation expected to take longer than 5ms (CPU work such as PIL/ONNX inference, JSON parsing, DOM parsing) must run in a background thread and be awaited via `asyncio.to_thread`.
- **Async APIs at the lowest level possible.** A lib that does heavy work (disk IO, network, subprocess, inference — anything expected to exceed 5ms) exposes an **async** public API and owns the thread hop itself. Never `asyncio.to_thread` a project-local function from the calling side: if you find yourself writing that, the offload belongs inside the callee. The lib gets it right once, instead of every call site getting its own chance to forget.
- **The shape** is an async public function that is a thin `await asyncio.to_thread(_<name>_sync, ...)` over a private sync core — see `render/probe.py`, `text_type/detector.py`, `render/cache.py`, `render/fragments.py`. A `to_thread` of the lib's **own** private helper in the same module is that shape, not a violation; only reaching across a module boundary into another project module is.
- **Native asyncio beats a thread wherever it exists.** The rules above take precedence: put blocking HTTP on `httpx` and a blocking socket probe on `asyncio.open_connection` rather than wrapping either in `to_thread`. When porting `requests`/`urlopen` to `httpx`, pass `follow_redirects=True` — httpx does not follow redirects by default, and both of those do.
- **Sync entries are the escape hatch for a consumer with genuinely no event loop** — a separate process (the GC CLI), a WSGI worker thread (the WsgiDAV hook), `SyncChallenger`, or a sync helper that is already offloaded whole. Give it a public `<name>_sync` onto the same core instead of duplicating the logic: `cache.sweep_sync`, `cache.invalidate_sync`, `paths.resolve_path_sync`, `migration.migrate_all_sync`, `fragments.build_fragment_sync`, and `Detector.detect` beside `detect_async` where the package already suffixes the async twin instead.

### MCP Tool Documentation

MCP tool docs are read by LLM clients, not humans — every ambiguity costs a failed or mis-parameterized call.

- **`description` has one job: what the tool does + when to use it** — typically 1-3 short sentences, including disambiguation from the similar tools on this server (e.g. brave_search vs google_search, x_search vs general web search, web_fetch vs read_media_or_doc). It must NOT contain failure-string documentation (the client sees the failure at runtime, so its shape is useless up front), return-type/return-content contracts, or implementation details and reasoning. The duplicate-call (cache) note is synthesized by the server from the class `staleness` (see `Tool.dynamic_description` in `app/lib/tool/base.py`) — never write it into `description` (the invariant suite in `tests/app/mcp/test_registry.py` mechanically enforces the no-handwritten-note and no-failure-boilerplate halves; the no-return-contract half is a convention the invariants do not check). A tool whose `is_slow` is true answers a call with the task acknowledgement (the ack) at runtime, but that behavior is not part of the description. Parameter-level facts belong in the owning `Param.description`, not the tool description. MCP clients (including `agent_cli`) see the synthesized dynamic description; the console sees the base description plus the structured `is_slow`/`staleness`/`cached` fields.
- **Canonical multi-action tools must document like a CLI `--help` screen.** Each action gets its own block with explicit `required:` and `optional:` lines, including types, defaults, and valid values. Never rely on the client to infer which params an action needs.
- **Every parameter's docstring must state which actions/paths require it and which ignore it** ("Required by: X, Y. Not used by: Z").
- **The base `Param.description` is a short purpose statement only** — it must not restate the concrete non-empty default, the declared-enum value set (as a flat list), the min/max bounds, or the kind boilerplate (the resource transfer/listing instructions, the YYYY-MM-DD format, the http(s) URL scheme), because the server synthesizes those onto the wire from the `Param`'s `kind`/`default`/`enum` (see `Param.wire_description` in `app/lib/tool/base.py`), where every optional param carries the note, `Optional (default: X).` for a concrete non-empty scalar default and `Optional.` otherwise. The base must not self-describe as optional, because the synthesized note owns optionality. Genuinely param-specific facts stay in the base: truncation/`task_id` continuation, application details (redfin's "maximum 30 days" fallback), action-dependent value sets, per-value meanings for an enum, the cross-param "at least one of" rule, and the "Required by:/Not used by:" scoping on multi-action tools. The invariants in `tests/app/mcp/test_registry.py` (3g/3h/3i/3j) enforce no-restatement, no-kind-boilerplate, no-failure-strings, and no-"share"-term at the param level.
- **Constrain choice params with `enum`** — a declared `enum` is always enforced and projected: the registry synthesizes the `Literal` so the JSON schema carries the `enum`, and the wire description gains the "Valid values" note. There is no enforcement flag to set (there is no `enum_enforced`); the author's only job is to declare `enum=(…)`, and a prose-only enum is impossible.
- **Tool description and function docstring must agree** — what MCP clients see first is the synthesized dynamic description (the base description plus the cache note composed from `staleness`); the `run()` docstring is a minimal purpose statement consistent with the description, and param-level facts live in the `Param.description`s. What an MCP client sees for a param is the **synthesized** `Param.wire_description` (base + kind/default/enum notes) **plus** the wire schema `enum`, while the console shows the base plus the structured fields — the param-level mirror of the tool-level dynamic description.
- **Tool docs use only commas and periods — never semicolons** — this covers `description`, the synthesized param wire text, the `Param.description`s, and the `run()` docstring.
- **"Resource" is the only term for the resource — "share" is banned** — the word "share" must not be used to refer to the resource in any tool or param description (or the synthesized notes); "resource" is the only term (e.g. "the resource root", "in the resource"). The WebDAV **service** may still be "the WebDAV endpoint at /dav"; only the term for the resource itself is banned. Invariant 3j enforces this on the served description surface.

### After Code Changes

1. **Self-check lint compliance** — run the lint script to verify the code passes all linting checks:

```bash
python /root/.config/opencode/skills/code-standard/scripts/lint.py . --exclude data --exclude .superpowers --exclude docs
```

The `docs/` folder contains design documents and mockup scripts that are not part of the application, so it is excluded from linting.

2. **Code review** — once lint passes, invoke the **code-review** skill to perform a code review of all changes before considering the work complete.
3. **Manual test against the live container (required)** — after code changes, verify the change works against the running container (e.g. call the affected MCP tool via `scripts/agent_cli`, hit the HTTP endpoints, or inspect `docker compose logs agent_hub` from `/var/ai`). Unit and live tests alone do not substitute for this step; the work is not complete until the change is confirmed working live. For visual verification, a screenshot can be taken with the `web_screenshot` tool by using `http://localhost:8315` as the service URL. When the change touches the UI (the console, noVNC-visible surfaces, or any served HTML/CSS/JS), that screenshot is required: take a live `web_screenshot` of the affected page and inspect it for visual defects (broken layout, missing styles, overlapping or missing elements), because a UI change is not verified until the live screenshot shows no visual regression.
