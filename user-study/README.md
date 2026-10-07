# DiagramLab user study

A React/Vinext interface with a Cloudflare Worker backend and durable D1 (SQLite) storage. It uses the repository's **ten current dimensions**: Layout, Connectivity, Presence, Details, Legibility, Aesthetics, Palette, Ink Balance, Density, and Balance. Each response is an integer from **1 to 10**; scores start unanswered.

## Participant experience

- Anonymous session after an explicit consent checkbox; no name or email is stored.
- 24 diagrams by default, exactly 8 easy / 8 medium / 8 hard, randomly ordered once for each session. Assignments of 20–30 are supported; counts differ by at most one for sizes not divisible by three. Use 21, 24, 27 or 30 for exact balance.
- Difficulty labels, provenance, and model scores are withheld from the participant API. Each diagram is shown with its caption, and the Presence question asks about the components the caption mentions.
- Correctness and Beauty tabs, explanatory anchors, zoom/fullscreen, comments, progress navigation, automatic draft saves and submission locking.
- Returning with the same browser cookie resumes the saved session. Clearing cookies or switching browsers creates a new anonymous session; this prototype does not deduplicate real people or provide cross-device recovery.
- Responses are stored in D1, not browser storage. Only a random bearer session cookie is stored in the browser (HttpOnly, SameSite=Lax, Secure on HTTPS, 30 days). Keep the researcher key separate from participant links.

## Demo versus a real study

The bundled `config/study.json` contains **24 existing repository diagrams** (10 DiagramGen examples, 11 Transformer ablations, 3 Beauty variants). Their captions were written for the demo from each diagram's generation prompt; real studies should use the diagrams' own captions. All images are under `public/diagrams/`; the DiagramGen source card is retained as `DATASET_SOURCE.md`. Difficulty labels are **provisional demonstration labels**, deliberately balanced to exercise the allocator. They are not validated measures of quality or rating difficulty. Demo sessions and CSV rows are explicitly marked `mode=demo`. No model APIs are called by this app.

Before recruitment:

1. **List your diagrams in a CSV manifest.** Copy `config/diagram-manifest.example.csv` next to your images and fill one row per diagram. Required columns are `file` (path relative to the CSV), `difficulty` (`easy`, `medium` or `hard`) and either `caption` (the diagram's caption text) or `caption_file` (a path to a text file holding it). `id` and `source` are optional; a missing `id` becomes `D-` plus the first 10 hex characters of the image hash, so it stays stable when rows are reordered. Prefer a varied corpus; the demo has related variants of the same base diagrams. Difficulty should describe the pre-agreed sampling criterion, not be silently inferred from model scores.
2. **Import it.** A dry run checks every row and the difficulty balance without writing anything:

   ```bash
   npm run import:diagrams -- path/to/manifest.csv --dry-run
   npm run import:diagrams -- path/to/manifest.csv --mode research --id flowchart-study-v1 --sample-size 24 \
     --difficulty-note "Easy: <=6 nodes, no branches. Medium: ... Hard: ..."
   npm run validate:study
   ```

   The importer copies each image to `public/diagrams/<first 20 hex of SHA-256>.<ext>`, records the full SHA-256, and rewrites `config/study.json`. It refuses duplicate images, duplicate IDs, unknown difficulties, missing files, and pools too small for a balanced assignment, naming the row at fault. It never deletes old images.
3. **Choose `sampleSize` (20–30).** The pool can be larger than the assignment. Each session needs `floor(sampleSize / 3)` diagrams per difficulty, so 24 needs at least 8 easy, 8 medium and 8 hard. Use 21, 24, 27 or 30 for exact balance; other sizes differ by at most one per group.
4. **Review wording and mode.** Check the rating wording in `lib/metrics.ts` and the consent text for your protocol. Use `--mode research`, a new `--id` per study round, and a real `--difficulty-note`. Restart the dev server, or run `npm run deploy:cloudflare` again, after importing.
5. **Versions.** Each catalog hash becomes a new study version. Existing sessions retain their original assignment snapshots, captions, difficulty and provenance. Keep old image assets available while those sessions remain active. The dashboard and CSV endpoint select the currently configured version; historical versions remain in the database.

This is a working collection interface, not a claim that the demo sample or anchors are scientifically calibrated. The UI does not compute an overall score or expose model scores to raters.

## Run locally

Node.js 22.13+ is required. From `user-study/`:

```bash
npm ci
cp .env.example .env
# Replace STUDY_ADMIN_KEY in .env with a unique random value of at least 24 characters.
npm run build
node --import ./scripts/sites-env.mjs ./node_modules/wrangler/bin/wrangler.js d1 execute DB --local --config dist/server/wrangler.json --persist-to .wrangler/state --file drizzle/0000_glorious_george_stacy.sql
npm run dev -- --hostname 127.0.0.1
```

Apply that migration only once to a fresh local database. After later schema changes, generate and apply only new migrations. The local preview is at the URL printed by the dev server. Production uses a separate database; local test responses are never published.

The generated local `.env` already contains the researcher key for this checkout. Do not overwrite it with the example unless intentionally replacing the key. `.env`, `.wrangler/`, `.vinext/`, dependencies, and build output are ignored by Git. `.openai/hosting.json` stores only the Site ID and logical database binding.

## Deploy to Cloudflare

The app runs on Cloudflare Workers with a D1 database. The free plan covers a study of this size; it allows about 100,000 database row writes per day, roughly 300 participants per day. Above that, the Workers Paid plan costs $5 per month.

One-time setup:

1. Create a free Cloudflare account, open **Workers & Pages** in the dashboard once, and pick a `workers.dev` subdomain.
2. From `user-study/`, log in. This opens the browser:

   ```bash
   npx wrangler login
   ```

Deploy, and redeploy after any change:

```bash
npm run deploy:cloudflare
```

The script validates the catalog, finds or creates the D1 database named in `config/cloudflare.json` and saves its id there, builds, applies new migrations from `drizzle/`, deploys, and sets a researcher key if none exists. It prints the participant link and saves the researcher key to `.env.cloudflare`, which Git ignores. Every step is safe to re-run.

- `npm run deploy:cloudflare -- --dry-run` builds and checks the bundle without logging in or uploading anything.
- `npm run deploy:cloudflare -- --rotate-key` replaces the researcher key, for example on a new computer without `.env.cloudflare`.
- Change `workerName` in `config/cloudflare.json` before the first deploy to change the link, `https://<workerName>.<subdomain>.workers.dev`.
- The deployed database is separate from the local one. Local test sessions are never uploaded.

## Researcher workspace and CSV

Open `/researcher` and enter the `STUDY_ADMIN_KEY`. Locally it is in `.env`; for the Cloudflare deployment it is in `.env.cloudflare`, a different key. Every dashboard/export request is authorized server-side; the access key is kept in page memory and cleared on reload. Participant sessions cannot read other responses or export data.

Choose **Completed sessions** or **All sessions, including partial**, then **Export CSV**. The result is UTF-8 CSV with a BOM, one row per assigned participant–diagram pair:

```text
study_version, mode, participant_code, session_status,
diagram_id, diagram_source, image_sha256, difficulty, presentation_order,
layout, connectivity, presence, details, legibility,
aesthetics, palette, ink_balance, density, balance,
rating_complete, comment, active_duration_ms,
session_started_at, rating_updated_at, session_submitted_at
```

Unanswered dimensions are blank, never zero. Durations are accumulated foreground viewing time, not a reliable attention measure. Comments are CSV-escaped and formula-like text is neutralized for spreadsheets. The API is `GET /api/export?scope=completed` or `?scope=all` with the `x-study-admin-key` header. Exporting does not change or delete responses.

The deployed `workers.dev` link is public: anyone with it can start a session, so share it only with participants. Researcher access stays protected by the key.

## Validation

```bash
npm run validate:study
npm test
npx tsc --noEmit
npm run lint
python3 tests/smoke_api.py http://127.0.0.1:5173
```

The API smoke test is restricted to localhost and creates demo-only local sessions. It checks the full 24 × 10 flow, exact difficulty balance, authorization, session isolation, invalid scores, stable ordering, submission locking, complete/partial CSV export and spreadsheet-formula escaping. It must not run against a real participant study. The test saves its session IDs in the OS temporary directory for cleanup.
