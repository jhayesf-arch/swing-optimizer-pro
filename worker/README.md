# Automated analysis worker

Put an OpenCap session in a Google Drive folder; a few minutes later, a
**Swing Optimizer Results** folder appears inside it with:

- one PDF report per swing (the app's own report, identical to a manual upload)
- one video per swing: Camera 0 | Camera 1 | 3-D skeleton, quarter speed
- a **Summary** Google Doc: every swing's scores in a table, plus a coach's note
  written by Claude (summary, what's working, the top fix, which swings to re-film)
- `STATUS.txt`: processing, done (with any swings that could not be analyzed), or the error

Claude only writes the note. Every number comes from the analysis engine, and the
note is given the reasons parts were skipped, so it does not coach on a bad capture.
Without an `ANTHROPIC_API_KEY`, the summary is just the table.

## Using it

1. Copy the session from OpenCap (`MarkerData`, `OpenSimData`, `Videos`,
   `sessionMetadata.yaml`) into a new folder inside the inbox folder.
2. Name the folder with the athlete's details, name first:
   `Mason youth right 28in 18oz`, `Joe college left 33in 30oz`.
   Level is one of `youth`, `high school`, `college`, `pro`. Anything left out uses
   the defaults (level `DEFAULT_LEVEL`, handedness inferred, the app's default bat).
   Or put an `athlete.yaml` (or a Google Doc named `athlete`) in the folder:

   ```yaml
   name: Mason
   level: youth
   handedness: right
   bat_length_in: 28
   bat_weight_oz: 18
   ```

3. The worker waits until nothing in the folder has changed for `SETTLE_MINUTES`,
   so a half-finished upload is not processed.

To process a session again (new settings, new code), delete its
**Swing Optimizer Results** folder.

The raw `.mov` files and OpenPose pickles are not downloaded; only the `.trc`,
`.mot`, `.osim` and the synced `*_sync.mp4` videos are used.

## Running it

On your computer or any server, from the repo root:

```bash
pip install -r backend/requirements.txt -r worker/requirements.txt
playwright install chromium        # once; needs ffmpeg on the PATH too

# one session folder on disk, no Drive needed
python -m worker.run local "path/to/Mason youth right 28in" --out results/

# watch Drive
python -m worker.run watch
```

On Render, `render.yaml` defines it as a background worker
(`swing-optimizer-worker`, built from `Dockerfile.worker`). Background workers are
a paid Render feature; the `standard` plan (2 GB) is set because the browser,
OpenSim and video rendering together need more than 512 MB.

## Settings (environment variables)

| Variable | |
|---|---|
| `INBOX_FOLDER_ID` | Drive folder that sessions are put in (the ID at the end of its URL) |
| `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`, `GOOGLE_OAUTH_REFRESH_TOKEN` | Drive sign-in, see below |
| `ANTHROPIC_API_KEY` | for the written coach's note (optional) |
| `SUMMARY_MODEL` | default `claude-opus-5-5` |
| `SUMMARY_EFFORT` | default `medium` (`low` is cheaper, `high` more thorough) |
| `DEFAULT_LEVEL` | when the folder does not say; default `high_school` |
| `POLL_SECONDS` | how often to check the inbox; default 300 |
| `SETTLE_MINUTES` | quiet time before a folder counts as fully uploaded; default 10 |
| `VIDEO_WORKERS` | videos rendered at once; default 2 |
| `RESULTS_FOLDER_NAME` | default `Swing Optimizer Results` |

### Drive sign-in

The worker acts as your Google account, so it can read the inbox and write the
results into your Drive.

1. In [Google Cloud Console](https://console.cloud.google.com/), create a project,
   enable the **Google Drive API**, and set up the **OAuth consent screen**
   (External, add yourself as a test user).
2. Set the consent screen's publishing status to **In production**. While it stays
   in *Testing*, Google expires the sign-in after 7 days and the worker stops.
   Unverified is fine for your own use; you will see a warning screen once.
3. Create an **OAuth client ID** of type *Desktop app* and download its JSON.
4. On your computer: `pip install google-auth-oauthlib`, then
   `python -m worker.auth client_secret.json`. Sign in with the account that owns
   the Drive folder. It prints the three `GOOGLE_OAUTH_*` values to set on the worker.

A Google **service account** also works (`GOOGLE_SERVICE_ACCOUNT_JSON`), but only
for a Workspace **shared drive**: service accounts have no storage of their own, so
they cannot upload into a personal My Drive.

## Cost

One Claude request per session: about 1,000 input tokens per swing (a 14-swing
session is about 15,000) plus the note and the model's thinking. With Claude Opus 5.5
at medium effort that is roughly 5–15 cents a session.
Processing time is about 10 s per swing for the report and 15–30 s per swing for
the video.

## Tests

`python -m worker.test_worker` (no browser, Drive or API key needed).
