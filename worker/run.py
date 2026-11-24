"""Automated session analysis: reports, videos and a written summary for every swing.

Process one session folder on disk:
    python -m worker.run local <session_dir> --out <results_dir> [--level youth] [--name Mason]

Watch a Google Drive folder and process each new session put in it:
    python -m worker.run watch            (configured by environment variables, see worker/README.md)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Dict, List, Optional

from . import report as report_mod
from .session import Session, load_session
from .summary import build_summary

RESULTS_FOLDER = os.environ.get('RESULTS_FOLDER_NAME', 'Swing Optimizer Results')
VIDEO_WORKERS = int(os.environ.get('VIDEO_WORKERS', '2'))


def log(msg: str) -> None:
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def make_videos(session: Session, out_dir: str) -> Dict[str, str]:
    """Camera 0 | Camera 1 | skeleton video per swing that has both synced videos and markers."""
    jobs = []
    for t in session.trials:
        if not (t.trc and t.cam0 and t.cam1):
            continue
        out = os.path.join(out_dir, f'{session.athlete.name}_{t.name}_Video.mp4')
        cmd = [sys.executable, '-m', 'worker.video', t.trc, t.cam0, t.cam1, out,
               f'{session.athlete.name} - {t.name}']
        jobs.append((t.name, out, cmd))

    def run(job):
        name, out, cmd = job
        r = subprocess.run(cmd, cwd=report_mod.REPO, capture_output=True, text=True)
        return name, (out if r.returncode == 0 else None), r.stderr[-400:]

    made = {}
    with ThreadPoolExecutor(VIDEO_WORKERS) as ex:
        for name, out, err in ex.map(run, jobs):
            if out:
                made[name] = out
            else:
                log(f'  video {name} failed: {err.strip()}')
    return made


def process(session_dir: str, out_dir: str, name: Optional[str] = None,
            level: Optional[str] = None, app_url: Optional[str] = None) -> Dict:
    """Everything for one session, written to out_dir. Returns what was made."""
    os.makedirs(out_dir, exist_ok=True)
    session = load_session(session_dir, default_name=name,
                           default_level=level or os.environ.get('DEFAULT_LEVEL', 'high_school'))
    a = session.athlete
    log(f'{a.name}: {len(session.trials)} swings, level {a.level}, '
        f'handedness {a.handedness or "inferred"}, bat {a.bat_length_in or "default"} in')
    if not session.trials:
        raise RuntimeError('no swings found (expected OpenSimData/Kinematics/*.mot)')

    if app_url:
        results = report_mod.make_reports(session, out_dir, app_url, log=log)
    else:
        with report_mod.local_app(os.path.join(out_dir, '.app.log')) as url:
            results = report_mod.make_reports(session, out_dir, url, log=log)
    report_mod.write_summary_json(results, os.path.join(out_dir, '.results.json'))

    videos = make_videos(session, out_dir)
    log(f'  {len(videos)} videos')
    summary_path = os.path.join(out_dir, f'{a.name}_Summary.md')
    with open(summary_path, 'w') as fh:
        fh.write(build_summary(a, results, session.skipped))
    log(f'  summary written')
    return {'athlete': a.name, 'summary': summary_path,
            'reports': [r['pdf'] for r in results if 'pdf' in r],
            'videos': list(videos.values()),
            'failed': [r['trial'] for r in results if 'error' in r]}


# ---------------------------------------------------------------- Drive watch mode

def _upload_results(svc, made: Dict, results_id: str) -> None:
    from . import drive
    drive.upload(svc, made['summary'], results_id, 'text/markdown', as_google_doc=True,
                 name=os.path.splitext(os.path.basename(made['summary']))[0])
    for p in made['reports']:
        drive.upload(svc, p, results_id, 'application/pdf')
    for p in made['videos']:
        drive.upload(svc, p, results_id, 'video/mp4')


def _status(svc, results_id: str, text: str, work: str) -> None:
    from . import drive
    p = os.path.join(work, 'STATUS.txt')
    with open(p, 'w') as fh:
        fh.write(text + '\n')
    drive.upload(svc, p, results_id, 'text/plain')


def watch_once(svc, inbox_id: str, settle: timedelta) -> int:
    """Process every session folder in the inbox that has no results yet. Returns how many."""
    from . import drive
    done = 0
    for folder in drive.children(svc, inbox_id, folders_only=True):
        if drive.find_child(svc, folder['id'], RESULTS_FOLDER):
            continue                                   # already processed (delete it to re-run)
        changed = drive.newest_change(svc, folder['id'])
        if changed and drive.now() - changed < settle:
            log(f"{folder['name']}: still uploading, waiting")
            continue
        log(f"{folder['name']}: new session")
        with tempfile.TemporaryDirectory() as work:
            results_id = drive.ensure_folder(svc, folder['id'], RESULTS_FOLDER)
            _status(svc, results_id, f'Processing since {drive.now():%Y-%m-%d %H:%M} UTC', work)
            try:
                src = os.path.join(work, 'session')
                n = drive.download_tree(svc, folder['id'], src)
                log(f'  downloaded {n} files')
                made = process(src, os.path.join(work, 'out'), name=folder['name'])
                _upload_results(svc, made, results_id)
                msg = (f"Done {drive.now():%Y-%m-%d %H:%M} UTC: {len(made['reports'])} reports, "
                       f"{len(made['videos'])} videos.")
                if made['failed']:
                    msg += ' Not analyzed: ' + ', '.join(made['failed'])
                _status(svc, results_id, msg, work)
                done += 1
            except Exception as e:
                log(f'  FAILED: {e}')
                traceback.print_exc()
                _status(svc, results_id, f'Failed: {e}\n\nDelete this folder to try again.', work)
    return done


def watch() -> None:
    from . import drive
    inbox = os.environ.get('INBOX_FOLDER_ID')
    if not inbox:
        sys.exit('set INBOX_FOLDER_ID to the Drive folder that sessions are put in')
    every = int(os.environ.get('POLL_SECONDS', '300'))
    settle = timedelta(minutes=float(os.environ.get('SETTLE_MINUTES', '10')))
    svc = drive.service()
    log(f'watching Drive folder {inbox} every {every} s')
    while True:
        try:
            watch_once(svc, inbox, settle)
        except Exception as e:                         # a Drive hiccup must not stop the worker
            log(f'poll failed: {e}')
        time.sleep(every)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog='python -m worker.run', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    loc = sub.add_parser('local', help='process one session folder on disk')
    loc.add_argument('session_dir')
    loc.add_argument('--out', required=True)
    loc.add_argument('--name', help='athlete name (default: from the folder name)')
    loc.add_argument('--level', help='default level when the folder does not say')
    loc.add_argument('--app-url', help='use a running app instead of starting one')
    sub.add_parser('watch', help='watch the Drive inbox folder')
    args = ap.parse_args(argv)
    if args.cmd == 'local':
        made = process(args.session_dir, args.out, args.name, args.level, args.app_url)
        log(f"done: {len(made['reports'])} reports, {len(made['videos'])} videos, summary {made['summary']}")
    else:
        watch()


if __name__ == '__main__':
    main()
