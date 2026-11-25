"""Make each swing's PDF report by running it through the app itself, in a headless browser.

The PDF is the app's own printed report, so it is identical to what a coach gets by
uploading the swing by hand. The analysis JSON the app receives is kept too; the
AI summary is written from it.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional

from .session import Session, Trial

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_TIMEOUT_MS = 300_000


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@contextmanager
def local_app(log_path: Optional[str] = None) -> Iterator[str]:
    """Start the Swing Optimizer app on a private port for the duration of a run."""
    port = _free_port()
    log = open(log_path, 'ab') if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', str(port)],
        cwd=os.path.join(REPO, 'backend'), stdout=log, stderr=subprocess.STDOUT)
    url = f'http://127.0.0.1:{port}'
    try:
        for _ in range(120):
            if proc.poll() is not None:
                raise RuntimeError('the app exited during startup; see the app log')
            try:
                urllib.request.urlopen(url + '/api/health', timeout=2)
                break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError('the app did not start within 60 s')
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        if log_path:
            log.close()


def _feet_inches(height_m: float):
    inches = height_m / 0.0254
    ft = int(inches // 12)
    inch = round(inches - ft * 12)
    if inch == 12:
        ft, inch = ft + 1, 0
    return ft, inch


def make_reports(session: Session, out_dir: str, app_url: str, log=print) -> List[Dict]:
    """One PDF per swing in out_dir. Returns one result per trial (scores, analysis JSON or error)."""
    from playwright.sync_api import TimeoutError as PWTimeout, sync_playwright

    a = session.athlete
    results = []
    launch = {'executable_path': os.environ['CHROMIUM_PATH']} if os.environ.get('CHROMIUM_PATH') else {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(**launch)
        try:
            for trial in session.trials:
                res = _one(browser, trial, session, out_dir, app_url, PWTimeout)
                results.append(res)
                log(f"  {trial.name}: " + (f"score {res['score']}, hand {res['hand_speed']} mph"
                                           if 'error' not in res else 'FAILED - ' + res['error']))
        finally:
            browser.close()
    return results


def _one(browser, trial: Trial, session: Session, out_dir: str, app_url: str, PWTimeout) -> Dict:
    a = session.athlete
    page = browser.new_page()
    page_errors: List[str] = []
    page.on('pageerror', lambda e: page_errors.append(str(e)))
    page.add_init_script('window.print = () => { window.__printed = true; };')
    try:
        page.goto(app_url + '/')
        if trial.trc:
            page.set_input_files('#trc-input', trial.trc)
        if session.model:
            page.set_input_files('#model-input', session.model)
        page.set_input_files('#file-input', trial.mot)
        page.wait_for_selector('#demo-modal:not(.hidden)', timeout=15_000)
        if a.height_m:
            ft, inch = _feet_inches(a.height_m)
            page.fill('#height-ft', str(ft))
            page.fill('#height-in', str(inch))
        if a.mass_kg:
            page.fill('#weight-lbs', str(round(a.mass_kg * 2.20462)))
        # Age is only printed; leave it blank rather than print the form's default.
        page.fill('#athlete-age', str(a.age) if a.age else '')
        page.select_option('#athlete-handedness', a.handedness or '')
        page.click(f'.skill-pill[data-level="{a.level}"]')
        if a.bat_length_in:
            page.fill('#bat-length-in', f'{a.bat_length_in:g}')
        if a.bat_weight_oz:
            page.fill('#bat-weight-oz', f'{a.bat_weight_oz:g}')

        with page.expect_response(lambda r: '/api/analyze/upload' in r.url,
                                  timeout=ANALYSIS_TIMEOUT_MS) as resp_info:
            page.click('#btn-run-physics')
        try:
            analysis = resp_info.value.json()
        except Exception:
            analysis = None
        if not analysis or not analysis.get('success'):
            err = (analysis or {}).get('error') or f'HTTP {resp_info.value.status}'
            return {'trial': trial.name, 'error': f'analysis failed: {err}'}
        try:
            page.wait_for_selector('#results-section:not(.hidden)', timeout=60_000)
        except PWTimeout:
            toast = page.eval_on_selector('#error-toast', 'el => el.innerText') if page.query_selector('#error-toast') else ''
            return {'trial': trial.name, 'error': f'report did not render: {toast}'.strip()}
        page.wait_for_timeout(2500)          # charts and the 3-D view finish drawing
        page.click('#btn-report')
        page.wait_for_function('window.__printed === true', timeout=15_000)
        page.emulate_media(media='print')
        pdf = os.path.join(out_dir, f'{a.name}_{trial.name}_Report.pdf')
        page.pdf(path=pdf, format='A4', print_background=True,
                 margin={'top': '14mm', 'bottom': '14mm', 'left': '14mm', 'right': '14mm'})
        tiles = page.evaluate("""() => ({
            score: document.getElementById('swing-score-number').textContent.trim(),
            hand: document.getElementById('exit-velo-number').textContent.trim(),
            mech: document.getElementById('efficiency-number').textContent.trim(),
        })""")
        return {'trial': trial.name, 'pdf': pdf, 'score': tiles['score'], 'hand_speed': tiles['hand'],
                'mechanics': tiles['mech'], 'analysis': analysis.get('data'),
                'page_errors': page_errors}
    except PWTimeout as e:
        return {'trial': trial.name, 'error': f'timed out: {e}'}
    finally:
        page.close()


def write_summary_json(results: List[Dict], path: str) -> None:
    slim = [{k: v for k, v in r.items() if k != 'analysis'} for r in results]
    with open(path, 'w') as fh:
        json.dump(slim, fh, indent=1)
