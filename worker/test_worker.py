"""Tests for the automated worker that need no browser, Drive or API key.

Run from the repo root: python -m pytest worker  (or python -m worker.test_worker)
"""
import os
import sys
import tempfile
import types

from worker import summary
from worker.session import Athlete, load_session, parse_folder_name


def test_folder_name_gives_athlete_details():
    assert parse_folder_name('Mason youth right 28in 18oz') == {
        'name': 'Mason', 'level': 'youth', 'handedness': 'right',
        'bat_length_in': 28.0, 'bat_weight_oz': 18.0}
    assert parse_folder_name('Joe Smith - college L 33in') == {
        'name': 'Joe Smith', 'level': 'college', 'handedness': 'l', 'bat_length_in': 33.0}
    assert parse_folder_name('Tosh high school')['level'] == 'high school'
    assert parse_folder_name('Hunter_1008') == {'name': 'Hunter 1008'}


def _touch(root, rel, text=''):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as fh:
        fh.write(text)
    return p


def test_session_finds_swings_and_skips_calibration():
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, 'Mason youth right 28in')
        for t in ('neutral', 'swing_1', 'swing_2', 'swing_10'):
            _touch(root, f'MarkerData/{t}.trc')
        for t in ('neutral', 'swing_1', 'swing_10'):
            _touch(root, f'OpenSimData/Kinematics/{t}.mot')
        _touch(root, 'OpenSimData/Model/LaiUhlrich2022.osim')
        _touch(root, 'OpenSimData/Model/LaiUhlrich2022_shoulder_scaled.osim')
        for c in ('Cam0', 'Cam1'):
            _touch(root, f'Videos/{c}/InputMedia/swing_1/swing_1_sync.mp4')
        _touch(root, 'sessionMetadata.yaml', 'height_m: 1.51\nmass_kg: 39\n')
        _touch(root, 'athlete.yaml', 'bat_weight_oz: 18\nlevel: hs\n')   # file wins over folder name

        s = load_session(root)
        assert [t.name for t in s.trials] == ['swing_1', 'swing_10']     # natural order, no neutral
        assert s.skipped == ['swing_2: no .mot (OpenCap did not finish this trial)']
        assert s.model.endswith('_scaled.osim')
        assert s.trials[0].cam0 and s.trials[0].cam1 and not s.trials[1].cam0
        a = s.athlete
        assert (a.name, a.level, a.handedness) == ('Mason', 'high_school', 'right')
        assert (a.height_m, a.mass_kg, a.bat_length_in, a.bat_weight_oz) == (1.51, 39.0, 28.0, 18.0)


def _result(trial, score=72.0, seq_status='skipped'):
    return {'trial': trial, 'pdf': f'{trial}.pdf', 'analysis': {
        'metrics': {'max_hand_speed_mph': 17.7},
        'mechanics_score': {'score': 90},
        'capture_quality': {'usable': True, 'reasons': []},
        'phase_report': {
            'swing_score_detail': {'score': score, 'components': [
                {'key': 'hand_speed', 'label': 'Hand / Bat Speed', 'value': 17.7, 'unit': 'mph',
                 'score': 72.5, 'status': 'scored', 'reason': ''},
                {'key': 'sequence_quality', 'label': 'Sequence Quality', 'value': -77.3,
                 'unit': 'ms lag', 'score': None, 'status': seq_status,
                 'reason': 'torso signal unstable'},
            ]},
            'prescriptions': [
                {'key': 'sequence_quality', 'label': 'Sequence Quality', 'cue': 'hips first',
                 'drill': 'step-behind', 'why': 'order matters'},
                {'key': 'hand_speed', 'label': 'Hand / Bat Speed', 'cue': 'fast hands',
                 'drill': 'overload bats', 'why': 'speed'},
            ]}}}


def test_grounding_drops_advice_for_unmeasured_parts():
    g = summary.swing_grounding(_result('s1'))
    assert [p['label'] for p in g['prescriptions']] == ['Hand / Bat Speed']
    seq = next(p for p in g['parts'] if p['part'] == 'Sequence Quality')
    assert seq['why_skipped'] == 'torso signal unstable' and 'score_0_100' not in seq
    failed = summary.swing_grounding({'trial': 's2', 'error': 'analysis failed: x'})
    assert failed == {'swing': 's2', 'analyzed': False, 'why': 'analysis failed: x'}


def test_summary_without_api_key_is_the_table(monkeypatch=None):
    old = {k: os.environ.pop(k, None) for k in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN')}
    try:
        results = [_result('s1'), _result('s2', score=None), {'trial': 's3', 'error': 'boom'}]
        md = summary.build_summary(Athlete(name='Mason', level='youth', handedness='right',
                                           bat_length_in=28), results, ['s4: no .mot'])
    finally:
        os.environ.update({k: v for k, v in old.items() if v})
    assert md.startswith('# Mason: session summary')
    assert '| s1 | 72 | 90 | 17.7 |  |' in md
    assert '| s2 | — | 90 | 17.7 | score withheld: capture unreliable |' in md
    assert '| s3 | — | — | — | not analyzed: boom |' in md
    assert 'Not processed: s4: no .mot' in md


def test_coach_note_sends_grounding_and_reads_text():
    sent = {}

    class FakeMessages:
        def create(self, **kw):
            sent.update(kw)
            return types.SimpleNamespace(stop_reason='end_turn', content=[
                types.SimpleNamespace(type='thinking', thinking=''),
                types.SimpleNamespace(type='text', text='**Summary**: solid day.')])

    fake = types.ModuleType('anthropic')
    fake.Anthropic = lambda: types.SimpleNamespace(beta=types.SimpleNamespace(messages=FakeMessages()))
    fake.APIStatusError = fake.APIConnectionError = RuntimeError
    real = sys.modules.get('anthropic')
    sys.modules['anthropic'] = fake
    os.environ['ANTHROPIC_API_KEY'] = 'test'
    try:
        note = summary.coach_note(summary.grounding(Athlete(name='M', level='youth'), [_result('s1')], []))
    finally:
        os.environ.pop('ANTHROPIC_API_KEY')
        if real is not None:
            sys.modules['anthropic'] = real
        else:
            del sys.modules['anthropic']
    assert note == '**Summary**: solid day.'
    assert sent['model'] == summary.MODEL and sent['fallbacks'] == 'default'
    assert '"swing": "s1"' in sent['messages'][0]['content']
    assert 'thinking' not in sent                       # Opus 5.5: thinking stays on; effort sets depth


class FakeDrive:
    """Just enough of the Drive v3 client for watch_once: list, create, update."""

    def __init__(self):
        self.files_by_id, self.n = {}, 0

    def add(self, name, parent, folder=False, age_min=60):
        from datetime import datetime, timedelta, timezone
        self.n += 1
        fid = f'id{self.n}'
        t = (datetime.now(timezone.utc) - timedelta(minutes=age_min)).strftime('%Y-%m-%dT%H:%M:%S.000Z')
        self.files_by_id[fid] = {'id': fid, 'name': name, 'parent': parent, 'modifiedTime': t,
                                 'createdTime': t, 'mimeType': 'application/vnd.google-apps.folder'
                                 if folder else 'application/octet-stream'}
        return fid

    def files(self):
        return self

    def _exec(self, value):
        return types.SimpleNamespace(execute=lambda: value)

    def list(self, q, **kw):
        parent = q.split("'")[1]
        out = [f for f in self.files_by_id.values() if f['parent'] == parent
               and ('mimeType =' not in q or f['mimeType'].endswith('folder'))]
        return self._exec({'files': out})

    def create(self, body, media_body=None, **kw):
        fid = self.add(body['name'], body['parents'][0],
                       folder=body.get('mimeType', '').endswith('folder'), age_min=0)
        if body.get('mimeType', '').endswith('document'):
            self.files_by_id[fid]['mimeType'] = body['mimeType']
        return self._exec({'id': fid})

    def update(self, fileId, media_body=None, **kw):
        self.files_by_id[fileId]['updated'] = True
        return self._exec({'id': fileId})

    def names(self, parent):
        return sorted(f['name'] for f in self.files_by_id.values() if f['parent'] == parent)


def test_watch_processes_new_sessions_once():
    from datetime import timedelta
    from worker import drive, run

    svc = FakeDrive()
    inbox = svc.add('inbox', 'root', folder=True)
    new = svc.add('Mason youth', inbox, folder=True)
    svc.add('x.trc', new)
    done = svc.add('Tosh college', inbox, folder=True)
    svc.add(run.RESULTS_FOLDER, done, folder=True)
    busy = svc.add('Sean college', inbox, folder=True)
    svc.add('y.trc', busy, age_min=1)                   # still uploading

    calls = []

    def fake_process(src, out, name=None, **kw):
        calls.append(name)
        os.makedirs(out, exist_ok=True)
        paths = {}
        for f in ('Mason_Summary.md', 'Mason_t1_Report.pdf', 'Mason_t1_Video.mp4'):
            paths[f] = _touch(out, f, 'x')
        return {'athlete': 'Mason', 'summary': paths['Mason_Summary.md'],
                'reports': [paths['Mason_t1_Report.pdf']], 'videos': [paths['Mason_t1_Video.mp4']],
                'failed': ['t2']}

    real_dl, real_proc = drive.download_tree, run.process
    drive.download_tree = lambda svc_, fid, dest: (os.makedirs(dest, exist_ok=True), 1)[1]
    run.process = fake_process
    try:
        assert run.watch_once(svc, inbox, timedelta(minutes=10)) == 1
        assert run.watch_once(svc, inbox, timedelta(minutes=10)) == 0   # results folder now exists
    finally:
        drive.download_tree, run.process = real_dl, real_proc
    assert calls == ['Mason youth']
    res = drive.find_child(svc, new, run.RESULTS_FOLDER)['id']
    assert svc.names(res) == ['Mason_Summary', 'Mason_t1_Report.pdf', 'Mason_t1_Video.mp4', 'STATUS.txt']
    status = drive.find_child(svc, res, 'STATUS.txt')
    assert status.get('updated')                           # "processing" was replaced by "done"
    assert drive.find_child(svc, res, 'Mason_Summary')['mimeType'].endswith('document')


if __name__ == '__main__':
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); n += 1
    print(f'{n} worker tests passed')
