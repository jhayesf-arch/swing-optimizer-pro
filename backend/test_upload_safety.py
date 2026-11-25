"""Uploads never touch paths the client names, never share files between requests,
and are size-limited. Run: python test_upload_safety.py"""
import io
import os
import tempfile

from fastapi.testclient import TestClient

import main

client = TestClient(main.app)
JUNK = b'not a real motion file\n'


def _post(name, data=JUNK, path='/api/analyze/upload', field='file'):
    return client.post(path, files={field: (name, io.BytesIO(data), 'application/octet-stream')})


def test_a_filename_with_a_path_cannot_reach_outside_the_temp_folder():
    with tempfile.TemporaryDirectory() as outside:
        victim = os.path.join(outside, 'victim.mot')
        with open(victim, 'w') as fh:
            fh.write('keep me')
        _post(victim)                                          # absolute path as the filename
        _post(os.path.join('..', '..', os.path.relpath(victim, main.TMP_DIR)))   # relative escape
        assert open(victim).read() == 'keep me'


def test_batch_filenames_cannot_reach_outside_the_temp_folder_either():
    with tempfile.TemporaryDirectory() as outside:
        victim = os.path.join(outside, 'victim.mot')
        with open(victim, 'w') as fh:
            fh.write('keep me')
        _post(victim, path='/api/analyze/batch', field='files')
        assert open(victim).read() == 'keep me'


def test_each_request_cleans_up_after_itself():
    before = set(os.listdir(main.TMP_DIR))
    _post('swing.mot')
    _post('swing.mot', path='/api/analyze/batch', field='files')
    assert set(os.listdir(main.TMP_DIR)) <= before


def test_an_oversized_upload_is_refused():
    old = main.MAX_UPLOAD_BYTES
    main.MAX_UPLOAD_BYTES = 1000
    try:
        r = _post('big.mot', data=b'x' * 5000)
    finally:
        main.MAX_UPLOAD_BYTES = old
    assert r.status_code == 413, (r.status_code, r.text[:200])


if __name__ == '__main__':
    n = 0
    for k, fn in list(globals().items()):
        if k.startswith('test_'):
            fn(); n += 1
    print(f'{n} upload safety tests passed')
