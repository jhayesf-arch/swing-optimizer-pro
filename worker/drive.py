"""Google Drive access for the worker.

Two ways to sign in (see worker/README.md):
- Your own Google account (the usual choice, and the only one that works on a
  personal Gmail Drive): run `python -m worker.auth` once on your computer and set
  GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET and GOOGLE_OAUTH_REFRESH_TOKEN.
- A service account, for a Google Workspace shared drive only: service accounts have
  no storage of their own, so they cannot upload into a personal My Drive. Set
  GOOGLE_SERVICE_ACCOUNT_JSON (or GOOGLE_APPLICATION_CREDENTIALS) and add the
  account to the shared drive.
"""
from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

FOLDER = 'application/vnd.google-apps.folder'
GDOC = 'application/vnd.google-apps.document'
# The raw phone videos and OpenPose pickles are large and the worker does not need them:
# reports use the .trc/.mot/.osim and videos use the synced *_sync.mp4 files.
SKIP_SUFFIXES = ('.mov', '.pkl', '.pickle', '.avi')
SCOPES = ['https://www.googleapis.com/auth/drive']
LIST_KW = {'supportsAllDrives': True, 'includeItemsFromAllDrives': True}


def credentials():
    if os.environ.get('GOOGLE_OAUTH_REFRESH_TOKEN'):
        from google.oauth2.credentials import Credentials
        missing = [k for k in ('GOOGLE_OAUTH_CLIENT_ID', 'GOOGLE_OAUTH_CLIENT_SECRET') if not os.environ.get(k)]
        if missing:
            raise RuntimeError('also set ' + ' and '.join(missing))
        return Credentials(None, refresh_token=os.environ['GOOGLE_OAUTH_REFRESH_TOKEN'],
                           token_uri='https://oauth2.googleapis.com/token',
                           client_id=os.environ['GOOGLE_OAUTH_CLIENT_ID'],
                           client_secret=os.environ['GOOGLE_OAUTH_CLIENT_SECRET'], scopes=SCOPES)
    from google.oauth2 import service_account
    raw = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON')
    if raw:
        creds = service_account.Credentials.from_service_account_info(json.loads(raw), scopes=SCOPES)
    else:
        path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if not path:
            raise RuntimeError('no Google sign-in configured: run `python -m worker.auth` and set the '
                               'GOOGLE_OAUTH_* variables it prints (see worker/README.md)')
        creds = service_account.Credentials.from_service_account_file(path, scopes=SCOPES)
    return creds


def service():
    from googleapiclient.discovery import build
    return build('drive', 'v3', credentials=credentials(), cache_discovery=False)


def children(svc, folder_id: str, folders_only: bool = False) -> List[Dict]:
    q = f"'{folder_id}' in parents and trashed = false"
    if folders_only:
        q += f" and mimeType = '{FOLDER}'"
    out, token = [], None
    while True:
        resp = svc.files().list(q=q, pageToken=token, pageSize=200,
                                fields='nextPageToken, files(id, name, mimeType, modifiedTime, createdTime, size)',
                                **LIST_KW).execute()
        out += resp.get('files', [])
        token = resp.get('nextPageToken')
        if not token:
            return out


def newest_change(svc, folder_id: str) -> Optional[datetime]:
    """Latest change anywhere under the folder: used to wait until an upload has finished.

    createdTime counts too, since a synced upload can keep the file's original modified time."""
    latest = None
    for f in children(svc, folder_id):
        t = max(datetime.fromisoformat(f[k].replace('Z', '+00:00'))
                for k in ('modifiedTime', 'createdTime') if f.get(k))
        if f['mimeType'] == FOLDER:
            t = max(t, newest_change(svc, f['id']) or t)
        latest = t if latest is None or t > latest else latest
    return latest


def download_tree(svc, folder_id: str, dest: str) -> int:
    """Copy a folder to dest, skipping files the worker does not need. Returns files written."""
    from googleapiclient.http import MediaIoBaseDownload

    os.makedirs(dest, exist_ok=True)
    n = 0
    for f in children(svc, folder_id):
        path = os.path.join(dest, f['name'])
        if f['mimeType'] == FOLDER:
            n += download_tree(svc, f['id'], path)
            continue
        if f['name'].lower().endswith(SKIP_SUFFIXES):
            continue
        if f['mimeType'] == GDOC:                      # e.g. an "athlete" Google Doc
            req = svc.files().export_media(fileId=f['id'], mimeType='text/plain')
            path += '.txt'
        elif f['mimeType'].startswith('application/vnd.google-apps'):
            continue
        else:
            req = svc.files().get_media(fileId=f['id'], supportsAllDrives=True)
        with io.FileIO(path, 'wb') as fh:
            dl = MediaIoBaseDownload(fh, req, chunksize=16 * 1024 * 1024)
            done = False
            while not done:
                _, done = dl.next_chunk()
        n += 1
    return n


def find_child(svc, parent_id: str, name: str) -> Optional[Dict]:
    for f in children(svc, parent_id):
        if f['name'] == name:
            return f
    return None


def ensure_folder(svc, parent_id: str, name: str) -> str:
    f = find_child(svc, parent_id, name)
    if f and f['mimeType'] == FOLDER:
        return f['id']
    body = {'name': name, 'mimeType': FOLDER, 'parents': [parent_id]}
    return svc.files().create(body=body, fields='id', supportsAllDrives=True).execute()['id']


def upload(svc, path: str, parent_id: str, mime: str, as_google_doc: bool = False,
           name: Optional[str] = None) -> str:
    """Upload a file, replacing one of the same name in that folder."""
    from googleapiclient.http import MediaFileUpload

    name = name or os.path.basename(path)
    old = find_child(svc, parent_id, name)
    media = MediaFileUpload(path, mimetype=mime, resumable=os.path.getsize(path) > 5 * 1024 * 1024)
    if old:
        return svc.files().update(fileId=old['id'], media_body=media, fields='id',
                                  supportsAllDrives=True).execute()['id']
    body = {'name': name, 'parents': [parent_id]}
    if as_google_doc:
        body['mimeType'] = GDOC
    return svc.files().create(body=body, media_body=media, fields='id',
                              supportsAllDrives=True).execute()['id']


def now() -> datetime:
    return datetime.now(timezone.utc)
