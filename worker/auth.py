"""One-time Google sign-in for the worker. Run on your own computer, not on the server:

    pip install google-auth-oauthlib
    python -m worker.auth path/to/client_secret.json

It opens a browser, you sign in with the Google account that owns the Drive folder,
and it prints the three GOOGLE_OAUTH_* values to put in the worker's environment.
"""
import json
import sys

from .drive import SCOPES


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(sys.argv[1], SCOPES)
    creds = flow.run_local_server(port=0, access_type='offline', prompt='consent')
    client = json.load(open(sys.argv[1]))
    client = client.get('installed') or client.get('web') or {}
    print('\nSet these on the worker (keep them secret):\n')
    print(f"GOOGLE_OAUTH_CLIENT_ID={client.get('client_id')}")
    print(f"GOOGLE_OAUTH_CLIENT_SECRET={client.get('client_secret')}")
    print(f'GOOGLE_OAUTH_REFRESH_TOKEN={creds.refresh_token}')


if __name__ == '__main__':
    main()
