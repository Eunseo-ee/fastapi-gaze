from google_auth_oauthlib.flow import InstalledAppFlow

flow = InstalledAppFlow.from_client_secrets_file(
    'client_secret.json',
    scopes=['https://www.googleapis.com/auth/drive.file']
)
creds = flow.run_local_server(port=8080)

with open('drive_token.json', 'w') as token:
    token.write(creds.to_json())
