from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import os

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def get_drive_service():
    creds = None
    if os.path.exists("tokens/drive_token.json"):
        creds = Credentials.from_authorized_user_file("tokens/drive_token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        os.makedirs("tokens", exist_ok=True)
        with open("tokens/drive_token.json", "w") as token:
            token.write(creds.to_json())
    return build("drive", "v3", credentials=creds)


def upload_to_drive(file_path: str, folder_id: str = None):
    """파일을 Google Drive에 업로드하고 공유 링크 반환"""
    service = get_drive_service()

    file_metadata = {"name": os.path.basename(file_path)}
    if folder_id:
        file_metadata["parents"] = [folder_id]

    media = MediaFileUpload(file_path, resumable=True)
    uploaded = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id")
        .execute()
    )

    file_id = uploaded.get("id")
    # 링크 공개 설정
    service.permissions().create(
        fileId=file_id, body={"role": "reader", "type": "anyone"}
    ).execute()
    link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
    return link
