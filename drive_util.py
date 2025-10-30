from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import os

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def get_drive_service():
    """로컬 또는 Render 환경에 따라 Drive 인증 자동 처리"""
    creds = None

    # 1️⃣ 토큰 파일이 있으면 우선 사용
    if os.path.exists(TOKENS_PATH):
        creds = Credentials.from_authorized_user_file(TOKENS_PATH, SCOPES)

    # 2️⃣ 토큰이 없거나 만료된 경우 (로컬일 때만 새로 로그인)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Render는 여기서 인증창이 안 뜨므로 실행 안 됨 (로컬에서만 작동)
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)

        # 토큰 저장 (로컬용)
        if not os.path.exists("/etc/secrets"):
            os.makedirs("tokens", exist_ok=True)
            with open("tokens/drive_token.json", "w") as token:
                token.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def upload_to_drive(file_path: str, folder_id: str = None):
    """Google Drive에 파일 업로드 후 공유 링크 반환"""
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

    # 공개 링크로 전환
    service.permissions().create(
        fileId=file_id, body={"role": "reader", "type": "anyone"}
    ).execute()

    return f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
