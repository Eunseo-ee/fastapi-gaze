from fastapi import FastAPI, Form
from pydantic import BaseModel
import requests, subprocess, os, json, tempfile

app = FastAPI()


class VideoRequest(BaseModel):
    drive_link: str  # 요청 본문에서 JSON으로 받음

@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/process")
def process_video(driveLink: str = Form(...)):
    # 1️⃣ Google Drive에서 영상 다운로드
    file_id = driveLink.split("/d/")[1].split("/")[0]
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    video_path = "input.mp4"

    r = requests.get(url)
    with open(video_path, "wb") as f:
        f.write(r.content)

    # 2️⃣ 실행 경로 기준 절대 경로 설정
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    ckpt_path = os.path.join(BASE_DIR, "model.ckpt")
    obj_path = os.path.join(BASE_DIR, "obj.pt")
    head_path = os.path.join(BASE_DIR, "head.pt")

    # 3️⃣ run_from_video.py 실행
    command = [
        "python", os.path.join(BASE_DIR, "run_from_video.py"),
        "--ckpt", ckpt_path,
        "--obj", obj_path,
        "--head", head_path,
        "--video", video_path
    ]

    process = subprocess.run(command, capture_output=True, text=True)
    stdout = process.stdout.strip()
    stderr = process.stderr.strip()

    print("[FastAPI] run_from_video.py stderr:\n", stderr)

    # 4️⃣ JSON 결과 파싱
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "error": "Failed to parse JSON output",
            "stdout": stdout,
            "stderr": stderr
        }

    # 5️⃣ 결과 반환 (Google Drive 업로드 결과 포함)
    return result