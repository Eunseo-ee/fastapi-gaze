from fastapi import FastAPI, Form
import requests, subprocess, os, json, tempfile

app = FastAPI()


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/process")
def process_video(driveLink: str = Form(...)):
    # 1️⃣ Google Drive 영상 다운로드
    file_id = driveLink.split("/d/")[1].split("/")[0]
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(requests.get(url).content)
        video_path = tmp.name

    # 2️⃣ 모델 실행 (stdout으로 결과 JSON 출력)
    process = subprocess.Popen(
        [
            "python",
            "run_from_video.py",
            "--ckpt",
            "model.ckpt",
            "--obj",
            "obj.pt",
            "--head",
            "head.pt",
            "--video",
            video_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout, stderr = process.communicate()

    # 3️⃣ 결과 파싱
    try:
        result = json.loads(stdout.decode("utf-8"))
    except Exception as e:
        result = {"error": str(e), "stderr": stderr.decode("utf-8")}

    # 4️⃣ 임시 파일 정리
    os.remove(video_path)

    # 5️⃣ 응답 반환
    return result
