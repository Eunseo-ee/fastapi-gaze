from fastapi import FastAPI, Form
import requests, subprocess, os

app = FastAPI()

@app.post("/process")
def process_video(driveLink: str = Form(...)):
    # 1. Drive 영상 다운로드
    file_id = driveLink.split("/d/")[1].split("/")[0]
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    video_path = "input.mp4"
    r = requests.get(url)
    open(video_path, "wb").write(r.content)

    # 2. 모델 실행
    subprocess.run([
        "python", "gaze_tracking.py",
        "--ckpt", "model.ckpt",
        "--obj", "obj.pt",
        "--head", "head.pt",
        "--video", video_path
    ])

    # 3. 결과 반환
    result_path = "out/output.mp4"
    return {"result_path": result_path}
