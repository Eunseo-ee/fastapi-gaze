from fastapi import FastAPI, Form
import requests, subprocess, os

app = FastAPI()

class VideoRequest(BaseModel):
    drive_link: str  # 요청 본문에서 JSON으로 받음

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/process")
def process_video(req: VideoRequest):
    try:
        # 1️⃣ 구글드라이브 링크에서 파일 ID 추출
        file_id = req.drive_link.split("/d/")[1].split("/")[0]
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        video_path = "input.mp4"

        # 2️⃣ 파일 다운로드
        r = requests.get(url)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail="Drive 파일 다운로드 실패")
        with open(video_path, "wb") as f:
            f.write(r.content)

        # 3️⃣ run_from_video.py 실행
        cmd = [
            "python", "run_from_video.py",
            "--ckpt", "model.ckpt",
            "--obj", "obj.pt",
            "--head", "head.pt",
            "--video", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)

        # 4️⃣ 결과 확인
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)

        result_path = "out/input__vis.mp4"
        ranking_path = "out/input_ranking.txt"

        if not os.path.exists(result_path):
            raise HTTPException(status_code=500, detail="출력 영상 없음")

        # (선택) 구글 드라이브 업로드 함수가 있다면 여기서 호출
        # drive_link = upload_to_drive(result_path)

        # (선택) ranking.txt를 읽어서 응답에 포함
        ranking_data = None
        if os.path.exists(ranking_path):
            with open(ranking_path, "r", encoding="utf-8") as f:
                ranking_data = f.read()

        return {
            "status": "success",
            "result_path": result_path,
            "ranking_path": ranking_path,
            "ranking_text": ranking_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))