import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
from typing import List, Tuple, Optional, Dict
import torchvision.transforms.functional as TF
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
import sys


from src.modeling.sharingan import Sharingan
from src.utils.common import spatial_argmax2d, square_bbox


DET_THR = 0.2  # head detection threshold
CONF_THR = 0.25
IOU_THR = 0.45
NUM_CLASSES = 6  # 0~5만 사용(필요 없으면 None)

# CKPT_PATH = "checkpoints/best_gaze_synth_experiment_3.ckpt" # Sharingan ckpt 경로
# OBJ_MODEL_PATH = "weights/detect_experiment_object.pt"    # 물체 YOLO 가중치
# HEAD_MODEL_PATH = "weights/experiment_head.pt"
# VIDEO_PATH = "./data/20250630_160223.mp4"


IMG_MEAN = [0.461037, 0.486372, 0.471720]
IMG_STD = [0.212238, 0.208506, 0.205668]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLOR_OBJ = (0, 200, 0)  # 초록
COLOR_HIT = (0, 255, 255)  # 노랑
COLOR_HEAD = (255, 180, 80)  # 하늘/주황톤
COLOR_GAZE = (0, 0, 255)  # 빨강

# run_from_video.py 맨 위 (import들 밑)
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)


# ========================= UTILITY FUNCTIONS =========================== #
def resolve_path(p: str | Path) -> Path:
    """
    Path만 사용해 경로를 절대경로로 변환.
    - 절대경로면 그대로 반환
    - 상대경로면 현재 작업 디렉터리(Path.cwd()) 기준으로 절대화
    """
    p = Path(p).expanduser()
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def expand_bbox(bbox, img_w, img_h, k=0.1):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    bbox[0] = max(0, bbox[0] - k * w)
    bbox[1] = max(0, bbox[1] - k * h)
    bbox[2] = min(img_w, bbox[2] + k * w)
    bbox[3] = min(img_h, bbox[3] + k * h)
    return bbox


def expand_xyxy_rel_img(
    xyxy: np.ndarray, img_w: int, img_h: int, kx: float = 0.02, ky: float = 0.02
) -> np.ndarray:
    """
    xyxy: (N,4) [x1,y1,x2,y2] float32
    img_w, img_h: 이미지 크기
    kx, ky: 이미지 폭/높이에 대한 비율로 확장 (예: 0.02 -> 양쪽으로 각각 이미지의 2%씩)
    """
    if xyxy.size == 0:
        return xyxy
    dx = kx * img_w
    dy = ky * img_h
    out = xyxy.copy()
    out[:, 0] = np.maximum(0.0, out[:, 0] - dx)
    out[:, 1] = np.maximum(0.0, out[:, 1] - dy)
    out[:, 2] = np.minimum(img_w - 1.0, out[:, 2] + dx)
    out[:, 3] = np.minimum(img_h - 1.0, out[:, 3] + dy)
    return out.astype(np.float32)


def load_obj_detection_model(device, weights_path: Optional[str] = None):
    model = YOLO(weights_path)
    model = model.to(device)
    model.eval()
    return model


def load_head_detection_model(device, weights_path: Optional[str] = None):
    model = YOLO(weights_path)
    model = model.to(device)
    model.eval()
    return model


def detect_heads(image, model):
    image_cv = np.array(image)[..., ::-1]
    results = model(
        image_cv, imgsz=640, conf=0.25, iou=0.45, classes=[0], amp=False, verbose=False
    )
    boxes = results[0].boxes.xyxy.cpu().numpy()  # (N, 4)
    confs = results[0].boxes.conf.cpu().numpy()  # (N,)
    detections = np.hstack([boxes, confs[:, None]])  # (N, 5)
    return detections


def load_sharingan_model(ckpt_path, device):
    sharingan = Sharingan(
        patch_size=16,
        token_dim=768,
        image_size=224,
        gaze_feature_dim=512,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_global_tokens=0,
        encoder_mlp_ratio=4.0,
        encoder_use_qkv_bias=True,
        encoder_drop_rate=0.0,
        encoder_attn_drop_rate=0.0,
        encoder_drop_path_rate=0.0,
        decoder_feature_dim=128,
        decoder_hooks=[2, 5, 8, 11],
        decoder_hidden_dims=[48, 96, 192, 384],
        decoder_use_bn=True,
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    checkpoint = {
        name.replace("model.", ""): value
        for name, value in checkpoint["state_dict"].items()
    }
    sharingan.load_state_dict(checkpoint, strict=True)
    sharingan.eval()
    sharingan.to(device)
    return sharingan


def rank_interest_scores(
    interest: Dict[int, int],
    class_names: Optional[List[str]] = None,
    top_k: Optional[int] = None,
) -> List[Tuple[int, int, str]]:
    """
    관심도 dict를 점수 내림차순으로 정렬하여 [(cls_id, score, name)] 리스트 반환
    """
    ranked = sorted(interest.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_k is not None:
        ranked = ranked[:top_k]
    out = []
    for cid, score in ranked:
        name = (
            class_names[cid] if (class_names and 0 <= cid < len(class_names)) else "-"
        )
        out.append((cid, score, name))
    return out


def predict_gaze(image, sharingan, head_detector, tracker=None):
    # 1. Convert image
    image_np = np.array(image)
    img_h, img_w, _ = image_np.shape

    raw_detections = detect_heads(image_np, head_detector)
    detections = []
    for raw in raw_detections:
        bbox, conf = raw[:4], raw[4]
        if conf > DET_THR:
            bbox = expand_bbox(bbox, img_w, img_h, k=0.2)
            cls_ = np.array([0.0])
            detection = np.concatenate(
                [bbox, conf[None], cls_]
            )  # [x1,y1,x2,y2,conf,class_id]
            detections.append(detection)

    if len(detections) == 0:
        return (
            torch.tensor([]),
        ) * 6  # gaze_points, gaze_vecs, inouts, head_bboxes, gaze_heatmaps, pids

    detections = np.stack(detections)
    if tracker is not None:
        tracks = tracker.update(detections, image_np)
        if len(tracks) == 0:
            return (torch.tensor([]),) * 6
        pids = (tracks[:, 4] - 1).astype(int)
        head_bboxes = torch.from_numpy(tracks[:, :4]).float()
    else:
        head_bboxes = torch.from_numpy(detections[:, :4]).float()
        pids = np.arange(len(head_bboxes))

    t_head_bboxes = square_bbox(head_bboxes, img_w, img_h)

    # 3. Extract and transform heads
    heads = []
    for bbox in t_head_bboxes:
        head = TF.resize(
            TF.to_tensor(image.crop(bbox.numpy())), (224, 224), antialias=True
        )
        heads.append(head)
    heads = torch.stack(heads)
    heads = TF.normalize(heads, mean=IMG_MEAN, std=IMG_STD)

    # 4. Transform Image
    image = TF.to_tensor(image)
    image = TF.resize(image, (224, 224), antialias=True)
    image = TF.normalize(image, mean=IMG_MEAN, std=IMG_STD)

    # 5. Normalize head bboxes
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    t_head_bboxes /= scale

    # 6. build input sample
    sample = {
        "image": image.unsqueeze(0).to(DEVICE),
        "heads": heads.unsqueeze(0).to(DEVICE),
        "head_bboxes": t_head_bboxes.unsqueeze(0).to(DEVICE),
    }

    # 7. predict gaze
    with torch.no_grad():
        gaze_vecs, gaze_heatmaps, inouts = sharingan(sample)
        gaze_heatmaps = gaze_heatmaps.squeeze(0).cpu()
        gaze_vecs = gaze_vecs.squeeze(0).cpu()
        gaze_points = spatial_argmax2d(gaze_heatmaps, normalize=True)
        inouts = torch.sigmoid(inouts.squeeze(0)).flatten().cpu()

    return gaze_points, gaze_vecs, inouts, head_bboxes, gaze_heatmaps, pids


def point_in_any_box(pt: Tuple[float, float], boxes_xyxy: np.ndarray):
    """포인트가 포함된 박스 인덱스(여럿이면 더 작은 박스 우선). 없으면 None."""
    if boxes_xyxy.size == 0:
        return None
    x, y = pt
    inside = (
        (boxes_xyxy[:, 0] <= x)
        & (x <= boxes_xyxy[:, 2])
        & (boxes_xyxy[:, 1] <= y)
        & (y <= boxes_xyxy[:, 3])
    )
    idxs = np.where(inside)[0]
    if len(idxs) == 0:
        return None
    areas = (boxes_xyxy[idxs, 2] - boxes_xyxy[idxs, 0]) * (
        boxes_xyxy[idxs, 3] - boxes_xyxy[idxs, 1]
    )
    return idxs[np.argmin(areas)]


# ========================= 비디오 처리 유틸 ========================= #


def safe_open_video_writer(
    out_path: str, fps: float, size: tuple[int, int]
) -> cv2.VideoWriter:
    W, H = size
    ext = Path(out_path).suffix.lower()

    # 1) 우선 컨테이너에 맞는 코덱 시도
    if ext == ".mp4":
        candidates = [
            ("mp4v", ".mp4"),  # 가장 무난
            ("avc1", ".mp4"),  # H.264 (인코더 필요)
            ("H264", ".mp4"),  # 같은 계열
        ]
    else:
        candidates = [
            ("XVID", ".avi"),
            ("MJPG", ".avi"),
        ]

    for fourcc_tag, new_ext in candidates:
        # 확장자 보정
        path_try = str(Path(out_path).with_suffix(new_ext))
        fourcc = cv2.VideoWriter_fourcc(*fourcc_tag)
        vw = cv2.VideoWriter(path_try, fourcc, max(fps, 1.0), (W, H))
        if vw.isOpened():
            if path_try != out_path:
                print(
                    f"[INFO] changed output to {path_try} (codec={fourcc_tag})",
                    file=sys.stderr,
                )
            return vw

    for fourcc_tag in ["XVID", "MJPG"]:
        path_try = str(Path(out_path).with_suffix(".avi"))
        fourcc = cv2.VideoWriter_fourcc(*fourcc_tag)
        vw = cv2.VideoWriter(path_try, fourcc, max(fps, 1.0), (W, H))
        if vw.isOpened():
            print(
                f"[INFO] fallback to {path_try} (codec={fourcc_tag})", file=sys.stderr
            )
            return vw

    raise RuntimeError(
        "Failed to open VideoWriter with common codecs. "
        "Install FFmpeg with x264/x265 or use AVI/MJPG."
    )


def run_object_detector_on_frame(
    model: YOLO,
    frame_bgr: np.ndarray,
    conf: float = CONF_THR,
    iou: float = IOU_THR,
    num_classes: Optional[int] = NUM_CLASSES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    단일 프레임에서 물체 탐지.
    Returns:
        xyxy: (N,4) float32
        cls:  (N,)  int64
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise RuntimeError("frame_bgr cannot be None")

    # Ultralytics는 ndarray도 입력 가능 (BGR OK)
    res = model.predict(frame_bgr, conf=conf, iou=iou, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)

    xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
    cls = res.boxes.cls.cpu().numpy().astype(np.int64)
    if num_classes is not None:
        keep = (cls >= 0) & (cls < num_classes)
        xyxy, cls = xyxy[keep], cls[keep]
    return xyxy, cls


def get_pred_gaze_points_pixels_from_frame(
    frame_bgr: np.ndarray,
    sharingan,
    head_detector,
    filter_by_inout: bool = False,
    io_thr: float = 0.5,
) -> Tuple[np.ndarray, int]:
    """
    프레임 1장에 대해 Sharingan으로 예측한 모든 시선 점을 '픽셀 좌표'로 반환.
    Returns:
        pts_pix: (M, 2) float32, 각 head의 (x,y) in pixels
        num_heads: 감지된 head 수
    """
    if frame_bgr is None:
        raise ValueError("frame_bgr cannot be None")

    H, W = frame_bgr.shape[:2]
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    out = predict_gaze(img_pil, sharingan, head_detector, tracker=None)
    gaze_points, gaze_vecs, inouts, head_bboxes, gaze_heatmaps, pids = out

    if gaze_points is None or len(gaze_points) == 0:
        return np.zeros((0, 2), dtype=np.float32), 0

    gp = (
        gaze_points.cpu().numpy()
        if hasattr(gaze_points, "cpu")
        else np.array(gaze_points)
    )
    pts_pix = gp * np.array([W, H], dtype=np.float32)

    if filter_by_inout and (inouts is not None) and (len(inouts) == len(pts_pix)):
        io = inouts.cpu().numpy() if hasattr(inouts, "cpu") else np.array(inouts)
        keep = io > io_thr
        pts_pix = pts_pix[keep]

    return pts_pix.astype(np.float32), (0 if head_bboxes is None else len(head_bboxes))


def get_pred_gaze_points_and_heads_from_frame(
    frame_bgr: np.ndarray,
    sharingan,
    head_detector,
    filter_by_inout: bool = False,
    io_thr: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        gaze_pts_pix: (K,2) float32  - 픽셀 좌표 (x,y)
        head_xyxy:    (M,4) int32    - 머리 박스 (픽셀)
    """
    H, W = frame_bgr.shape[:2]
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    gaze_points, gaze_vecs, inouts, head_bboxes, gaze_heatmaps, pids = predict_gaze(
        img_pil, sharingan, head_detector, tracker=None
    )

    # (1) gaze 픽셀 좌표
    if gaze_points is None or len(gaze_points) == 0:
        gaze_pts_pix = np.zeros((0, 2), dtype=np.float32)
    else:
        gp = (
            gaze_points.cpu().numpy()
            if hasattr(gaze_points, "cpu")
            else np.array(gaze_points)
        )
        gaze_pts_pix = gp * np.array([W, H], dtype=np.float32)
        if (
            filter_by_inout
            and (inouts is not None)
            and (len(inouts) == len(gaze_pts_pix))
        ):
            io = inouts.cpu().numpy() if hasattr(inouts, "cpu") else np.array(inouts)
            gaze_pts_pix = gaze_pts_pix[io > io_thr]

    # (2) head 박스 (픽셀)
    if head_bboxes is None or (
        hasattr(head_bboxes, "numel") and head_bboxes.numel() == 0
    ):
        head_xyxy = np.zeros((0, 4), dtype=np.int32)
    else:
        if hasattr(head_bboxes, "cpu"):
            head_xyxy = head_bboxes.cpu().numpy()
        else:
            head_xyxy = np.array(head_bboxes)
        head_xyxy = head_xyxy.astype(np.int32)

    return gaze_pts_pix.astype(np.float32), head_xyxy


def accumulate_interest_from_video(
    video_path: str,
    obj_model: YOLO,
    head_model: YOLO,
    sharingan: Sharingan,
    expand_kx: float = 0.01,
    expand_ky: float = 0.01,
    frame_stride: int = 1,  # 매 프레임이면 1, 2면 1/2 샘플링
    max_frames: Optional[int] = None,
    filter_by_inout: bool = False,
    io_thr: float = 0.5,
) -> Tuple[Dict[int, int], Dict[str, int]]:
    """
    동영상을 처음부터 끝까지 돌며:
      - obj_model로 프레임의 물체 탐지 (박스 확장)
      - head_model + sharingan으로 모든 head의 gaze 픽셀 좌표 추정
      - 각 gaze 점이 포함된 물체 박스의 '클래스' 관심도 +1
    Returns:
        interest: {class_id: score}
        stats: {'n_frames':X, 'n_proc_frames':Y, 'n_heads':Z, 'n_hits':K}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    interest: Dict[int, int] = defaultdict(int)
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        raise RuntimeError(f"Frame count is zero")
    n_proc = 0
    n_heads = 0
    n_hits = 0

    processed_frames = 0
    frame_idx = 0

    # 비디오 총 프레임 수 가져오기
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames), file=sys.stderr):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1

        # 프레임 스킵(샘플링)
        if frame_stride > 1 and (frame_idx % frame_stride != 1):
            continue

        H, W = frame_bgr.shape[:2]

        # (1) obj detect
        obj_xyxy, obj_cls = run_object_detector_on_frame(
            obj_model, frame_bgr, conf=CONF_THR, iou=IOU_THR, num_classes=NUM_CLASSES
        )
        if obj_xyxy.size > 0:
            obj_xyxy = expand_xyxy_rel_img(obj_xyxy, W, H, kx=expand_kx, ky=expand_ky)

        # (2) gaze predict
        pts_pix, heads_here = get_pred_gaze_points_pixels_from_frame(
            frame_bgr=frame_bgr,
            sharingan=sharingan,
            head_detector=head_model,
            filter_by_inout=filter_by_inout,
            io_thr=io_thr,
        )

        n_proc += 1
        n_heads += int(heads_here)

        # (3) 관심도 누적
        if obj_xyxy.size > 0 and pts_pix.size > 0:
            for pt in pts_pix:
                idx = point_in_any_box((float(pt[0]), float(pt[1])), obj_xyxy)
                if idx is None:
                    continue
                cls_id = int(obj_cls[idx])
                if (NUM_CLASSES is None) or (0 <= cls_id < NUM_CLASSES):
                    interest[cls_id] += 1
                    n_hits += 1

        processed_frames += 1
        if (max_frames is not None) and (processed_frames >= max_frames):
            break

    cap.release()

    stats = {
        "n_frames": total_frames,
        "n_proc_frames": n_proc,
        "n_heads": n_heads,
        "n_hits": n_hits,
    }
    return dict(interest), stats


# 메타/랭킹/저장 유틸


def write_ranking_txt(
    out_path: str,
    ranked: List[Tuple[int, int, str]],  # [(cls_id, score, name), ...]
    class_meta: Optional[Dict[int, Dict[str, str]]] = None,
    default_name_from_id: bool = True,
    add_unranked_from_meta: bool = True,  # ← 추가: meta에만 있는 클래스도 출력
    unranked_sort: str = "id",  # "id" 또는 "name"
) -> None:
    """
    포맷: "순위, 제품명, 카테고리, 가격"
    - ranked 항목: 1부터 숫자 순위 부여
    - class_meta에는 있으나 ranked엔 없는 항목: 순위 "-"로 출력 (숫자 순위 뒤에 이어서)
    """
    lines = []

    # 1) ranked 먼저 (숫자 순위)
    present_cids = set()
    for rank, (cid, score, name) in enumerate(ranked, start=1):
        present_cids.add(cid)
        if class_meta and cid in class_meta:
            pname = class_meta[cid].get(
                "name", name if default_name_from_id else str(cid)
            )
            cat = class_meta[cid].get("category", "-")
            price = class_meta[cid].get("price", "-")
        else:
            pname = name if default_name_from_id else str(cid)
            cat = "-"
            price = "-"
        lines.append(f"{rank},{pname},{cat},{price},{score}")

    # 2) meta에만 있는 나머지 (순위 "-")
    if add_unranked_from_meta and class_meta:
        # 누락된 cid 수집
        missing = [cid for cid in class_meta.keys() if cid not in present_cids]

        # 정렬 방식 선택
        if unranked_sort == "name":
            missing.sort(key=lambda cid: class_meta[cid].get("name", ""))
        else:  # "id"
            missing.sort()

        for cid in missing:
            rec = class_meta[cid]
            pname = rec.get("name", str(cid) if default_name_from_id else str(cid))
            cat = rec.get("category", "-")
            price = rec.get("price", "-")
            lines.append(f"-, {pname}, {cat}, {price},0")

    # 저장
    out_dir = str(Path(out_path).parent)
    if out_dir and out_dir not in (".", ""):
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[SAVE] ranking saved to: {out_path}", file=sys.stderr)


import os
from pathlib import Path

# --- 경로 자동 감지 (로컬 or Render) ---
CREDENTIALS_PATH = (
    "/etc/secrets/credentials.json"
    if os.path.exists("/etc/secrets/credentials.json")
    else "credentials.json"
)

TOKENS_PATH = (
    "/etc/secrets/drive_token.json"
    if os.path.exists("/etc/secrets/drive_token.json")
    else "tokens/drive_token.json"
)


def upload_to_drive(file_path, mime_type="video/mp4", folder_id=None):
    """
    Google Drive에 파일 업로드 후 (webContentLink, webViewLink) 반환
    """
    scopes = ["https://www.googleapis.com/auth/drive.file"]

    # ✅ 경로 수정: Render 환경에서는 /etc/secrets/drive_token.json 사용
    creds = Credentials.from_authorized_user_file(TOKENS_PATH, scopes)
    service = build("drive", "v3", credentials=creds)

    file_metadata = {"name": Path(file_path).name}
    if folder_id:
        file_metadata["parents"] = [folder_id]

    media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
    uploaded = (
        service.files()
        .create(
            body=file_metadata,
            media_body=media,
            fields="id, webContentLink, webViewLink",
        )
        .execute()
    )

    return uploaded["webContentLink"], uploaded["webViewLink"]


# ========================= 비디오 시각화 & 저장 =========================


def _put_filled_text(
    img, text, org, fs=0.5, color=(255, 255, 255), bg=(0, 0, 0), thickness=1, pad=3
):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, thickness)
    x, y = org
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + pad), bg, -1)
    cv2.putText(
        img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, thickness, cv2.LINE_AA
    )


def build_filename(date, start, end):
    # "18:00" → "1800"
    start_sanitized = start.replace(":", "")
    end_sanitized = end.replace(":", "")
    return f"{date}_{start_sanitized}~{end_sanitized}"


def draw_annotated_frame(
    frame_bgr: np.ndarray,
    obj_xyxy: np.ndarray,  # (N,4) float32  (확장 적용 후)
    obj_cls: Optional[np.ndarray],  # (N,)  int64
    head_xyxy: np.ndarray,  # (M,4) int32
    gaze_pts: np.ndarray,  # (K,2) float32
    class_names: Optional[List[str]] = None,
    thickness_obj: int = 2,
    thickness_head: int = 2,
    gaze_radius: int = 4,
) -> np.ndarray:
    """
    - 객체 박스: 기본 '초록' / 시선 점이 들어간 박스는 '노랑'으로 덮어 그림
    - 머리 박스: 하늘색
    - 시선 점: 빨강 원
    """
    vis = frame_bgr.copy()

    # 어떤 obj 박스가 하나라도 시선을 포함했는지 계산
    hit_mask = (
        np.zeros(len(obj_xyxy), dtype=bool)
        if obj_xyxy.size > 0
        else np.zeros((0,), dtype=bool)
    )
    if obj_xyxy.size > 0 and gaze_pts.size > 0:
        gx = gaze_pts[:, 0][:, None]
        gy = gaze_pts[:, 1][:, None]
        inside = (
            (gx >= obj_xyxy[:, 0])
            & (gx <= obj_xyxy[:, 2])
            & (gy >= obj_xyxy[:, 1])
            & (gy <= obj_xyxy[:, 3])
        )
        hit_mask = inside.any(axis=0)

    # 1) 객체 박스
    for i, box in enumerate(obj_xyxy):
        x1, y1, x2, y2 = map(int, box)
        color = COLOR_HIT if (i < len(hit_mask) and hit_mask[i]) else COLOR_OBJ
        thick = max(thickness_obj, 3) if color == COLOR_HIT else thickness_obj
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
        if obj_cls is not None and i < len(obj_cls):
            cid = int(obj_cls[i])
            label = f"{cid}"
            if class_names and (0 <= cid < len(class_names)):
                label = class_names[cid]
            _put_filled_text(
                vis, label, (x1 + 3, max(15, y1 - 4)), fs=0.5, bg=(30, 30, 30)
            )

    # 2) 머리 박스
    for hb in head_xyxy:
        x1, y1, x2, y2 = map(int, hb)
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_HEAD, thickness_head)

    # 3) 시선 점
    for gx, gy in gaze_pts:
        cv2.circle(vis, (int(gx), int(gy)), gaze_radius, COLOR_GAZE, -1)

    return vis


import subprocess
import shutil


def convert_avi_to_mp4(avi_path: str) -> str:
    """
    FFmpeg로 AVI(MJPG)를 MP4(H.264)로 변환한다.
    avi_path: 원본 avi 파일 경로
    Returns: 변환된 mp4 파일 경로
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "FFmpeg가 설치되어 있지 않습니다. 서버 환경에 ffmpeg 설치가 필요합니다."
        )

    mp4_path = avi_path.replace(".avi", ".mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        avi_path,
        "-vcodec",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        mp4_path,
    ]

    # FFmpeg 실행
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return mp4_path


def process_video_once_and_export(
    video_path: str,
    obj_model: YOLO,
    head_model: YOLO,
    sharingan: Sharingan,
    class_meta: Dict[int, Dict[str, str]],
    date: str,
    start: str,
    end: str,
    out_video_path: Optional[str] = None,  # None이면 입력명__vis.mp4
    out_ranking_txt: Optional[str] = None,  # None이면 ./out/<입력명>_ranking.txt
    expand_kx: float = 0.01,
    expand_ky: float = 0.01,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    filter_by_inout: bool = False,
    io_thr: float = 0.5,
) -> Tuple[str, str, Dict[int, int], Dict[str, int]]:
    """
    한 번의 패스로:
      - 프레임마다 obj/head 탐지 + gaze 예측
      - 관심도 누적 (시선이 포함된 obj 박스의 class에 +1)
      - 즉시 시각화 프레임 작성하여 동영상으로 저장
      - 종료 후 랭킹 계산/저장

    Returns:
        out_video_path, out_ranking_txt, interest_dict, stats
    """
    # class_names 만들기
    max_id = max(class_meta.keys()) if class_meta else -1
    class_names = [
        class_meta[i]["name"] if i in class_meta else f"id{i}"
        for i in range(max_id + 1)
    ]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    # 출력 경로 기본값
    # --- 날짜 기반 출력 파일명 생성 ---
    base_name = build_filename(date, start, end)

    # 🔹 출력 영상 파일명: 2025-10-25_18:00~19:00.mp4
    out_video_path = str(Path(video_path).with_name(f"{base_name}.avi"))

    # 🔹 랭킹 CSV 파일명: gaze-tracking_2025-10-25_18:00~19:00.csv
    out_ranking_txt = str(Path("./out") / f"gaze-tracking_{base_name}.csv")

    fourcc_in = int(cap.get(cv2.CAP_PROP_FOURCC))
    writer = cv2.VideoWriter(
        out_video_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H)
    )

    # 누적
    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # 이 실행에서 실제로 처리할 프레임 수(= stride/최대프레임 반영) 계산
    if n_frames_total > 0:
        frames_cap = (
            n_frames_total if max_frames is None else min(n_frames_total, max_frames)
        )
        total_to_process = (frames_cap + frame_stride - 1) // frame_stride
    else:
        total_to_process = None  # 길이를 모르면 None으로 두고 동적 바를 사용

    # tqdm 진행률 바 준비
    pbar = tqdm(
        total=total_to_process,
        desc="Processing",
        unit="f",
        dynamic_ncols=True,
        smoothing=0.1,
    )

    interest: Dict[int, int] = defaultdict(int)
    n_frames = n_frames_total
    n_proc = 0
    n_heads = 0
    n_hits = 0

    processed = 0
    frame_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_stride > 1 and (frame_idx % frame_stride != 1):
            continue

        # (1) 객체 탐지 + 확장
        obj_xyxy, obj_cls = run_object_detector_on_frame(
            obj_model, frame_bgr, conf=CONF_THR, iou=IOU_THR, num_classes=NUM_CLASSES
        )
        if obj_xyxy.size > 0:
            obj_xyxy = expand_xyxy_rel_img(obj_xyxy, W, H, kx=expand_kx, ky=expand_ky)

        # (2) 머리/시선
        gaze_pts, head_xyxy = get_pred_gaze_points_and_heads_from_frame(
            frame_bgr=frame_bgr,
            sharingan=sharingan,
            head_detector=head_model,
            filter_by_inout=filter_by_inout,
            io_thr=io_thr,
        )

        n_proc += 1
        n_heads += int(head_xyxy.shape[0])

        # (3) 관심도 누적
        if obj_xyxy.size > 0 and gaze_pts.size > 0:
            for pt in gaze_pts:
                idx = point_in_any_box((float(pt[0]), float(pt[1])), obj_xyxy)
                if idx is None:
                    continue
                cid = int(obj_cls[idx])
                if (NUM_CLASSES is None) or (0 <= cid < NUM_CLASSES):
                    interest[cid] += 1
                    n_hits += 1

        # (4) 시각화 프레임 기록
        vis = draw_annotated_frame(
            frame_bgr=frame_bgr,
            obj_xyxy=obj_xyxy,
            obj_cls=obj_cls,
            head_xyxy=head_xyxy,
            gaze_pts=gaze_pts,
            class_names=class_names,
            thickness_obj=2,
            thickness_head=2,
            gaze_radius=5,
        )
        writer.write(vis)

        # 진행률 업데이트  # NEW
        if pbar is not None:
            pbar.update(1)

        processed += 1
        if (max_frames is not None) and (processed >= max_frames):
            break

    cap.release()
    writer.release()

    # (5) 랭킹 계산/저장
    ranked = rank_interest_scores(interest, class_names=class_names, top_k=None)
    # 기존 write_ranking_txt 유틸 재사용
    write_ranking_txt(
        out_ranking_txt,
        ranked,
        class_meta=class_meta,
        add_unranked_from_meta=True,
        unranked_sort="id",
    )

    stats = {
        "n_frames": n_frames,
        "n_proc_frames": n_proc,
        "n_heads": n_heads,
        "n_hits": n_hits,
    }
    print(
        f"[DONE] video: {out_video_path} / ranking: {out_ranking_txt} / stats: {stats}",
        file=sys.stderr,
    )
    print(f"[INFO] AVI saved: {out_video_path}", file=sys.stderr)

    # 🔥 AVI → MP4 변환
    try:
        mp4_path = convert_avi_to_mp4(out_video_path)
        print(f"[INFO] Converted to MP4: {mp4_path}", file=sys.stderr)

        # AVI 파일 삭제 (선택)
        os.remove(out_video_path)

        # 업로드할 파일 경로를 MP4로 교체
        out_video_path = mp4_path

    except Exception as e:
        print(
            f"[WARN] FFmpeg 변환 실패 → AVI 그대로 업로드합니다: {e}", file=sys.stderr
        )

    print(
        f"[DONE] video: {out_video_path} / ranking: {out_ranking_txt} / stats: {stats}",
        file=sys.stderr,
    )

    return out_video_path, out_ranking_txt, dict(interest), stats


if __name__ == "__main__":
    # 클래스 이름
    CLASS_META = {
        0: {"name": "birak", "category": "beverage", "price": "2000원"},
        1: {"name": "sprite", "category": "beverage", "price": "1800원"},
        2: {"name": "cornChip", "category": "chip", "price": "1250원"},
        3: {"name": "cornChee", "category": "snack", "price": "1500원"},
        4: {"name": "hotShrimp", "category": "snack", "price": "1500원"},
        5: {"name": "chamCracker", "category": "cracker", "price": "1200원"},
    }

    parser = argparse.ArgumentParser(
        description="Run object/head detection + Sharingan gaze on a video once, "
        "save visualization video and ranking .txt in one pass."
    )
    # 경로 옵션
    parser.add_argument("--ckpt", type=str, help="Sharingan checkpoint path")
    parser.add_argument("--obj", type=str, help="YOLO object weights path")
    parser.add_argument("--head", type=str, help="YOLO head weights path")
    parser.add_argument("--video", type=str, help="Input video path")

    parser.add_argument(
        "--out-video", type=str, default=None, help="Output visualization video path"
    )
    parser.add_argument(
        "--out-ranking", type=str, default=None, help="Output ranking .txt path"
    )
    parser.add_argument(
        "--expand-kx",
        type=float,
        default=0.01,
        help="Object box expand ratio (relative to width)",
    )
    parser.add_argument(
        "--expand-ky",
        type=float,
        default=0.01,
        help="Object box expand ratio (relative to height)",
    )
    parser.add_argument(
        "--frame-stride", type=int, default=1, help="Sample every Nth frame"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Limit number of processed frames"
    )
    parser.add_argument(
        "--filter-inout",
        action="store_true",
        default=False,
        help="Filter gaze by in/out score",
    )
    parser.add_argument(
        "--io-thr", type=float, default=0.5, help="Threshold for in/out filtering"
    )

    # 날짜·시간 기반 출력 파일명 생성용
    parser.add_argument("--date", type=str, required=True, help="Date (YYYY-MM-DD)")
    parser.add_argument("--start", type=str, required=True, help="Start time (HH:MM)")
    parser.add_argument("--end", type=str, required=True, help="End time (HH:MM)")

    args = parser.parse_args()

    # args.ckpt = "checkpoints/best_gaze_synth_experiment_3.ckpt"
    # args.obj = "weights/detect_experiment_object.pt"
    # args.head = "weights/experiment_head.pt"
    # args.video = "./data/20250630_164710.mp4"

    # 3) 모델 로드
    obj_model = load_obj_detection_model(DEVICE, weights_path=args.obj)
    head_model = load_head_detection_model(DEVICE, weights_path=args.head)
    sharingan = load_sharingan_model(args.ckpt, DEVICE)

    # 4) 비디오 처리
    out_vid, out_txt, interest, stats = process_video_once_and_export(
        video_path=args.video,
        obj_model=obj_model,
        head_model=head_model,
        sharingan=sharingan,
        class_meta=CLASS_META,
        date=args.date,
        start=args.start,
        end=args.end,
        out_video_path=args.out_video,  # None이면 자동 경로(입력명__vis.mp4)
        out_ranking_txt=args.out_ranking,  # None이면 ./out/<입력명>_ranking.txt
        expand_kx=args.expand_kx,
        expand_ky=args.expand_ky,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        filter_by_inout=args.filter_inout,
        io_thr=args.io_thr,
    )

    # --- JSON 결과 출력 (FastAPI용) ---
    # --- Google Drive 업로드 --- #
    try:
        folder_id = "1ZRAfqwSe7vnxMqN6rlu9KcJxTmMvxMBz"
        video_link, video_view = upload_to_drive(
            out_vid, "video/mp4", folder_id=folder_id
        )
        txt_link, txt_view = upload_to_drive(out_txt, "text/csv", folder_id=folder_id)

        # 로컬 파일 정리
        os.remove(out_vid)
        os.remove(out_txt)

    except Exception as e:
        video_link = None
        txt_link = None
        print(f"[Drive Upload Error] {e}", file=sys.stderr)

    # --- JSON 결과 출력 (FastAPI용) --- #
    import json

    result = {
        "video_drive_link": video_link,
        "ranking_drive_link": txt_link,
        "interest": interest,
        "stats": stats,
    }

    sys.stdout.write(json.dumps(result, default=str))
    sys.stdout.flush()
    print(f"[UPLOAD] Video uploaded to: {video_link}", file=sys.stderr)
    print(f"[UPLOAD] Ranking uploaded to: {txt_link}", file=sys.stderr)
