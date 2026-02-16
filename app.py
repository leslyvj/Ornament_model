import streamlit as st
import os
import json
import cv2
import base64
import requests
import numpy as np
import time
import tempfile
import pandas as pd
from dotenv import load_dotenv
from PIL import Image

# ─── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Jewelry Damage Detector", page_icon="💎", layout="wide")

# Load .env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(dotenv_path=env_path)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Class names
CLASS_NAMES = {
    0: "Bangle", 1: "Chain", 2: "Gold Bangle", 3: "Gold Bangle (Hinged)",
    4: "Gold Bangle (Plain)", 5: "Gold Bracelet", 6: "Gold Stud Earrings (Pair)",
    7: "Mangalsutra", 8: "Necklace", 9: "Ring", 10: "damage", 11: "missing_gem"
}

# ─── Pipeline Functions ──────────────────────────────────────────────────────
def create_ultra_grid(crops, grid_cols=4):
    if not crops:
        return None
    cell_h, cell_w = 512, 512
    resized_crops = [cv2.resize(c, (cell_w, cell_h)) for c in crops]
    rows = []
    for i in range(0, len(resized_crops), grid_cols):
        row_batch = resized_crops[i:i + grid_cols]
        while len(row_batch) < grid_cols:
            row_batch.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))
        rows.append(np.hstack(row_batch))
    return np.vstack(rows)


def analyze_grid_openrouter(grid_img, item_metadata):
    _, buffer = cv2.imencode('.jpg', grid_img)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    hints = "\n".join([f"- {m['itemId']}: Likely {m['detected_class']}" for m in item_metadata])
    ids_str = ", ".join([m["itemId"] for m in item_metadata])

    prompt = f"""
    The image contains a high-resolution grid of jewelry items. 
    Analyze each item sequentially (left-to-right, top-to-bottom).
    Provide metadata for these IDs: [{ids_str}].
    
    Context Hints from Detection:
    {hints}

    Rules (BE EXTREMELY STRICT):
    1. MISSING STONES: Look for empty sockets, holes, or gaps where a gem/stone should be. If found, add 'Missing gems' to defectsFound and set status to 'Fail'.
    2. DEFORMATION: If an item is bent, twisted, or broken, add 'Deformed' to defectsFound and set status to 'Fail'.
    3. STONE COUNT: Count ALL visible stones carefully.
    4. WEIGHT: Estimate gross weight realistically (e.g., 20.0g-40.0g for necklaces, 10.0g-20.0g for bangles).

    Return JSON list of objects only:
    [{{"itemId": "ID", "classification": "Necklace/Bangle/Ring/etc", "purityMarking": "22K/18K/Unverifiable", "huid": "6-digit code or Unverifiable", "defectsFound": [], "stoneCount": int, "estGrossWeight": float, "estNetWeight": float, "status": "Pass/Fail"}}]
    """

    print(f"🚀 Calling Gemini API for {len(item_metadata)} items...")
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Jewelry Damage Detector",
        },
        data=json.dumps({
            "model": "google/gemini-2.0-flash-lite-001",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}],
            "response_format": {"type": "json_object"}
        })
    )
    print(f"📡 Gemini Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        content = result['choices'][0]['message']['content']
        print(f"📝 Gemini Response: {content[:200]}...") # Log first 200 chars
        data = json.loads(content)
        if isinstance(data, dict) and 'featureData' in data:
            return data['featureData']
        elif isinstance(data, list):
            return data
        elif isinstance(data, dict):
            for key in data:
                if isinstance(data[key], list):
                    return data[key]
        return [data] if isinstance(data, dict) else data
    else:
        print(f"❌ Gemini Error: {response.text}")
        return None


def deduplicate_boxes(boxes, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    boxes = sorted(boxes, key=lambda x: x['conf'], reverse=True)
    keep = []
    for box in boxes:
        is_dup = False
        for kept in keep:
            x1 = max(box['xyxy'][0], kept['xyxy'][0])
            y1 = max(box['xyxy'][1], kept['xyxy'][1])
            x2 = min(box['xyxy'][2], kept['xyxy'][2])
            y2 = min(box['xyxy'][3], kept['xyxy'][3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            a1 = (box['xyxy'][2] - box['xyxy'][0]) * (box['xyxy'][3] - box['xyxy'][1])
            a2 = (kept['xyxy'][2] - kept['xyxy'][0]) * (kept['xyxy'][3] - kept['xyxy'][1])
            if inter / (a1 + a2 - inter + 1e-6) > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(box)
    return keep


def draw_detections(img, clean_boxes):
    """Draw bounding boxes and labels on the image."""
    annotated = img.copy()
    
    # Color palette for different classes
    COLORS = [
        (255, 107, 107), (78, 205, 196), (255, 195, 0), (106, 76, 147),
        (0, 168, 255), (255, 140, 66), (153, 204, 0), (255, 87, 51),
        (0, 191, 165), (100, 149, 237), (255, 105, 180), (50, 205, 50)
    ]
    
    for i, box in enumerate(clean_boxes):
        x1, y1, x2, y2 = map(int, box['xyxy'])
        cls_id = box['cls_id']
        conf = box['conf']
        label = f"{CLASS_NAMES.get(cls_id, 'Jewelry')} {conf:.0%}"
        color = COLORS[cls_id % len(COLORS)]
        
        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        
        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Convert BGR to RGB for display
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def run_pipeline(image_path, model_choice):
    if model_choice == "RT-DETR":
        from ultralytics import RTDETR
        # Use relative path for deployment
        model_path = os.path.join(os.path.dirname(__file__), "models", "rtdetr_best.pt")
        model = RTDETR(model_path)
        results = model(image_path, conf=0.25)
    else:
        from ultralytics import YOLO
        # Use relative path for deployment
        model_path = os.path.join(os.path.dirname(__file__), "models", "yolo_best.pt")
        model = YOLO(model_path)
        results = model(image_path, conf=0.05)

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    all_boxes = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id <= 9:
            all_boxes.append({
                'cls_id': cls_id,
                'conf': float(box.conf[0]),
                'xyxy': box.xyxy[0].tolist()
            })

    clean_boxes = deduplicate_boxes(all_boxes, iou_threshold=0.5)
    
    # Draw detections on image
    annotated_img = draw_detections(img, clean_boxes)

    item_metadata = []
    raw_crops = []

    for i, box in enumerate(clean_boxes):
        x1, y1, x2, y2 = map(int, box['xyxy'])
        pad_h = int((y2 - y1) * 0.1)
        pad_w = int((x2 - x1) * 0.1)
        y1_pad, y2_pad = max(0, y1 - pad_h), min(h, y2 + pad_h)
        x1_pad, x2_pad = max(0, x1 - pad_w), min(w, x2 + pad_w)

        crop = img[y1_pad:y2_pad, x1_pad:x2_pad]
        raw_crops.append(crop)
        item_metadata.append({
            "itemId": f"ornament_{len(raw_crops):02d}",
            "detected_class": CLASS_NAMES.get(box['cls_id'], "Jewelry")
        })

    if not raw_crops:
        return [], 0, annotated_img

    grid_img = create_ultra_grid(raw_crops)
    feature_data = analyze_grid_openrouter(grid_img, item_metadata)

    if feature_data and isinstance(feature_data, list):
        return feature_data, len(clean_boxes), annotated_img
    else:
        fallback = [{"itemId": m["itemId"], "classification": m["detected_class"], "status": "Manual Check Required"} for m in item_metadata]
        return fallback, len(clean_boxes), annotated_img


# ─── UI ──────────────────────────────────────────────────────────────────────
st.title("💎 Jewelry Damage Detector")
st.caption("Upload an image of jewelry to detect items and analyze for defects using AI")

st.divider()

col1, col2 = st.columns([3, 1])

with col2:
    model_choice = st.selectbox("Detection Model", ["RT-DETR", "YOLOv11"], index=0)
    if model_choice == "RT-DETR":
        st.info("🔬 **RT-DETR** — Transformer-based detector with global context understanding.")
    else:
        st.info("⚡ **YOLOv11** — Fast single-stage detector optimized for speed.")

with col1:
    uploaded_file = st.file_uploader("Upload Jewelry Image", type=["jpg", "jpeg", "png", "webp"])

    uploaded_bytes = None
    if uploaded_file:
        uploaded_bytes = uploaded_file.getvalue()
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

analyze_clicked = st.button("🔍 Analyze Image", type="primary")

# ─── Analysis ────────────────────────────────────────────────────────────────
if analyze_clicked:
    if not uploaded_bytes:
        st.warning("Please upload an image first.")
    elif not OPENROUTER_API_KEY:
        st.error("Missing OPENROUTER_API_KEY in .env file.")
    else:
        with st.spinner(f"Analyzing with {model_choice}..."):
            start = time.time()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_bytes)
                tmp_path = tmp.name

            try:
                results, num_detections, annotated_img = run_pipeline(tmp_path, model_choice)
                elapsed = time.time() - start

                st.divider()

                # ─── Stats ────────────────────────────────────────
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Items Detected", num_detections)
                passed = sum(1 for r in results if str(r.get('status', '')).lower() == 'pass')
                c2.metric("Passed", passed)
                failed = sum(1 for r in results if str(r.get('status', '')).lower() == 'fail')
                c3.metric("Failed", failed)
                c4.metric("Pipeline Time", f"{elapsed:.1f}s")

                # ─── Detection Output ─────────────────────────────
                st.subheader(f"🔎 {model_choice} Detection Output")
                st.image(annotated_img, caption=f"{model_choice} — {num_detections} items detected", use_container_width=True)

                # ─── Gemini Results Table ─────────────────────────
                st.subheader("✅ Gemini Analysis Results")

                # Build DataFrame from results
                table_data = []
                for item in results:
                    defects = item.get("defectsFound", [])
                    defects_str = ", ".join(defects) if defects else "None"
                    table_data.append({
                        "Item ID": item.get("itemId", "-"),
                        "Class": item.get("classification", "-"),
                        "Purity": item.get("purityMarking", "Unverifiable"),
                        "HUID": item.get("huid", "Unverifiable"),
                        "Defects": defects_str,
                        "Stones": item.get("stoneCount", 0),
                        "Gross Wt. (g)": f"{item.get('estGrossWeight', 0):.2f}",
                        "Net Wt. (g)": f"{item.get('estNetWeight', 0):.2f}",
                        "Status": item.get("status", "Manual Check Required")
                    })

                df = pd.DataFrame(table_data)

                def color_status(val):
                    if val == "Pass":
                        return "background-color: #dcfce7; color: #166534; font-weight: 600"
                    elif val == "Fail":
                        return "background-color: #fee2e2; color: #991b1b; font-weight: 600"
                    else:
                        return "background-color: #fef3c7; color: #92400e; font-weight: 600"

                styled_df = df.style.map(color_status, subset=["Status"])
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

                st.caption("A list of all items identified and analyzed from the image.")

                # ─── JSON Download ────────────────────────────────
                st.download_button(
                    "📥 Download JSON Results",
                    data=json.dumps({"featureData": results}, indent=2),
                    file_name="jewelry_analysis.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"Pipeline error: {e}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                os.unlink(tmp_path)
