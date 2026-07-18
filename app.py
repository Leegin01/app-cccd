import streamlit as st
import cv2
import numpy as np
import re
from PIL import Image
import tempfile
import os
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="AI Scan CCCD", page_icon="🪪")
st.title("🪪 Trích xuất thông tin CCCD/VNeID")
st.markdown("Hướng dẫn: Chụp ảnh rõ nét, đặt thẻ trên nền tối để đạt độ chính xác cao nhất.")

# --- 1. CÁC HÀM XỬ LÝ ẢNH (OPENCV) ---
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def perspective_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (width, height))

def process_card_image(image_path):
    img = cv2.imread(image_path)
    ratio = img.shape[0] / 500.0
    orig = img.copy()
    gray = cv2.cvtColor(cv2.resize(img, (int(img.shape[1]/ratio), 500)), cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 75, 200)
    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            return Image.fromarray(cv2.cvtColor(perspective_transform(orig, approx.reshape(4, 2) * ratio), cv2.COLOR_BGR2RGB))
    return Image.fromarray(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))

# --- 2. AI OCR & REGEX ---
@st.cache_resource
def load_model():
    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = 'cpu'
    return Predictor(config)

predictor = load_model()

def get_info(text):
    data = {"Số CCCD": "N/A", "Họ tên": "N/A", "Ngày sinh": "N/A", "Địa chỉ": "N/A"}
    # Tìm số
    id_match = re.search(r"\d{12}", text)
    if id_match: data["Số CCCD"] = id_match.group()
    # Tìm ngày sinh
    dob_match = re.search(r"\d{2}/\d{2}/\d{4}", text)
    if dob_match: data["Ngày sinh"] = dob_match.group()
    # Tìm tên (Chữ in hoa sau cụm từ 'Họ và tên')
    name_match = re.search(r"(?:tên|name)[:\s]+([A-ZÀ-Ỹ\s]+)", text, re.I)
    if name_match: data["Họ tên"] = name_match.group(1).strip()
    # Tìm nơi thường trú/cư trú
    addr_match = re.search(r"(?:trú|residence)[:\s]+([^\n]+)", text, re.I)
    if addr_match: data["Địa chỉ"] = addr_match.group(1).strip()
    return data

# --- 3. GIAO DIỆN CHÍNH ---
up_file = st.file_uploader("Tải ảnh thẻ lên", type=['jpg', 'png', 'jpeg'])
if up_file:
    st.image(up_file, caption="Ảnh gốc", use_container_width=True)
    if st.button("Bắt đầu trích xuất", type="primary"):
        with st.spinner("Đang xử lý..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                f.write(up_file.getbuffer())
                path = f.name
            card_img = process_card_image(path)
            st.image(card_img, caption="Ảnh đã xử lý", use_container_width=True)
            raw_text = predictor.predict(card_img)
            result = get_info(raw_text)
            st.success("Kết quả:")
            st.json(result)
            os.remove(path)