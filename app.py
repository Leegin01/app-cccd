import streamlit as st
import cv2
import numpy as np
import re
import pytesseract
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Hệ thống Scan CCCD (Free 100%)", page_icon="🪪")
st.title("🪪 Trích xuất thông tin CCCD hoàn toàn miễn phí")
st.markdown("Hệ thống sử dụng **Tesseract OCR**, không giới hạn số lượt quét.")

# ==========================================
# KHU VỰC 1: XỬ LÝ ẢNH CẮT PHẲNG THẺ
# ==========================================
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

def crop_and_align_card(image_path):
    img = cv2.imread(image_path)
    ratio = img.shape[0] / 500.0
    orig = img.copy()
    img_resized = cv2.resize(img, (int(img.shape[1]/ratio), 500))
    
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    
    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    for c in cnts:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            return perspective_transform(orig, approx.reshape(4, 2) * ratio)
    return orig # Trả về ảnh gốc nếu không tìm thấy viền

# ==========================================
# KHU VỰC 2: GIAO DIỆN VÀ XỬ LÝ CHÍNH
# ==========================================
up_file = st.file_uploader("Tải ảnh thẻ lên", type=['jpg', 'png', 'jpeg'])

if up_file:
    st.image(up_file, caption="Ảnh gốc", use_container_width=True)
    if st.button("Bắt đầu trích xuất", type="primary"):
        with st.spinner("Đang cắt ảnh và đọc chữ..."):
            try:
                # 1. Lưu file tạm để OpenCV xử lý
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                    f.write(up_file.getbuffer())
                    temp_path = f.name
                
                # 2. Xử lý cắt phẳng thẻ
                aligned_img = crop_and_align_card(temp_path)
                
                # Hiển thị ảnh sau khi đã cắt viền
                st.image(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB), caption="Ảnh đã tự động căn chỉnh", use_container_width=True)
                
                # 3. Dùng Tesseract đọc chữ (Sử dụng gói tiếng Việt 'vie')
                # Chuyển ảnh sang dạng PIL để Tesseract đọc tốt hơn
                pil_img = Image.fromarray(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
                raw_text = pytesseract.image_to_string(pil_img, lang='vie')
                
                # 4. Trích xuất thông tin bằng Regex
                info = {
                    "Số CCCD": "Không tìm thấy",
                    "Ngày sinh": "Không tìm thấy",
                    "Họ và tên": "Không tìm thấy",
                    "Nơi thường trú": "Không tìm thấy"
                }

                id_match = re.search(r"\d{12}", raw_text)
                if id_match: info["Số CCCD"] = id_match.group()

                dob_match = re.search(r"\d{2}/\d{2}/\d{4}", raw_text)
                if dob_match: info["Ngày sinh"] = dob_match.group()

                name_match = re.search(r"(?:Họ và tên|name)[:\s]*([A-ZÀ-Ỹ\s]+)", raw_text, re.IGNORECASE)
                if name_match: info["Họ và tên"] = name_match.group(1).strip()

                addr_match = re.search(r"(?:thường trú|residence)[:\s]*([^\n]+)", raw_text, re.IGNORECASE)
                if addr_match: info["Nơi thường trú"] = addr_match.group(1).strip()

                st.success("Kết quả trích xuất:")
                st.table(info)
                
                # Xem text thô để kiểm tra độ chính xác
                with st.expander("Bấm vào đây để xem toàn bộ chữ AI đọc được (Text thô)"):
                    st.text(raw_text)

                # Dọn dẹp file tạm
                os.remove(temp_path)

            except Exception as e:
                st.error(f"Lỗi hệ thống: {e}")
