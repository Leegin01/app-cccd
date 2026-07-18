import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import json
import re
import google.generativeai as genai
import pytesseract

st.set_page_config(page_title="Hệ thống Scan Giấy Tờ Đa Động Cơ", page_icon="🪪")
st.title("🪪 Trích xuất Giấy tờ (Gemini AI & Tesseract)")
st.markdown("Hệ thống tích hợp **OpenCV** làm sạch ảnh, hỗ trợ tùy chọn động cơ AI đám mây hoặc OCR mã nguồn mở.")

# Giao diện tùy chọn Động cơ OCR
ocr_engine = st.radio(
    "⚙️ Chọn công nghệ trích xuất:",
    ("Gemini AI (Độ chính xác cao, Cần Internet & API Key)", "Tesseract OCR (Miễn phí, Tốc độ cao, Chạy nội bộ)")
)

# Lấy API Key từ "két sắt" bảo mật của Streamlit Cloud. Nếu chưa cài đặt két sắt thì để trống.
try:
    default_api_key = st.secrets["GEMINI_API_KEY"]
except:
    default_api_key = ""

api_key = ""

# Nếu chọn Gemini thì hiển thị ô nhập đã được điền sẵn Key mặc định
if ocr_engine.startswith("Gemini"):
    api_key = st.text_input("🔑 Nhập Google Gemini API Key:", value=default_api_key, type="password")

# Giao diện chọn loại giấy tờ
loai_giay_to = st.selectbox(
    "Bạn muốn quét loại giấy tờ nào?",
    ("Căn cước công dân / VNeID", "Hộ chiếu (Passport)")
)

# ==========================================
# KHU VỰC 1: XỬ LÝ ẢNH BẰNG OPENCV
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
    return orig 

# ==========================================
# KHU VỰC 2: TRÍCH XUẤT THÔNG TIN
# ==========================================
up_file = st.file_uploader("Tải ảnh giấy tờ lên", type=['jpg', 'png', 'jpeg'])

if up_file:
    st.image(up_file, caption="Ảnh gốc", use_container_width=True)
    if st.button("Bắt đầu trích xuất", type="primary"):
        
        # Bắt lỗi nếu API Key bị xóa rỗng
        if ocr_engine.startswith("Gemini") and not api_key:
            st.warning("Vui lòng nhập Gemini API Key để sử dụng mô hình này!")
        else:
            with st.spinner("Hệ thống đang tiền xử lý ảnh và trích xuất dữ liệu..."):
                try:
                    # 1. Lưu file tạm và chạy OpenCV để cắt phẳng ảnh
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                        f.write(up_file.getbuffer())
                        temp_path = f.name
                    
                    aligned_img = crop_and_align_card(temp_path)
                    st.image(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB), caption="Ảnh đã được OpenCV căn chỉnh", use_container_width=True)
                    pil_img = Image.fromarray(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
                    
                    # ---------------------------------------------------------
                    # PHÂN NHÁNH 1: SỬ DỤNG GEMINI AI
                    # ---------------------------------------------------------
                    if ocr_engine.startswith("Gemini"):
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        if loai_giay_to == "Căn cước công dân / VNeID":
                            prompt = """Bạn là một hệ thống OCR chuyên nghiệp. Hãy đọc thẻ Căn cước công dân này.
                            Trả về kết quả DUY NHẤT dưới dạng chuỗi JSON định dạng như sau:
                            {"Số CCCD": "...", "Họ và tên": "...", "Ngày sinh": "...", "Giới tính": "...", "Quê quán": "...", "Nơi thường trú": "..."}"""
                        else:
                            prompt = """Bạn là một hệ thống OCR chuyên nghiệp. Hãy đọc thông tin Hộ chiếu (Passport) này song ngữ Anh - Việt.
                            Trả về kết quả DUY NHẤT dưới dạng chuỗi JSON định dạng như sau:
                            {"Số Hộ chiếu": "...", "Họ và tên": "...", "Ngày sinh": "...", "Quốc tịch": "...", "Số CCCD": "...", "Ngày hết hạn": "..."}"""
                        
                        response = model.generate_content([prompt, pil_img])
                        raw_json = response.text.replace("```json", "").replace("```", "").strip()
                        
                        try:
                            info_dict = json.loads(raw_json)
                            st.success("Hoàn thành trích xuất bằng Gemini AI!")
                            st.table(info_dict)
                        except json.JSONDecodeError:
                            st.error("Lỗi định dạng JSON từ Gemini.")
                            st.write("Dữ liệu thô:", response.text)

                    # ---------------------------------------------------------
                    # PHÂN NHÁNH 2: SỬ DỤNG TESSERACT OCR + REGEX
                    # ---------------------------------------------------------
                    else:
                        # Đọc text thô bằng Tesseract
                        raw_text = pytesseract.image_to_string(pil_img, lang='vie')
                        info = {}
                        
                        if loai_giay_to == "Căn cước công dân / VNeID":
                            info = {"Số CCCD": "Không tìm thấy", "Ngày sinh": "Không tìm thấy", "Họ và tên": "Không tìm thấy", "Nơi thường trú": "Không tìm thấy"}
                            id_match = re.search(r"\d{12}", raw_text)
                            if id_match: info["Số CCCD"] = id_match.group()

                            dob_match = re.search(r"\d{2}/\d{2}/\d{4}", raw_text)
                            if dob_match: info["Ngày sinh"] = dob_match.group()

                            name_match = re.search(r"(?:Họ và tên|name)[:\s]*([A-ZÀ-Ỹ\s]+)", raw_text, re.IGNORECASE)
                            if name_match: info["Họ và tên"] = name_match.group(1).strip()

                            addr_match = re.search(r"(?:thường trú|residence)[:\s]*([^\n]+)", raw_text, re.IGNORECASE)
                            if addr_match: info["Nơi thường trú"] = addr_match.group(1).strip()
                        else:
                            # Hộ chiếu
                            info = {"Số Hộ chiếu": "Không tìm thấy", "Số CCCD/Định danh": "Không tìm thấy", "Ngày sinh": "Không tìm thấy", "Họ và tên": "Không tìm thấy"}
                            passport_no_match = re.search(r"[A-Z]\d{7}", raw_text)
                            if passport_no_match: info["Số Hộ chiếu"] = passport_no_match.group()
                            
                            id_match = re.search(r"\b\d{9}\b|\b\d{12}\b", raw_text)
                            if id_match: info["Số CCCD/Định danh"] = id_match.group()

                            dob_match = re.search(r"\d{2}/\d{2}/\d{4}", raw_text)
                            if dob_match: info["Ngày sinh"] = dob_match.group()
                            
                            name_match = re.search(r"(?:Họ và tên|name|Họ\/Surname)[:\s]*([A-ZÀ-Ỹ\s]+)", raw_text, re.IGNORECASE)
                            if name_match: info["Họ và tên"] = name_match.group(1).strip()

                        st.success("Hoàn thành trích xuất bằng Tesseract OCR!")
                        st.table(info)
                        
                        with st.expander("Bấm vào đây để xem dữ liệu Tesseract thô"):
                            st.text(raw_text)

                    # Dọn dẹp file tạm
                    os.remove(temp_path)

                except Exception as e:
                    st.error(f"Lỗi quá trình xử lý: {e}")
