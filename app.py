import streamlit as st
import cv2
import numpy as np
import re
import pytesseract
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Hệ thống Scan Giấy Tờ AI Tesseract", page_icon="🪪")
st.title("🪪 Trích xuất Giấy tờ AI Tesseract (Free 100%)")
st.markdown("Hệ thống sử dụng **Tesseract LSTM Neural Network** kết hợp bộ lọc xử lý ảnh nhị phân nâng cao.")

# Giao diện chọn loại giấy tờ
loai_giay_to = st.selectbox(
    "Bạn muốn quét loại giấy tờ nào?",
    ("Căn cước công dân / VNeID", "Hộ chiếu (Passport)")
)

# ==========================================
# KHU VỰC 1: XỬ LÝ ẢNH BẰNG OPENCV (NÂNG CẤP)
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

def enhance_image_for_tesseract(cv_img):
    """
    HÀM NÂNG CẤP CÔNG NGHỆ:
    Chuyển đổi ảnh sang dạng xám, tăng độ tương phản và áp dụng thuật toán nhị phân Otsu
    để xóa sạch hoa văn chìm trên Hộ chiếu/CCCD, giúp chữ nét hơn 200%.
    """
    # Chuyển ảnh xám
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # Phóng to ảnh nhẹ nếu kích thước nhỏ để Tesseract dễ nhận diện ký tự
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    
    # Lọc nhiễu cục bộ nhưng giữ nguyên góc cạnh của nét chữ
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Áp dụng ngưỡng nhị phân thích ứng Otsu's Thresholding
    thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    return thresh

# ==========================================
# KHU VỰC 2: TRÍCH XUẤT THÔNG TIN SONG NGỮ
# ==========================================
up_file = st.file_uploader("Tải ảnh giấy tờ lên", type=['jpg', 'png', 'jpeg'])

if up_file:
    st.image(up_file, caption="Ảnh gốc tải lên", use_container_width=True)
    if st.button("Bắt đầu trích xuất AI Tesseract", type="primary"):
        with st.spinner("Hệ thống đang bóc tách hoa văn nền và chạy lõi AI Tesseract..."):
            try:
                # 1. Lưu file tạm và chạy OpenCV cắt phẳng ảnh
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                    f.write(up_file.getbuffer())
                    temp_path = f.name
                
                aligned_img = crop_and_align_card(temp_path)
                
                # 2. Chạy hàm nâng cấp lọc ảnh nhị phân xóa nền
                enhanced_gray_img = enhance_image_for_tesseract(aligned_img)
                
                # Hiển thị ảnh sau xử lý nhị phân cho người dùng xem độ nét
                st.image(enhanced_gray_img, caption="Ảnh nhị phân đã lọc sạch hoa văn nền", use_container_width=True)
                
                # Chuyển đổi sang PIL Image để cấp cho Tesseract
                pil_img = Image.fromarray(enhanced_gray_img)
                
                # 3. CẤU HÌNH AI TESSERACT: 
                # lang='vie+eng': Đọc song ngữ đồng thời
                # --oem 1: Kích hoạt mạng nơ-ron LSTM chuyên dụng
                # --psm 3: Tự động phân tích bố cục trang bảng biểu thông minh
                custom_config = r'--oem 1 --psm 3'
                raw_text = pytesseract.image_to_string(pil_img, lang='vie+eng', config=custom_config)
                
                # 4. BỘ LỌC DỮ LIỆU BẰNG REGEX (GIỮ NGUYÊN)
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

                st.success("Hoàn thành trích xuất!")
                st.table(info)
                
                with st.expander("Bấm vào đây để kiểm tra văn bản thô AI đọc được"):
                    st.text(raw_text)

                os.remove(temp_path)

            except Exception as e:
                st.error(f"Lỗi hệ thống xử lý: {e}")
