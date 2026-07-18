import streamlit as st
import cv2
import numpy as np
import re
import pytesseract
from PIL import Image
import tempfile
import os
import pandas as pd
import io
from pyzbar.pyzbar import decode

st.set_page_config(page_title="Hệ thống Quét Giấy Tờ Hàng Loạt", page_icon="🪪", layout="wide")
st.title("🪪 Hệ thống Trích xuất Giấy tờ hàng loạt & Xuất Excel")
st.markdown("Hệ thống kết hợp **Giải mã QR Bộ Công An (Chính xác 100%)** và **AI Tesseract LSTM**.")

loai_giay_to = st.selectbox(
    "📁 Chọn loại giấy tờ bạn chuẩn bị tải lên hàng loạt:",
    ("Căn cước công dân / VNeID", "Hộ chiếu (Passport)")
)

# ==========================================
# KHU VỰC 1: XỬ LÝ ẢNH BẰNG OPENCV & TESSERACT (GIỮ NGUYÊN)
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
    if img is None: return None
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
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

# ==========================================
# KHU VỰC 2: GIAO DIỆN TẢI NHIỀU FILE & XỬ LÝ LẶP
# ==========================================
uploaded_files = st.file_uploader(
    "📸 Chọn hoặc kéo thả nhiều ảnh giấy tờ vào đây:", 
    type=['jpg', 'png', 'jpeg'], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"🔗 Đã ghi nhận {len(uploaded_files)} tệp ảnh sẵn sàng xử lý.")
    
    if st.button("🚀 Bắt đầu trích xuất hàng loạt", type="primary"):
        all_results = [] 
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, up_file in enumerate(uploaded_files):
            status_text.text(f"🔄 Đang xử lý tệp ({idx + 1}/{len(uploaded_files)}): {up_file.name}")
            row_data = {
                "STT": idx + 1,
                "Tên File Ảnh": up_file.name,
                "Loại Giấy Tờ": loai_giay_to,
                "Số Định Danh / Hộ Chiếu": "Không tìm thấy",
                "Họ và tên": "Không tìm thấy",
                "Ngày tháng năm sinh": "Không tìm thấy",
                "Địa chỉ thường trú / Quốc tịch": "Không tìm thấy",
                "Phương pháp xử lý": "Thất bại",
                "Văn bản thô (AI nhìn thấy)": ""
            }
            
            try:
                # Load ảnh gốc để xử lý
                original_pil_img = Image.open(up_file)
                qr_extracted = False
                
                # BƯỚC 1: CỐ GẮNG ĐỌC MÃ QR TRƯỚC (NẾU LÀ CCCD)
                if loai_giay_to == "Căn cước công dân / VNeID":
                    decoded_objects = decode(original_pil_img)
                    if decoded_objects:
                        qr_data = decoded_objects[0].data.decode('utf-8')
                        # Cấu trúc mã QR Bộ Công An: Số CCCD|Số CMND cũ|Họ tên|Ngày sinh|Giới tính|Địa chỉ|Ngày cấp
                        parts = qr_data.split('|')
                        if len(parts) >= 6:
                            row_data["Số Định Danh / Hộ Chiếu"] = parts[0]
                            row_data["Họ và tên"] = parts[2]
                            
                            # Xử lý chuỗi ngày sinh (VD: 01011990 -> 01/01/1990)
                            dob_raw = parts[3]
                            if len(dob_raw) == 8:
                                row_data["Ngày tháng năm sinh"] = f"{dob_raw[:2]}/{dob_raw[2:4]}/{dob_raw[4:]}"
                            else:
                                row_data["Ngày tháng năm sinh"] = dob_raw
                                
                            row_data["Địa chỉ thường trú / Quốc tịch"] = parts[5]
                            row_data["Phương pháp xử lý"] = "Đọc mã QR (Đúng 100%)"
                            row_data["Văn bản thô (AI nhìn thấy)"] = "Bóc tách trực tiếp từ mã hóa của BCA."
                            qr_extracted = True

                # BƯỚC 2: NẾU KHÔNG CÓ MÃ QR, TỰ ĐỘNG CHUYỂN SANG DÙNG TESSERACT OCR NHƯ CŨ
                if not qr_extracted:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                        f.write(up_file.getbuffer())
                        temp_path = f.name
                    
                    aligned_img = crop_and_align_card(temp_path)
                    if aligned_img is not None:
                        enhanced_gray_img = enhance_image_for_tesseract(aligned_img)
                        pil_img = Image.fromarray(enhanced_gray_img)
                        
                        custom_config = r'--oem 1 --psm 3'
                        raw_text = pytesseract.image_to_string(pil_img, lang='vie+eng', config=custom_config)
                        clean_raw_text = raw_text.replace('\n', ' | ').strip()
                        row_data["Văn bản thô (AI nhìn thấy)"] = clean_raw_text
                        row_data["Phương pháp xử lý"] = "Tesseract OCR"
                        
                        if loai_giay_to == "Căn cước công dân / VNeID":
                            id_match = re.search(r"(?:\D|^)(\d[\s]*){12}(?:\D|$)", raw_text)
                            if id_match: row_data["Số Định Danh / Hộ Chiếu"] = re.sub(r"\s+", "", id_match.group())

                            dob_match = re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", raw_text)
                            if dob_match: row_data["Ngày tháng năm sinh"] = dob_match.group()

                            name_match = re.search(r"(?:Họ và tên|name)[:\-\s]*([A-ZÀ-Ỹ\s]+)", raw_text, re.IGNORECASE)
                            if name_match: row_data["Họ và tên"] = name_match.group(1).strip()

                            addr_match = re.search(r"(?:thường trú|residence)[:\-\s]*([^\n]+)", raw_text, re.IGNORECASE)
                            if addr_match: row_data["Địa chỉ thường trú / Quốc tịch"] = addr_match.group(1).strip()
                        else:
                            passport_no_match = re.search(r"[A-Z][\s]*\d{7}", raw_text)
                            if passport_no_match: row_data["Số Định Danh / Hộ Chiếu"] = re.sub(r"\s+", "", passport_no_match.group())
                            
                            dob_match = re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", raw_text)
                            if dob_match: row_data["Ngày tháng năm sinh"] = dob_match.group()
                            
                            name_match = re.search(r"(?:Họ và tên|name|Họ[\s\/]*Surname)[:\-\s]*([A-ZÀ-Ỹ\s]+)", raw_text, re.IGNORECASE)
                            if name_match: row_data["Họ và tên"] = name_match.group(1).strip()
                            
                            nat_match = re.search(r"(?:Quốc tịch|Nationality)[:\-\s]*([A-ZÀ-Ỹ\s]+)", raw_text, re.IGNORECASE)
                            if nat_match: row_data["Địa chỉ thường trú / Quốc tịch"] = nat_match.group(1).strip()
                            
                    os.remove(temp_path)

                all_results.append(row_data)
                
            except Exception as e:
                row_data["Văn bản thô (AI nhìn thấy)"] = f"Lỗi: {e}"
                all_results.append(row_data)
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.text("✅ Đã hoàn thành xử lý toàn bộ danh sách tệp ảnh!")
        
        # ==========================================
        # KHU VỰC 3: HIỂN THỊ BẢNG & ĐÓNG GÓI FILE EXCEL
        # ==========================================
        if all_results:
            df = pd.DataFrame(all_results)
            st.subheader("📊 Bảng kết quả trích xuất tổng hợp")
            st.dataframe(df, use_container_width=True)
            
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='DanhSachKhachHang')
            
            st.markdown("---")
            st.subheader("📥 Tải về báo cáo Excel kết quả")
            st.download_button(
                label="📥 Bấm vào đây để tải file Excel (.xlsx)",
                data=excel_buffer.getvalue(),
                file_name="BaoCao_TrichXuat_GiayTo.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
        else:
            st.warning("Không có dữ liệu nào được trích xuất thành công.")
