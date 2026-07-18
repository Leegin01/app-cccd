import streamlit as st
import google.generativeai as genai
from PIL import Image
import pandas as pd
import io
import json
from pyzbar.pyzbar import decode
import cv2
import numpy as np
import time  
import pytesseract
import re
import tempfile
import os

st.set_page_config(page_title="Hệ thống Scan Giấy Tờ Đa Lớp", page_icon="🪪", layout="wide")
st.title("🪪 Hệ thống Trích xuất & So khớp CCCD Tự Động (Dual-AI)")
st.markdown("Kiến trúc 3 lớp: **Quét QR Code** ➔ **Gemini 3.5 Flash** ➔ Dự phòng tự động bằng **Tesseract AI**.")

# CẤU HÌNH BẢO MẬT API KEY (CHẠY NGẦM):
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = ""

loai_giay_to = st.selectbox(
    "📁 Chọn loại giấy tờ bạn chuẩn bị tải lên hàng loạt:",
    ("Căn cước công dân / VNeID (Quét hỗn hợp cả 2 mặt)", "Hộ chiếu (Passport - Việt Nam & Nước ngoài)")
)

uploaded_files = st.file_uploader(
    "📸 Chọn hoặc kéo thả cùng lúc nhiều ảnh giấy tờ (Thoải mái thả cả mặt trước và mặt sau):", 
    type=['jpg', 'png', 'jpeg'], 
    accept_multiple_files=True
)

# ==========================================
# CÁC HÀM XỬ LÝ ẢNH DÀNH CHO LỚP TESSERACT AI
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
# GIAO DIỆN CHÍNH & VÒNG LẶP XỬ LÝ
# ==========================================
if uploaded_files:
    st.info(f"🔗 Đã ghi nhận {len(uploaded_files)} tệp ảnh sẵn sàng xử lý.")
    
    if st.button("🚀 Bắt đầu trích xuất & So khớp dữ liệu", type="primary"):
        if not api_key:
            st.error("🚨 Không tìm thấy API Key! Vui lòng cấu hình GEMINI_API_KEY trong phần Secrets của Streamlit Cloud.")
        else:
            genai.configure(api_key=api_key)
            
            # 1. ĐÃ CẬP NHẬT MÔ HÌNH GEMINI 3.5 FLASH (Mô hình chính)
            model = genai.GenerativeModel('gemini-3.5-flash')
            
            matched_database = {}
            unmatched_records = [] 
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, up_file in enumerate(uploaded_files):
                status_text.text(f"🔍 Đang phân tích tệp ({idx + 1}/{len(uploaded_files)}): {up_file.name}")
                
                try:
                    pil_img = Image.open(up_file)
                    qr_extracted = False
                    extracted_data = {}
                    
                    # ---------------------------------------------------------
                    # LỚP 1: THỬ QUÉT MÃ QR TRƯỚC (CHÍNH XÁC 100%)
                    # ---------------------------------------------------------
                    if "Căn cước" in loai_giay_to:
                        decoded_objects = decode(pil_img)
                        if decoded_objects:
                            qr_data = decoded_objects[0].data.decode('utf-8')
                            parts = qr_data.split('|')
                            if len(parts) >= 6:
                                extracted_data = {
                                    "Loại mặt": "Mặt trước",
                                    "Số Định Danh / Hộ Chiếu": parts[0],
                                    "Họ và tên": parts[2].upper(),
                                    "Ngày tháng năm sinh": f"{parts[3][:2]}/{parts[3][2:4]}/{parts[3][4:]}" if len(parts[3]) == 8 else parts[3],
                                    "Địa chỉ thường trú / Quốc tịch": parts[5],
                                    "Ngày cấp": parts[6] if len(parts) > 6 else "Không tìm thấy",
                                    "Đặc điểm nhân dạng": "Nằm ở mặt sau"
                                }
                                qr_extracted = True
                                time.sleep(1)
                    
                    # ---------------------------------------------------------
                    # LỚP 2 & 3: GEMINI 3.5 FLASH VÀ ĐỘNG CƠ DỰ PHÒNG TESSERACT
                    # ---------------------------------------------------------
                    if not qr_extracted:
                        if "Căn cước" in loai_giay_to:
                            prompt = """
                            Bạn là một hệ thống AI OCR nghiệp vụ khách sạn. Hãy phân tích hình ảnh Căn cước công dân/VNeID này.
                            Nếu là MẶT SAU, hãy tìm SỐ CCCD GỒM 12 CHỮ SỐ ở dải mã vạch MRZ.
                            Trả về kết quả DUY NHẤT dưới dạng một chuỗi JSON:
                            {
                                "Loại mặt": "Điền 'Mặt trước' hoặc 'Mặt sau'",
                                "Số Định Danh / Hộ Chiếu": "12 số CCCD",
                                "Họ và tên": "Họ tên IN HOA (mặt sau ghi Không tìm thấy)",
                                "Ngày tháng năm sinh": "DD/MM/YYYY (mặt sau ghi Không tìm thấy)",
                                "Địa chỉ thường trú / Quốc tịch": "Địa chỉ cụ thể (mặt sau ghi Không tìm thấy)",
                                "Ngày cấp": "DD/MM/YYYY",
                                "Đặc điểm nhân dạng": "Đặc điểm nhận dạng"
                            }
                            """
                        else:
                            prompt = """
                            Hãy đọc hình ảnh Hộ chiếu này và trả về chuỗi JSON:
                            {
                                "Loại mặt": "Hộ chiếu",
                                "Số Định Danh / Hộ Chiếu": "Mã số hộ chiếu",
                                "Họ và tên": "Họ và tên IN HOA",
                                "Ngày tháng năm sinh": "DD/MM/YYYY",
                                "Địa chỉ thường trú / Quốc tịch": "Tên quốc gia / Quốc tịch",
                                "Ngày cấp": "Ngày cấp DD/MM/YYYY",
                                "Đặc điểm nhân dạng": "Không áp dụng"
                            }
                            """
                        
                        use_tesseract = False # Biến cờ hiệu kích hoạt động cơ dự phòng
                        
                        # Cố gắng gọi Lớp 2 (Gemini 3.5 Flash)
                        try:
                            response = model.generate_content([prompt, pil_img])
                            clean_json = response.text.strip().replace("```json", "").replace("```", "").strip()
                            extracted_data = json.loads(clean_json)
                        except Exception as api_err:
                            error_msg = str(api_err).lower()
                            # NẾU GEMINI BỊ NGHẼN MẠNG (429) HOẶC LỖI -> LẬP TỨC KÍCH HOẠT LỚP 3 (TESSERACT)
                            status_text.text(f"⚠️ Gemini bận hoặc quá tải. Đang tự động chuyển sang Tesseract AI cho tệp {up_file.name}...")
                            use_tesseract = True
                        
                        # 2. KHỞI CHẠY LỚP 3 (TESSERACT AI) NẾU CẦN THIẾT
                        if use_tesseract:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                                f.write(up_file.getbuffer())
                                temp_path = f.name
                            
                            aligned_img = crop_and_align_card(temp_path)
                            if aligned_img is not None:
                                enhanced_img = enhance_image_for_tesseract(aligned_img)
                                tess_img = Image.fromarray(enhanced_img)
                                
                                raw_text = pytesseract.image_to_string(tess_img, lang='vie+eng', config=r'--oem 1 --psm 3')
                                
                                # Phân tích bằng Regex
                                extracted_data = {
                                    "Loại mặt": "Không xác định (Tesseract)",
                                    "Số Định Danh / Hộ Chiếu": "Không tìm thấy",
                                    "Họ và tên": "Không tìm thấy",
                                    "Ngày tháng năm sinh": "Không tìm thấy",
                                    "Địa chỉ thường trú / Quốc tịch": "Không tìm thấy",
                                    "Ngày cấp": "Không tìm thấy",
                                    "Đặc điểm nhân dạng": "Bóc tách dự phòng"
                                }
                                
                                if "Căn cước" in loai_giay_to:
                                    id_match = re.search(r"(?:\D|^)(\d[\s]*){12}(?:\D|$)", raw_text)
                                    if id_match: extracted_data["Số Định Danh / Hộ Chiếu"] = re.sub(r"\s+", "", id_match.group())
                                    dob_match = re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", raw_text)
                                    if dob_match: extracted_data["Ngày tháng năm sinh"] = dob_match.group()
                                    name_match = re.search(r"(?:Họ và tên|name)[:\-\s]*([A-ZÀ-Ỹ\s]+)", raw_text, re.IGNORECASE)
                                    if name_match: extracted_data["Họ và tên"] = name_match.group(1).strip()
                                else:
                                    passport_no_match = re.search(r"[A-Z][\s]*\d{7}", raw_text)
                                    if passport_no_match: extracted_data["Số Định Danh / Hộ Chiếu"] = re.sub(r"\s+", "", passport_no_match.group())
                                    dob_match = re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", raw_text)
                                    if dob_match: extracted_data["Ngày tháng năm sinh"] = dob_match.group()
                                    name_match = re.search(r"(?:Họ và tên|name|Họ[\s\/]*Surname)[:\-\s]*([A-ZÀ-Ỹ\s]+)", raw_text, re.IGNORECASE)
                                    if name_match: extracted_data["Họ và tên"] = name_match.group(1).strip()
                                    
                            os.remove(temp_path)

                        time.sleep(1) # Giãn cách nhẹ 1 giây để an toàn bộ nhớ

                    # ---------------------------------------------------------
                    # THAO TÁC 3: THUẬT TOÁN SO KHỚP & ĐỒNG BỘ HAI MẶT
                    # ---------------------------------------------------------
                    id_key = extracted_data.get("Số Định Danh / Hộ Chiếu", "Không tìm thấy")
                    
                    if id_key == "Không tìm thấy" or len(id_key) < 5:
                        extracted_data["Tên File Gốc"] = up_file.name
                        unmatched_records.append(extracted_data)
                    else:
                        if id_key in matched_database:
                            current_record = matched_database[id_key]
                            current_record["Tên File Mặt Sau"] = up_file.name if "Mặt sau" in extracted_data.get("Loại mặt", "") else current_record.get("Tên File Mặt Sau", "Chưa quét")
                            if "Mặt trước" in extracted_data.get("Loại mặt", ""):
                                current_record["Tên File Mặt Trước"] = up_file.name
                            
                            for field in ["Họ và tên", "Ngày tháng năm sinh", "Địa chỉ thường trú / Quốc tịch", "Ngày cấp", "Đặc điểm nhân dạng"]:
                                if extracted_data.get(field) != "Không tìm thấy" and extracted_data.get(field) != "":
                                    current_record[field] = extracted_data.get(field)
                                    
                            current_record["Trạng Thái So Khớp"] = "✅ Khớp Thành Công 2 Mặt"
                        else:
                            matched_database[id_key] = {
                                "Số Định Danh / Hộ Chiếu": id_key,
                                "Tên File Mặt Trước": up_file.name if "Mặt trước" in extracted_data.get("Loại mặt", "") else "Chưa quét",
                                "Tên File Mặt Sau": up_file.name if "Mặt sau" in extracted_data.get("Loại mặt", "") else "Chưa quét",
                                "Họ và tên": extracted_data.get("Họ và tên", "Không tìm thấy"),
                                "Ngày tháng năm sinh": extracted_data.get("Ngày tháng năm sinh", "Không tìm thấy"),
                                "Địa chỉ thường trú / Quốc tịch": extracted_data.get("Địa chỉ thường trú / Quốc tịch", "Không tìm thấy"),
                                "Ngày cấp": extracted_data.get("Ngày cấp", "Không tìm thấy"),
                                "Đặc điểm nhân dạng": extracted_data.get("Đặc điểm nhân dạng", "Không tìm thấy"),
                                "Trạng Thái So Khớp": "⚠️ Chỉ mới quét 1 mặt"
                            }
                            
                except Exception as e:
                    st.error(f"Lỗi hệ thống khi xử lý tệp {up_file.name}: {e}")
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("✅ Đã hoàn thành phân tích toàn bộ dữ liệu!")
            
            # ---------------------------------------------------------
            # KHU VỰC ĐÓNG GÓI EXCEL
            # ---------------------------------------------------------
            final_rows = []
            stt = 1
            
            for key, data in matched_database.items():
                data["STT"] = stt
                final_rows.append(data)
                stt += 1
                
            for err_data in unmatched_records:
                final_rows.append({
                    "STT": stt,
                    "Số Định Danh / Hộ Chiếu": "Lỗi/Không tìm thấy ID",
                    "Tên File Mặt Trước": err_data.get("Tên File Gốc", "Lỗi"),
                    "Tên File Mặt Sau": "Lỗi đọc dữ liệu hình ảnh",
                    "Họ và tên": err_data.get("Họ và tên", "Mờ chữ"),
                    "Ngày tháng năm sinh": err_data.get("Ngày tháng năm sinh", "-"),
                    "Địa chỉ thường trú / Quốc tịch": err_data.get("Địa chỉ thường trú / Quốc tịch", "-"),
                    "Ngày cấp": err_data.get("Ngày cấp", "-"),
                    "Đặc điểm nhân dạng": err_data.get("Đặc điểm nhân dạng", "-"),
                    "Trạng Thái So Khớp": "❌ Lỗi trích xuất"
                })
                stt += 1
                
            if final_rows:
                df = pd.DataFrame(final_rows)
                column_order = ["STT", "Số Định Danh / Hộ Chiếu", "Họ và tên", "Ngày tháng năm sinh", 
                                "Địa chỉ thường trú / Quốc tịch", "Ngày cấp", "Đặc điểm nhân dạng", 
                                "Tên File Mặt Trước", "Tên File Mặt Sau", "Trạng Thái So Khớp"]
                df = df[column_order]
                
                st.subheader("📊 Bảng kết quả cấu trúc và đối sánh song biên")
                st.dataframe(df, use_container_width=True)
                
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='LưuTrúKháchHàng')
                
                st.markdown("---")
                st.subheader("📥 Tải về báo cáo tổng hợp")
                st.download_button(
                    label="📥 Bấm vào đây để tải file Excel đối sánh (.xlsx)",
                    data=excel_buffer.getvalue(),
                    file_name="BaoCao_SoKhop_DualAI.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )
