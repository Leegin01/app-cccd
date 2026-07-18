import streamlit as st
import easyocr # Cập nhật mới
import time
import re
# ... (các import cũ giữ nguyên)

# KHỞI TẠO EASYOCR (Chạy 1 lần duy nhất)
@st.cache_resource
def load_easyocr():
    return easyocr.Reader(['vi', 'en'], gpu=False) # Gpu=False để chạy trên CPU của máy chủ Cloud

reader = load_easyocr()

# ... (Giữ nguyên các hàm encode_image, crop, align cũ)

# TRONG VÒNG LẶP XỬ LÝ (Thay thế đoạn LỚP 3 Tesseract cũ bằng đoạn này):
                        
                        # 2. KHỞI CHẠY LỚP 3 (EASYOCR HOẶC TESSERACT) NẾU CẦN THIẾT
                        if use_tesseract:
                            status_text.text(f"🚀 Chuyển sang EasyOCR cho tệp {up_file.name}...")
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                                f.write(up_file.getbuffer())
                                temp_path = f.name
                            
                            # Đọc chữ bằng EasyOCR
                            results = reader.readtext(temp_path, detail=0)
                            raw_text = "\n".join(results)
                            
                            # Phân tích kết quả
                            extracted_data = {
                                "Loại mặt": "Mặt trước/sau (EasyOCR)",
                                "Số Định Danh / Hộ Chiếu": "Không tìm thấy",
                                "Họ và tên": "Không tìm thấy",
                                "Ngày tháng năm sinh": "Không tìm thấy",
                                "Địa chỉ thường trú / Quốc tịch": "Không tìm thấy",
                                "Ngày cấp": "Không tìm thấy",
                                "Đặc điểm nhân dạng": "Bóc tách EasyOCR"
                            }
                            
                            # Logic lọc dữ liệu từ raw_text của EasyOCR
                            id_matches = re.findall(r"\d{12}", raw_text)
                            if id_matches: extracted_data["Số Định Danh / Hộ Chiếu"] = id_matches[0]
                            
                            # (Các đoạn regex bóc tách Họ tên, NS tương tự Tesseract...)
                            
                            os.remove(temp_path)
